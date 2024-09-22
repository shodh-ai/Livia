# Copyright (c) 2024, EleutherAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM nvcr.io/nvidia/pytorch:24.02-py3

ENV DEBIAN_FRONTEND=noninteractive

# metainformation
LABEL org.opencontainers.image.version = "1.0"
LABEL org.opencontainers.image.authors = "tejas@inferq.com"
# LABEL org.opencontainers.image.source = "https://www.github.com/eleutherai/gpt-neox"
LABEL org.opencontainers.image.licenses = " Apache-2.0"
LABEL org.opencontainers.image.base.name="nvcr.io/nvidia/pytorch:24.02-py3"

#### System package (uses default Python 3 version in Ubuntu 20.04)

RUN apt-get update -y 
RUN apt-get install -y git htop iotop iftop nano unzip sudo pdsh tmux 
RUN apt-get install -y zstd software-properties-common build-essential autotools-dev 
RUN apt-get install -y cmake g++ gcc
RUN apt-get install -y curl wget less ca-certificates ssh
RUN apt-get install -y rsync iputils-ping net-tools libcupti-dev libmlx4-1 infiniband-diags ibutils ibverbs-utils
RUN apt-get install -y rdmacm-utils perftest rdma-core
RUN pip install --upgrade pip
RUN pip install gpustat


### SSH
RUN mkdir /var/run/sshd && \
    # Prevent user being kicked off after login
    sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd && \
    echo 'AuthorizedKeysFile     .ssh/authorized_keys' >> /etc/ssh/sshd_config && \
    echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config && \
    # FIX SUDO BUG: https://github.com/sudo-project/sudo/issues/42
    echo "Set disable_coredump false" >> /etc/sudo.conf

# Expose SSH port
EXPOSE 22

# Needs to be in docker PATH if compiling other items & bashrc PATH (later)
ENV PATH=/usr/local/mpi/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:${LD_LIBRARY_PATH}

# Create a wrapper for OpenMPI to allow running as root by default
RUN mv /usr/local/mpi/bin/mpirun /usr/local/mpi/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/local/mpi/bin/mpirun && \
    echo 'mpirun.real --allow-run-as-root --prefix /usr/local/mpi "$@"' >> /usr/local/mpi/bin/mpirun && \
    chmod a+x /usr/local/mpi/bin/mpirun

### User account
RUN useradd --create-home --uid 1000 --shell /bin/bash tejas && \
    usermod -aG sudo tejas && \
    echo "tejas ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# SSH config and bashrc
RUN mkdir -p /home/tejas/.ssh /job && \
    echo 'Host *' > /home/tejas/.ssh/config && \
    echo '    StrictHostKeyChecking no' >> /home/tejas/.ssh/config && \
    echo 'export PDSH_RCMD_TYPE=ssh' >> /home/tejas/.bashrc && \
    echo 'export PATH=/home/tejas/.local/bin:$PATH' >> /home/tejas/.bashrc && \
    echo 'export PATH=/usr/local/mpi/bin:$PATH' >> /home/tejas/.bashrc && \
    echo 'export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:$LD_LIBRARY_PATH' >> /home/tejas/.bashrc

#### Python packages
# RUN pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117 && pip cache purge
COPY requirements/requirements.txt .
COPY requirements/requirements-wandb.txt .
COPY requirements/requirements-onebitadam.txt .
COPY requirements/requirements-sparseattention.txt .
COPY requirements/requirements-flashattention.txt .
COPY requirements/requirements-tensorboard.txt .
COPY requirements/requirements-s3.txt .
COPY requirements/requirements-mamba.txt .
RUN pip install -r requirements.txt && pip install -r requirements-onebitadam.txt
RUN pip install -r requirements-sparseattention.txt
RUN pip install -r requirements-mamba.txt
RUN pip install -r requirements-flashattention.txt
RUN pip install -r requirements-wandb.txt
RUN pip install -r requirements-tensorboard.txt
RUN pip install -r requirements-s3.txt
RUN pip install megablocks
RUN pip install protobuf==3.20.*
# RUN pip cache purge

## Install APEX
RUN pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git@a651e2c24ecf97cbf367fd3f330df36760e1c597

COPY megatron/fused_kernels/ megatron/fused_kernels
RUN python megatron/fused_kernels/setup.py install

# Clear staging
RUN mkdir -p /tmp && chmod 0777 /tmp

RUN pip install triton --upgrade 
RUN pip install transformers --upgrade 

# #### SWITCH TO mchorse USER
USER tejas
WORKDIR /home/tejas
