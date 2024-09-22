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

"""
Assumes a dataset of jsonl files in the same format as the neox training set.
"""

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from tokenizers.normalizers import NFKC


from glob import glob
import os
import json
import argparse
import zstandard as zstd

def load_jsonl(input_path, quiet=True) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.rstrip("\n|\r")))
    if not quiet:
        print("Loaded {} records from {}".format(len(data), input_path))
    return data

# def load_jsonl_zst(input_path, quiet=True) -> list:
#     """
#     Read list of objects from a compressed JSON lines file (.jsonl.zst).
#     """
#     data = []
#     dctx = zstd.ZstdDecompressor()
#     with open(input_path, "rb") as f:
#         with dctx.stream_reader(f) as reader:
#             for line in reader:
#                 data.append(json.loads(line.decode('utf-8').rstrip("\n|\r")))
#     if not quiet:
#         print("Loaded {} records from {}".format(len(data), input_path))
#     return data

# def load_jsonl_zst(input_path, quiet=True) -> list:
#     """
#     Read list of objects from a compressed JSON lines file (.jsonl.zst).
#     """
#     data = []
#     dctx = zstd.ZstdDecompressor()
#     out = open("print_out.txt", "a")
#     print("Processing file : {}".format(input_path), file=out)
#     with open(input_path, "rb") as f:
#         with dctx.stream_reader(f) as reader:
#             buffer = ""
#             while True:
#                 chunk = reader.read(65536)  # Read in 64KB chunks
#                 if not chunk:
#                     break
#                 buffer += chunk.decode('utf-8')
#                 while '\n' in buffer:
#                     line, buffer = buffer.split('\n', 1)
#                     data.append(json.loads(line.rstrip("\n|\r")))
#             if buffer:  # Handle the last line if there's no newline at the end
#                 data.append(json.loads(buffer.rstrip("\n|\r")))
#     if not quiet:
#         print("Loaded {} records from {}".format(len(data), input_path))
#     return data


# def load_jsonl_zst(input_path, quiet=True):
#     """
#     Generator function to yield lines from a compressed JSON lines file (.jsonl.zst).
#     """
#     dctx = zstd.ZstdDecompressor()
#     with open(input_path, 'rb') as f:
#         with dctx.stream_reader(f) as reader:
#             buffer = b''
#             while True:
#                 chunk = reader.read(8192)
#                 if not chunk:
#                     break
#                 buffer += chunk
#                 while b'\n' in buffer:
#                     line, buffer = buffer.split(b'\n', 1)
#                     yield json.loads(line)
#             if buffer:
#                 yield json.loads(buffer)

def json_iterator(input_dir, text_key="text", EOT_token="<|endoftext|>"):
    all_jsonls = glob(f"{input_dir}/**/*.jsonl", recursive=True)
    for j in all_jsonls:
        data = load_jsonl(j)
        for doc in data:
            yield f"{EOT_token}{doc[text_key]}"


def train_tokenizer(
    input_dir: str, save_path: str, tokenizer_type: str = "BPE", vocab_size: int = 52000):
    """
    Trains a tokenizer on all the json files in `input_dir` and saves it to `save_path`

    :param input_dir: input directory containing jsonl files
    :param save_path: path to save tokenizer to
    :param tokenizer_type: type of tokenizer to train.
    :param vocab_size: int, size of tokenizer's vocab
    :return:
    """

    if tokenizer_type == "BPE":
        model = models.BPE()
    else:
        raise NotImplementedError(f"Tokenizer type {tokenizer_type} not implemented")
    tokenizer = Tokenizer(model)

    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer.normalizer = NFKC()

    # And then train

    # Adding BOS and EOS tokens
    special_tokens = ["<|endoftext|>", "<|padding|>", "<|unknown|>"]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens = special_tokens
    )


    tokenizer.train_from_iterator(json_iterator(input_dir,EOT_token=special_tokens[0]), trainer, length=None)

    # And Save it
    if save_path:
        tokenizer.save(save_path, pretty=True)
        print(f"Tokenizer saved at {save_path}")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="script for training a multilingual "
        "HF tokenizer on CC dumps with upweighting for low resource languages"
    )
    parser.add_argument(
        "--json_input_dir",
        type=str,
        help="Path to folder containing tokenizer training data in jsonl format",
    )
    parser.add_argument(
        "--tokenizer_output_path",
        type=str,
        help="Path to which your trained tokenizer will be saved (should end in .json)",
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        help="type of tokenizer to train, currently only BPE is supported",
        choices=["BPE"],
        default="BPE",
    )
    parser.add_argument(
        "-v",
        "--vocab_size",
        help="vocabulary size of tokenizer, default=32k",
        type=int,
        default=32000,
    )
    args_parsed = parser.parse_args(input_args)

    return args_parsed

def main(args):
    train_tokenizer(
        args.json_input_dir,
        save_path=args.tokenizer_output_path,
        tokenizer_type=args.tokenizer_type,
        vocab_size=args.vocab_size)


if __name__ == "__main__":
    args = parse_args()
    main(args)




# def load_jsonl_zst(input_path, quiet=True):
#     """
#     Generator function to yield lines from a compressed JSON lines file (.jsonl.zst).
#     """
#     dctx = zstd.ZstdDecompressor()
#     with open(input_path, "rb") as f:
#         with dctx.stream_reader(f) as reader:
#             buffer = ""
#             while True:
#                 chunk = reader.read(65536)  # Read in 64KB chunks
#                 if not chunk:
#                     break
#                 buffer += chunk.decode('utf-8')
#                 while '\n' in buffer:
#                     line, buffer = buffer.split('\n', 1)
#                     yield json.loads(line.rstrip("\n|\r"))
#             if buffer:  # Handle the last line if there's no newline at the end
#                 yield json.loads(buffer.rstrip("\n|\r"))

# def json_iterator(input_dir, text_key="text", EOT_token=""):
#     all_jsonls = glob(f"{input_dir}/**/*.jsonl.zst", recursive=True)
#     for j in all_jsonls:
#         for doc in load_jsonl_zst(j):
#             yield f"{EOT_token}{doc[text_key]}"

# def batch_iterator(iterator, batch_size):
#     batch = []
#     for item in iterator:
#         batch.append(item)
#         if len(batch) == batch_size:
#             yield batch
#             batch = []
#     if batch:
#         yield batch

# def train_tokenizer(
#     input_dir: str, save_path: str, tokenizer_type: str = "BPE", vocab_size: int = 52000, batch_size: int = 1000
# ):
#     if tokenizer_type == "BPE":
#         model = models.BPE()
#     else:
#         raise NotImplementedError(f"Tokenizer type {tokenizer_type} not implemented")
#     tokenizer = Tokenizer(model)

#     tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
#     tokenizer.decoder = decoders.ByteLevel()
#     tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
#     tokenizer.normalizer = NFKC()

#     special_tokens = ["<|endoftext|>", "<|padding|>", "<|unknown|>"]
#     trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    
#     iterator = json_iterator(input_dir, EOT_token=special_tokens[0])
#     for batch in batch_iterator(iterator, batch_size):
#         tokenizer.train_from_iterator(batch, trainer, length=None)

#     if save_path:
#         tokenizer.save(save_path, pretty=True)
#         print(f"Tokenizer saved at {save_path}")

# def parse_args(input_args=None):
#     parser = argparse.ArgumentParser(
#         description="script for training a multilingual "
#         "HF tokenizer on CC dumps with upweighting for low resource languages"
#     )
#     parser.add_argument(
#         "--json_input_dir",
#         type=str,
#         help="Path to folder containing tokenizer training data in jsonl format",
#     )
#     parser.add_argument(
#         "--tokenizer_output_path",
#         type=str,
#         help="Path to which your trained tokenizer will be saved (should end in .json)",
#     )
#     parser.add_argument(
#         "--tokenizer_type",
#         type=str,
#         help="type of tokenizer to train, currently only BPE is supported",
#         choices=["BPE"],
#         default="BPE",
#     )
#     parser.add_argument(
#         "-v",
#         "--vocab_size",
#         help="vocabulary size of tokenizer, default=52k",
#         type=int,
#         default=52000,
#     )
#     parser.add_argument(
#         "-b",
#         "--batch_size",
#         help="batch size for processing data, default=1000",
#         type=int,
#         default=1000,
#     )
#     args_parsed = parser.parse_args(input_args)
#     return args_parsed

# def main(args):
#     train_tokenizer(
#         args.json_input_dir,
#         save_path=args.tokenizer_output_path,
#         tokenizer_type=args.tokenizer_type,
#         vocab_size=args.vocab_size,
#         batch_size=args.batch_size,
#     )

# if __name__ == "__main__":
#     args = parse_args()
#     main(args)