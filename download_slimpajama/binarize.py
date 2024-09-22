import os
import yaml
from datasets import concatenate_datasets, load_dataset
from tokenizers import ByteLevelBPETokenizer
import datetime
import argparse
import logging

#@ create a logs folder
os.makedirs("./logs",exist_ok=True)

logging.basicConfig(
    level = logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="./logs/merge_datasets_tokenize_binarize.log",
    filemode="w"
)

class DatasetProcessor:
    def __init__(self, base_dir=""):
        self.base_dir = base_dir

    #@ preprocess map funtion
    def propocess_text(self, example):
        return {"text": "<s> " + example["text"] + "</s>"}
    
    #@ filter fucntion
    def filter_subset(self, example):
        # if example["meta"]['redpajama_set_name'] in ['RedPajamaWikipedia']:
        if example["meta"]['redpajama_set_name'] in ['RedPajamaStackExchange', 'RedPajamaBook','RedPajamaWikipedia', 'RedPajamaArXiv', 'RedPajamaGithub', 'RedPajamaC4', 'RedPajamaCommonCrawl']:
        # if example["meta"]['redpajama_set_name'] in ['RedPajamaStackExchange','RedPajamaWikipedia', 'RedPajamaArXiv', 'RedPajamaC4', 'RedPajamaCommonCrawl']:
            return True
        return False

    def process_dataset(self, dataset):
        remove_columns = ["meta"]
        filtered_dataset = dataset.filter(self.filter_subset)
        logging.info(f"filterting of datset finished")
        return filtered_dataset.map(self.propocess_text,
                                    remove_columns=remove_columns)

    def merge_datasets(self):
        processed_datasets = []

        Dataset = load_dataset("json", data_files = os.path.join(self.base_dir), split='train', cache_dir="/workspace/My_Test/gpt-neox-custom/download_slimpajama/",  num_proc=8)
        logging.info(f"successfuly loaded dataset")

        processed_dataset = self.process_dataset(Dataset)
        logging.info(f"finished maping the dataset ")
        #if processed_dataset:
        #    processed_datasets.append(processed_dataset)
        #    logging.info(f"length of the processed dataset is {len(processed_dataset)}")

        #Dataset.cleanup_cache_files()
        #logging.info(f"cleard cached")
        #merged_dataset = concatenate_datasets(processed_datasets)
        #logging.info(f"finished merging the datasets ")
        return processed_dataset
    

def train_tokenizer(data,st,vocab_size=32000, path="/workspace/My_Test/gpt-neox-custom/download_slimpajama/custom/"):
    logging.info(f"tokenizer training intialized")
    tokenizer = ByteLevelBPETokenizer(add_prefix_space = True, trim_offsets = True)
    startTime = datetime.datetime.now()
    tokenizer.train_from_iterator(data["text"],vocab_size=vocab_size, special_tokens=st)
    tokenizer.save_model(path)
    endTime = datetime.datetime.now()
    logging.info(f"Time taken for training tokenizer, {endTime-startTime}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "10"

    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title="tokenizer")
    group.add_argument(
        "--tokenizer-type",
        type=str,
        default="GPT2BPETokenizer", 
        choices=[
            "HFGPT2Tokenizer",
            "HFTokenizer",
            "GPT2BPETokenizer",
            "CharLevelTokenizer",
            "TiktokenTokenizer",
            "SPMTokenizer",
        ],
        help="What type of tokenizer to use.",
    )


    group = parser.add_argument_group(title="base dir")
    group.add_argument(
        "--base-dir",
        type = str,
        required = True,
        help = "path to the directory containing all the downloaded datsets"
    )

    group.add_argument(
        "-v", "--vocab-file",
        type=str, 
        default=None,
        help="Path to the vocab file"
    )

    group.add_argument(
        "-m", "--merge-file",
        type=str,
        default=None,
        help="Path to the BPE merge file (if necessary).",
    )

    group = parser.add_argument_group(title="workers")
    group.add_argument(
        "--workers",
        type = int,
        required = True,
        help = "no of workers"
    )

    group = parser.add_argument_group(title="output dir")
    group.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to output directory",
    )

    args = parser.parse_args()

    #//@note merge pre-downloaded datasets
    processor = DatasetProcessor(args.base_dir)
    merged_dataset = processor.merge_datasets()
    num_docs = len(merged_dataset)
    logging.info(f"Merged dataset length:,{num_docs}")
    merged_dataset.to_json(os.path.join(args.output_dir,"merged_2.jsonl"))

    #//@note train tokenzier
    if all([args.vocab_file,args.merge_file]):
        vocab_file = args.vocab_file
        merge_file = args.merge_file
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"File not found: {vocab_file}")
        if not os.path.exists(merge_file):
            raise FileNotFoundError(f"File not found: {merge_file}")

    else:
        st = ["<unk>", "<|endoftext|>"]
        train_tokenizer(merged_dataset,st)
        vocab_file = "/workspace/My_Test/gpt-neox-custom/download_slimpajama/custom/vocab.json"
        merge_file = "/workspace/My_Test/gpt-neox-custom/download_slimpajama/custom/merges.txt"

    # #//@note compute binaries
    input_jsonl_path = args.base_dir
    input_jsonl_path = os.path.join(args.output_dir,"merged_2.jsonl")
    tokenizer_type = args.tokenizer_type

    os.system("rm -rf json")
    os.system("rm -rf *.lock")

    #create a custom folder in output_dir
    # os.makedirs(os.path.join(args.output_dir,"custom"),exist_ok=True)
    output_prefix = os.path.join(args.output_dir,"custom")
    workers = args.workers
    dataset_impl = "mmap"

    #//@note craete binaries
    
    cmd = f"python ../tools/datasets/preprocess_data.py \
        --input {input_jsonl_path} --tokenizer-type {tokenizer_type} \
        --vocab-file {vocab_file} --merge-file {merge_file} --append-eod \
        --output-prefix {output_prefix} --dataset-impl {dataset_impl} --workers {workers}" 
        # --output-prefix {output_prefix} --dataset-impl {dataset_impl} --workers {workers} --num-docs {num_docs} " 
    
    startTime = datetime.datetime.now()
    logging.info(f"binarising the merged dataset")
    os.system(cmd)
    endTime = datetime.datetime.now()

    logging.info(f"Time taken for binarization, {endTime-startTime}")

    #//@note count total dataset tokens
    cmd = f"python ../tools/datasets/dataset_token_count.py \
            {os.path.join(args.output_dir,'custom_text_document')}"
    os.system(cmd)

