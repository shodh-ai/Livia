import os
import yaml
from datasets import load_from_disk, concatenate_datasets
from tokenizers import ByteLevelBPETokenizer
import datetime

class DatasetProcessor:
    def __init__(self, datasets_file, base_dir=""):
        self.base_dir = base_dir
        self.load_datasets(datasets_file)

    def load_datasets(self, datasets_file):
        with open(datasets_file, "r") as f:
            config = yaml.safe_load(f)
            self.datasets = [tag['name'] for tag in config['ADD_TAGS']]
            self.ADD_TAGS = {tag['name']: {
                'code': eval(tag['code']),
                'remove_columns': tag.get('remove_columns', [])
            } for tag in config['ADD_TAGS']}

    def process_dataset(self, dataset_name, dataset):
        if dataset_name in self.ADD_TAGS:
            remove_columns = self.ADD_TAGS[dataset_name]['remove_columns']
            return dataset.map(self.ADD_TAGS[dataset_name]['code'], remove_columns=remove_columns)
        return None

    def merge_datasets(self):
        processed_datasets = []
        for dataset_name in self.datasets:
            path = os.path.join(self.base_dir,*dataset_name.split(", "))
            Dataset = load_from_disk(os.path.join(self.base_dir, path))
            processed_dataset = self.process_dataset(dataset_name, Dataset)
            if processed_dataset:
                processed_datasets.append(processed_dataset)
                print(dataset_name)
                print(len(processed_dataset))
                print()


                for e in processed_dataset:
                    print(e.keys())
                    break
        merged_dataset = concatenate_datasets(processed_datasets)
        return merged_dataset
    

def train_tokenizer(data,st,vocab_size=50432,path="/workspace/gpt-neox-custom/custom_dataset/"):
    tokenizer = ByteLevelBPETokenizer()
    startTime = datetime.datetime.now()
    tokenizer.train_from_iterator(data["text"],vocab_size=vocab_size, special_tokens=st)
    tokenizer.save_model(path)
    endTime = datetime.datetime.now()
    print("Time taken for training tokenizer",endTime-startTime)




if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "10"

    #//@note merge pre-downloaded datasets
    processor = DatasetProcessor("./datasets.yaml","/data/data")
    merged_dataset = processor.merge_datasets()
    num_docs = len(merged_dataset)
    print("Merged dataset length:", num_docs)
    merged_dataset.to_json(f"../../Create_Dataset/merged.jsonl")

    #//@note train tokenzier
    st = ['<|endoftext|>', '<bos>', '<eos>', '<question>', '<answer>', '</question>', '</answer>', '<textbook>', '</textbook>', '<prompt>', '</prompt>', '<topic>', '</topic>', "<subfield>", "</subfield>", "<field>", "</field>", "<concept>", "</concept>"]
    train_tokenizer(merged_dataset,st)

    #//@note compute binaries
    input_jsonl_path = "../../Create_Dataset/merged.jsonl"
    tokenizer_type = "GPT2BPETokenizer"
    vocab_file = "../../Create_Dataset/vocab.json"
    merge_file = "../../Create_Dataset/merges.txt"
    output_prefix = "../../Create_Dataset/custom/custom"
    workers = 100
    dataset_impl = "mmap"

    cmd = f"sudo python ../tools/datasets/preprocess_data.py \
        --input {input_jsonl_path} --num-docs {num_docs} --tokenizer-type {tokenizer_type} \
        --vocab-file {vocab_file} --merge-file {merge_file} --append-eod \
        --output-prefix {output_prefix} --dataset-impl {dataset_impl} --workers {workers}" 
    
    startTime = datetime.datetime.now()
    os.system(cmd)
    endTime = datetime.datetime.now()

    print("Time taken for binarization",endTime-startTime)


    #//@note count total dataset tokens
    cmd = f"sudo python ../tools/datasets/dataset_token_count.py \
            ../../Create_Dataset/custom/custom_text_document"
    os.system(cmd)

