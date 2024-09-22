import os
from datasets import load_dataset

def main():
    # Path to the test.jsonl file
    jsonl_file_path = "test.jsonl"
    
    # Check if the file exists
    if not os.path.exists(jsonl_file_path):
        print(f"File {jsonl_file_path} does not exist.")
        return

    # Load the dataset
    dataset = load_dataset('json', data_files=jsonl_file_path, split='train')

    # Basic analysis
    print(f"Number of records: {len(dataset)}")

    # # Print column names
    # print(f"Column names: {dataset.column_names}")

    # # Print a summary of a text column
    # if 'text' in dataset.column_names:
    #     lengths = [len(record['text']) for record in dataset]
    #     print(f"column 'text' ==> sum(lengths): {sum(lengths)}, len(lengths): {len(lengths)}, avg(lengths): {sum(lengths)/len(lengths):.4f}")
        
    # # Extract the 'meta' column and find unique values
    # if 'meta' in dataset.column_names:
    #     meta_values = [record['meta'] for record in dataset if 'meta' in record]
    #     unique_meta_values = {frozenset(item.items()) for item in meta_values}
    #     print(f"Number of unique 'meta' values: {len(unique_meta_values)}")
    #     print("Unique 'meta' values:")
    #     for value in unique_meta_values:
    #         print(dict(value))
    # else:
    #     print("The 'meta' column is not present in the dataset.")

    # # Print a summary of a all indvidual column
    # if 'text' in dataset.column_names:
    #     for value in unique_meta_values:
    #         value = dict(value)
    #         filtered_dataset = [record for record in dataset if record['meta']['redpajama_set_name'] == value['redpajama_set_name']]
    #         lengths = [len(record['text']) for record in filtered_dataset]
    #         print(f"column {value['redpajama_set_name']} ==> sum(lengths): {sum(lengths)}, len(lengths): {len(lengths)}, avg(lengths): {sum(lengths)/len(lengths):.4f}")

    # print("RedPajamaStackExchange")
    # FD = [record for record in dataset if record['meta']['redpajama_set_name'] == "RedPajamaStackExchange"]
    # print(FD[0]['text'])
    # print("--"*10,"\n")
    # print(FD[1]['text'])
    # print("--"*10,"\n")
    # print(FD[-1]['text'])
    # print("--"*10,"\n")

    
    # print("RedPajamaGithub")
    # FD = [record for record in dataset if record['meta']['redpajama_set_name'] == "RedPajamaGithub"]
    # print(FD[0]['text'])
    # print("--"*10,"\n")
    # print(FD[1]['text'])
    # print("--"*10,"\n")
    # print(FD[-1]['text'])
    # print("--"*10,"\n")

    # print("RedPajamaCommonCrawl")
    # FD = [record for record in dataset if record['meta']['redpajama_set_name'] == "RedPajamaCommonCrawl"]
    # print(FD[0]['text'])
    # print("--"*10,"\n")
    # print(FD[1]['text'])
    # print("--"*10,"\n")
    # print(FD[-1]['text'])
    # print("--"*10,"\n")

    # print("RedPajamaC4")
    # FD = [record for record in dataset if record['meta']['redpajama_set_name'] == "RedPajamaC4"]
    # print(FD[0]['text'])
    # print("--"*10,"\n")
    # print(FD[1]['text'])
    # print("--"*10,"\n")
    # print(FD[-1]['text'])
    # print("--"*10,"\n")

    # print("RedPajamaArXiv")
    # FD = [record for record in dataset if record['meta']['redpajama_set_name'] == "RedPajamaArXiv"]
    # print(FD[0]['text'])
    # print("--"*10,"\n")
    # print(FD[1]['text'])
    # print("--"*10,"\n")
    # print(FD[-1]['text'])
    # print("--"*10,"\n")

    # print("RedPajamaBook")
    # FD = [record for record in dataset if record['meta']['redpajama_set_name'] == "RedPajamaBook"]
    # print(FD[0]['text'])
    # print("--"*10,"\n")
    # print(FD[1]['text'])
    # print("--"*10,"\n")
    # print(FD[-1]['text'])
    # print("--"*10,"\n")


    print("RedPajamaWikipedia")
    FD = [record for record in dataset if record['meta']['redpajama_set_name'] == "RedPajamaWikipedia"]
    print(FD[0]['text'])
    print("--"*10,"\n")
    print(FD[1]['text'])
    print("--"*10,"\n")
    print(FD[-1]['text'])
    print("--"*10,"\n")

if __name__ == "__main__":
    main()
