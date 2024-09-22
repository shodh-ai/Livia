import os
import glob
import json
import zstandard as zstd
import argparse

def read_zst_file(file_path):
    dctx = zstd.ZstdDecompressor()
    with open(file_path, 'rb') as f:
        with dctx.stream_reader(f) as reader:
            buffer = b''
            while True:
                chunk = reader.read(8192)
                if not chunk:
                    break
                buffer += chunk
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    yield json.loads(line)
            if buffer:
                yield json.loads(buffer)

def write_jsonl_file(data, output_path):
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def process_chunk(input_dir, output_dir, base_path, chunk_name):
    # Create the output path
    output_chunk_dir = os.path.join(output_dir, base_path)
    os.makedirs(output_chunk_dir, exist_ok=True)
    output_file = os.path.join(output_chunk_dir, f"{base_path}_{chunk_name}.jsonl")

    # Recursively find all .jsonl.zst files in the chunk
    zst_files = glob.glob(os.path.join(input_dir, chunk_name, "*.zst"))
    
    # Read and merge .zst files
    merged_data = []
    for file in zst_files:
        merged_data.extend(read_zst_file(file))

    # Write merged data to a single jsonl file
    write_jsonl_file(merged_data, output_file)
    print(f"Merged {len(zst_files)} files into {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Merge .jsonl.zst files into a single .jsonl file.")
    parser.add_argument('input_dir', type=str, help='Path to the input directory containing .jsonl.zst files')
    parser.add_argument('output_dir', type=str, help='Path to the output directory for the merged .jsonl files')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Traverse the directory structure
    for base_dir, subdirs, _ in os.walk(input_dir):
        for subdir in subdirs:
            if subdir.startswith('chunk'):
                # Determine base path name for output file
                relative_path = os.path.relpath(base_dir, input_dir)
                base_path = relative_path.replace(os.sep, '_')
                
                # Process each chunk directory
                process_chunk(base_dir, output_dir, base_path, subdir)


def see_zst_contents(file_path):
    with open(file_path, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(f)
        unzipped_data = stream_reader.read().decode('utf-8') 
        stream_reader.close()
    return unzipped_data
    
def filter_text_by_meta(data_list, target_set_name):
    filtered_data = []
    for data in data_list:
        data_dict = json.loads(data) 
        if data_dict.get('meta', {}).get('redpajama_set_name') == target_set_name:
            filtered_data.append(data_dict)
    return filtered_data

def save_data_as_jsonl(data, output_file_path):
    
    directory = os.path.dirname(output_file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)
        
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for item in data:
            # print(item)
            # print("##################################################################")
            json_line = json.dumps(item) 
            # print(json_line)
            f.write(json_line + '\n')
            
def process_directory_structure(base_folder,dest_folder,filter_datatype):
    print("inside  process_directory_structure")
    input_base = base_folder
    output_base = dest_folder+f"/{filter_datatype}"
    for zst_file_path in glob.glob(input_base + '/**/*.zst', recursive=True):
        output_folder_path = zst_file_path.replace(input_base, output_base)#.rsplit('/', 1)[0]
        output_folder_path = output_folder_path.replace(".zst", "")
        print(f"Processing {zst_file_path} and saving to {output_folder_path}")

        unzipped_content = see_zst_contents(zst_file_path)
        data_lines = unzipped_content.strip().split('\n')
        filtered_data = filter_text_by_meta(data_lines, filter_datatype)
        
        save_data_as_jsonl(filtered_data, output_folder_path)


if __name__ == "__main__":
    main()
    # base_folder = './DATA'  
    # dest_folder = "dest_folder"
    # filter_datatype = "RedPajamaCommoncrawl"
    # process_directory_structure(base_folder,dest_folder=filter_datatype)
    
# if __name__ == "__main__":
#     input_file_path = "./DATA/test/chunk1/example_holdout_0.jsonl.zst"
#     output_file_path = "./c4chunks/test/chunk1/" 
    
#     unzipped_content = see_zst_contents(input_file_path)
    
    
   
#     data_lines = unzipped_content.strip().split('\n')
    
#     filtered_data = filter_text_by_meta(data_lines, "RedPajamaC4")
#     print(filtered_data)
#     save_data_as_jsonl(filtered_data, output_file_path )
    
    
   





# import os
# import glob
# import json
# import zstandard as zstd
# import argparse
# import asyncio

# async def read_zst_file(file_path, semaphore):
#     async with semaphore:
#         dctx = zstd.ZstdDecompressor()
#         with open(file_path, 'rb') as f:
#             with dctx.stream_reader(f) as reader:
#                 buffer = b''
#                 while True:
#                     chunk = reader.read(8192)
#                     if not chunk:
#                         break
#                     buffer += chunk
#                     while b'\n' in buffer:
#                         line, buffer = buffer.split(b'\n', 1)
#                         yield json.loads(line)
#                 if buffer:
#                     yield json.loads(buffer)

# async def write_jsonl_file(data, output_path, semaphore):
#     async with semaphore:
#         with open(output_path, 'w') as f:
#             for item in data:
#                 f.write(json.dumps(item) + '\n')

# async def process_chunk(input_dir, output_dir, base_path, chunk_name, semaphore):
#     # Create the output path
#     output_chunk_dir = os.path.join(output_dir, base_path)
#     os.makedirs(output_chunk_dir, exist_ok=True)
#     output_file = os.path.join(output_chunk_dir, f"{base_path}_{chunk_name}.jsonl")

#     # Recursively find all .jsonl.zst files in the chunk
#     zst_files = glob.glob(os.path.join(input_dir, chunk_name, "*.zst"))
    
#     # Read and merge .zst files
#     merged_data = []
#     for file in zst_files:
#         async for record in read_zst_file(file, semaphore):
#             merged_data.append(record)

#     # Write merged data to a single jsonl file
#     await write_jsonl_file(merged_data, output_file, semaphore)
#     print(f"Merged {len(zst_files)} files into {output_file}")

# async def main():
#     parser = argparse.ArgumentParser(description="Merge .jsonl.zst files into a single .jsonl file.")
#     parser.add_argument('input_dir', type=str, help='Path to the input directory containing .jsonl.zst files')
#     parser.add_argument('output_dir', type=str, help='Path to the output directory for the merged .jsonl files')
#     parser.add_argument('--semaphore', type=int, default=10, help='Number of concurrent asynchronous tasks (default: 10)')
#     args = parser.parse_args()

#     input_dir = args.input_dir
#     output_dir = args.output_dir
#     semaphore_value = args.semaphore

#     # Ensure output directory exists
#     os.makedirs(output_dir, exist_ok=True)

#     # Set up a semaphore to limit concurrent I/O operations
#     semaphore = asyncio.Semaphore(semaphore_value)

#     # Traverse the directory structure
#     tasks = []
#     for base_dir, subdirs, _ in os.walk(input_dir):
#         for subdir in subdirs:
#             if subdir.startswith('chunk'):
#                 # Determine base path name for output file
#                 relative_path = os.path.relpath(base_dir, input_dir)
#                 base_path = relative_path.replace(os.sep, '_')
                
#                 # Process each chunk directory
#                 tasks.append(process_chunk(base_dir, output_dir, base_path, subdir, semaphore))

#     # Run all tasks concurrently
#     await asyncio.gather(*tasks)

# if __name__ == "__main__":
#     asyncio.run(main())
