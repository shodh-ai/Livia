import os
import requests
import argparse
from concurrent.futures import ThreadPoolExecutor

# Correct base URL for raw files
BASE_URL = "https://huggingface.co/datasets/cerebras/SlimPajama-627B/resolve/main/"

def read_file(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

def get_file_size(url):
    response = requests.head(url)  # Only get headers, not content
    file_size = int(response.headers.get('x-linked-size', -1))  # Get file size from headers
    return file_size if file_size != -1 else -1

def download(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded: {url} to {dest_path}")
        return True
    else:
        print(f"Failed to download: {url} (status code: {response.status_code})")
        return False

def download_file(url, dest_path, retries=3):
    for attempt in range(retries):
        url_file_size = get_file_size(url)
        if url_file_size != -1:
            if os.path.exists(dest_path):
                dest_file_size = int(os.path.getsize(dest_path))
                print(f"URL file size: {url_file_size}\nDest file size: {dest_file_size}", flush=True)
                if dest_file_size == url_file_size:
                    print(f"Already exists: {dest_path}, skipping download.")
                    return
                else:
                    if download(url, dest_path):
                        return
            else:
                if download(url, dest_path):
                    return
        print(f"Retrying ({attempt + 1}/{retries})...")

def main():
    parser = argparse.ArgumentParser(description="Download files from given links.")
    parser.add_argument('--path', type=str, required=True, help='Path to save downloaded files')
    parser.add_argument('--links', type=str, required=True, help='Directory containing link text files')
    parser.add_argument('--workers', type=int, default=10, help='Number of parallel download workers')
    parser.add_argument('--types', type=str, required=True, help='Comma-separated list of file types to process (e.g., "Validation, Test")')
    parser.add_argument('--retries', type=int, default=3, help='Number of retry attempts for failed downloads')
    args = parser.parse_args()

    PATH = args.path
    links_dir = args.links
    workers = args.workers
    types = args.types.split('|')
    retries = args.retries

    # Prepare download tasks
    download_tasks = []

    if 'Train' in types:
        train_paths = read_file(os.path.join(links_dir, 'Train.txt'))
        for path in train_paths:
            p = os.path.join(PATH, path)
            download_tasks.append((BASE_URL + path, p))

    if 'Validation' in types:
        validation_paths = read_file(os.path.join(links_dir, 'Validation.txt'))
        for path in validation_paths:
            p = os.path.join(PATH, path)
            download_tasks.append((BASE_URL + path, p))

    if 'Test' in types:
        test_paths = read_file(os.path.join(links_dir, 'Test.txt'))
        for path in test_paths:
            p = os.path.join(PATH, path)
            download_tasks.append((BASE_URL + path, p))

    # Download files in parallel
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_file, url, dest_path, retries) for url, dest_path in download_tasks]
        for future in futures:
            future.result()

if __name__ == "__main__":
    main()