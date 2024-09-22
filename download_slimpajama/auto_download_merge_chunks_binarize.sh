#!/bin/bash

SEMAPHORE=100
TWORKERS=40
PWORKERS=10
TYPES='Validation|Test'

# Create a log file
LOG_FILE="slimpajama_processing.log"

# Function to log and time a command
log_and_time() {
    local cmd="$1"
    echo "Running: $cmd"
    echo "Running: $cmd" >> "$LOG_FILE"
    local start_time=$(date +%s)
    { $cmd; } 2>&1 | tee -a "$LOG_FILE"
    local end_time=$(date +%s)
    local execution_time=$((end_time - start_time))
    echo "Execution Time: $execution_time seconds" >> "$LOG_FILE"
}

# log_and_time "python multithread_downloader.py --path DATA/ --links Slimpajama_zst_links --workers $TWORKERS --types $TYPES --retries 5"
log_and_time "python merge_zst_to_jsonl.py DATA/ chunks/ "
cd ../tools/datasets
# echo $(pwd) >> "$LOG_FILE"
log_and_time "python preprocess_data.py --input_dir ../../download_slimpajama/chunks/ --output-prefix ../../download_slimpajama/custom/data --workers $PWORKERS --append-eod --tokenizer HFTokenizer --vocab-file ../../download_slimpajama/slimpajama_val_test_trained_bpe_tok.json"
# log_and_time "python preprocess_data.py --input_dir ../../download_slimpajama/chunks/validation/ --output-prefix ../../download_slimpajama/custom/validation --workers $PWORKERS --append-eod --tokenizer HFTokenizer --vocab-file ../../download_slimpajama/tokenizer_loubnabnl_slimpajama.json"
# log_and_time "python preprocess_data.py --input_dir ../../download_slimpajama/chunks/train/ --output-prefix ../../download_slimpajama/custom/train --workers $PWORKERS --append-eod --tokenizer HFTokenizer --vocab-file ../../download_slimpajama/tokenizer_loubnabnl_slimpajama.json"
