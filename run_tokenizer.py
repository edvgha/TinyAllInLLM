import os
import time
import argparse
import numpy as np
from tokenizer import Tokenizer


def setup_args_parser():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer.")

    parser.add_argument('--data_dir', type=str, default='data', help="Directory containing input file")
    parser.add_argument('--input_file', type=str, default='TinyStoriesV2-GPT4-valid.txt', help="Input file to tokenize")
    parser.add_argument('--merges_file', type=str, default='merges.json', help="JSON file of BPE trained merges")
    parser.add_argument('--vocab_file', type=str, default='vocab.json', help="JSON file of BPE trained vocab")
    parser.add_argument('--special_token', type=str, default='<|endoftext|>', help="End of text token")
    parser.add_argument('--output_file', type=str, default='tokens.npy', help="File to save tokens")

    args = parser.parse_args()
    args.run_name = f"Tokenizer-{time.strftime('%Y%m%d-%H%M%S')}"
    return args


if __name__ == '__main__':
    start_time = time.time()

    args = setup_args_parser()

    try:
        tokenizer = Tokenizer.from_files(
            vocab_filepath=os.path.join(args.data_dir, args.vocab_file),
            merges_filepath=os.path.join(args.data_dir, args.merges_file),
            special_tokens=[args.special_token]
        )

        with open(os.path.join(args.data_dir, args.input_file), 'r', encoding='utf-8') as f:
            text = f.read()
        tokens = tokenizer.encode(text)
        np.save(os.path.join(args.data_dir, args.output_file), np.array(tokens))

    except FileNotFoundError:
        print("❌ Error: The file was not found. Please check the file path.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
            
    encoding_time = time.time()
    print(f'Time to encode: {encoding_time - start_time:.2f} seconds')