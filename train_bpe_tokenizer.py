import os
import time
import argparse
from bpe_tokenizer import train_bpe, save_vocab_json, save_merges_json

def setup_args_parser():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer.")

    parser.add_argument('--data_dir', type=str, default='data', help="Directory containing training data file")
    parser.add_argument('--train_file', type=str, default='TinyStoriesV2-GPT4-valid.txt', help="Name of the training data file")
    parser.add_argument('--vocab_size', type=int, required=True, help="Vocabulary size (max token ID + 1)")
    parser.add_argument('--special_token', type=str, default='<|endoftext|>', help="End of text token")
    parser.add_argument('--output_dir', type=str, default='data', help="Directory to save vocab and merges")

    args = parser.parse_args()
    args.run_name = f"Train-BPE-Tokenizer-{time.strftime('%Y%m%d-%H%M%S')}"
    return args


if __name__ == '__main__':
    start_time = time.time()

    args = setup_args_parser()

    vocab, merges = train_bpe(
        input_path=os.path.join(args.data_dir, args.train_file),
        vocab_size=args.vocab_size,
        special_tokens=[args.special_token]
    )

    save_vocab_json(vocab, os.path.join(args.output_dir, 'vocab.json'))
    save_merges_json(merges, os.path.join(args.output_dir, 'merges.json'))

    elapsed = time.time() - start_time
    print(f'Time: {elapsed:.2f} seconds')
    