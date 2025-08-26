import time
import argparse

import decoder

def setup_args_parser():
    parser = argparse.ArgumentParser(description='Generate text.')

    # Model args
    parser.add_argument('--model_file', type=str, required=True, help="The model")
    parser.add_argument('--vocab_size', type=int, required=True, help='Vocabulary size (max token ID + 1)')
    parser.add_argument('--embedding_dim', type=int, default=32, help='Dimension of token embeddings and model dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of Transformer blocks')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=64, help='Dimension of FF layer')
    parser.add_argument('--rope_theta', type=float, default=1000.0, help='Theat of the ROPE')
    parser.add_argument('--context_length', type=int, default=64, help='Context length for sequences')
    parser.add_argument('--device', type=str, required=True, help="Device on which model trained ('cpu', 'mps', 'cuda')")
    # Generator args
    parser.add_argument('--prompt', type=str, required=True, help='Initial text to start generation from')
    parser.add_argument('--max_tokens', type=int, default=100, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Softmax temperature for scaling logits')
    parser.add_argument('--top_p', type=float, default=0.9, help='Nucleus sampling threshold')
    parser.add_argument('--special_token', type=str, default='<|endoftext|>', help="End of text token")
    parser.add_argument('--merges_file', type=str, default='merges.json', help="JSON file of BPE trained merges")
    parser.add_argument('--vocab_file', type=str, default='vocab.json', help="JSON file of BPE trained vocab")
    parser.add_argument('--data_dir', type=str, default='data', help="Directory containing input file")

    args = parser.parse_args()
    args.run_name = f"GenerateText-{time.strftime('%Y%m%d-%H%M%S')}"
    return args


if __name__ == '__main__':
    args = setup_args_parser()
    generator = decoder.Decoder(args)
    completion = generator.decode()

    print(f"Prompt: '{args.prompt}' \n\n")
    print(f"Completion: '{completion}'")