import torch
import argparse
import time

import trainer

def setup_args_parser():
    parser = argparse.ArgumentParser(description="Train a TransformerLM model.")

    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="Directory to save checkpoints")
    # Logging
    parser.add_argument('--log_dir', type=str, default='logs', help="Directory to save logs")
    parser.add_argument('--experiment_name', type=str, default='TransformerLM', help='Experiment name')
    # Device arg
    parser.add_argument('--device', type=str, default='auto', help="Device to use ('cpu', 'cuda', 'mps', or 'auto')")
    # TensorBoard args
    parser.add_argument('--tensorboard_log_dir', type=str, default=None, help="Directory for TensorBoard logs")
    # Reproducibility
    parser.add_argument('--seed', type=int, default=1337, help="Random seed for reproducibility")

    args = parser.parse_args()
    if args.device == 'auto':
        if torch.backends.mps.is_available() and torch.backends.mps.is_built(): args.device = 'mps'
        elif torch.cuda.is_available(): args.device = 'cuda'
        else: args.device = 'cpu'
    args.run_name = f"TransformerLM-{time.strftime('%Y%m%d-%H%M%S')}"
    return args


if __name__ == '__main__':
    args = setup_args_parser()
    trainer = trainer.Trainer(args)
    trainer.train()