import torch
import argparse
import time

import trainer

def setup_args_parser():
    parser = argparse.ArgumentParser(description='Train a TransformerLM model.')

    # Data args
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing train.npy and val.npy')
    parser.add_argument('--train_file_name', type=str, default='train.npy', help='Name of the training data file (tokens)')
    parser.add_argument('--val_file_name', type=str, default='val.npy', help='Name of the validation data file (tokens)')
    # Model args
    parser.add_argument('--vocab_size', type=int, required=True, help='Vocabulary size (max token ID + 1)')
    parser.add_argument('--embedding_dim', type=int, default=32, help='Dimension of token embeddings and model dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of Transformer blocks')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=64, help='Dimension of FF layer')
    parser.add_argument('--rope_theta', type=float, default=1000.0, help='Theat of the ROPE')
    # Optimizer args
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay for AdamW')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping max L2 norm value (0 for no clipping)')
    parser.add_argument('--max_lr', type=float, default=1e-3, help='Maximum learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-4, help='Minimum learning (final) rate')
    parser.add_argument('--warmup_iters', type=int, default=10, help='Number of warm-up iterations')
    parser.add_argument('--cosine_cycle_iters', type=int, default=100, help='Number of cosine annealing iterations')
    # Training args
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--context_length', type=int, default=32, help='Context length for sequences')
    parser.add_argument('--max_iters', type=int, default=120, help='Total number of training iterations')
    parser.add_argument('--log_interval', type=int, default=10, help='Interval for logging training loss')
    parser.add_argument('--eval_interval', type=int, default=40, help='Interval for validation evaluation')
    parser.add_argument('--eval_iters', type=int, default=40, help='Number of batches for loss estimation during eval')
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help="Path to checkpoint to resume from (or 'latest')")
    parser.add_argument('--always_save_checkpoint', action='store_false', help='Always save last checkpoint on exit or max_iters')
    # Logging
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--experiment_name', type=str, default='TransformerLM', help='Experiment name')
    # Device arg
    parser.add_argument('--device', type=str, default='auto', help="Device to use ('cpu', 'cuda', 'mps', or 'auto')")
    # TensorBoard args
    parser.add_argument('--tensorboard_log_dir', type=str, default='runs', help='Directory for TensorBoard logs')
    # Reproducibility
    parser.add_argument('--seed', type=int, default=1337, help='Random seed for reproducibility')

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