import os
import time
import typing
import torch
import argparse
import logging
import torch.nn as nn
import torch.optim as optim
import numpy as np
import numpy.typing as npt
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from logging.handlers import RotatingFileHandler
from datetime import datetime

from modules import TransformerLanguageModel
from adamw import AdamW, lr_cosine_schedule, gradient_clipping
from data_loader import data_loading, save_checkpoint, load_checkpoint
from losses import cross_entropy


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(0)


def load_memmap_data(logger: logging.Logger, file_path: str) -> npt.NDArray:
    logger.info(f'Loading memory-mapped data from {file_path}')
    try:
        data = np.load(file_path, mmap_mode='r')
        logger.info(f'Successfully loaded. Shape: {data.shape}, dtype: {data.dtype}')
        return data
    except FileNotFoundError:
        logger.error(f'Data file not found at {file_path}')
        raise
    except Exception as e:
        logger.error(f'Error loading data from {file_path}: {e}')
        raise


class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args

        self.current_iter: int = 0
        self.best_val_loss: float = float('inf')

        self._setup_logging()
        self._setup_environment()
        self._init_tensorboard()
        self._init_model()
        self._init_model_optimizer()
        self._load_data()
        self._load_checkpoint_if_needed()

    def _setup_logging(self):
        Path(self.args.log_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(self.args.log_dir) / f"{self.args.experiment_name}_{timestamp}.log"
        self.logger = logging.getLogger(self.args.experiment_name)
        self.logger.setLevel(logging.INFO)

        handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=1)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def _setup_environment(self):
        set_seed(self.args.seed)
        self.device = torch.device(self.args.device)
        self.logger.info(f'Using device {self.device}')
        Path(self.args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _init_tensorboard(self):
        Path(self.args.tensorboard_log_dir).mkdir(parents=True, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=self.args.tensorboard_log_dir)
        self.logger.info(f'TensorBoard logs will be saved ')

    def _init_model(self):
        self.model = TransformerLanguageModel(vocab_size=self.args.vocab_size, 
                                              context_length=self.args.context_length, 
                                              d_model=self.args.embedding_dim, 
                                              num_layers=self.args.num_layers, 
                                              num_heads=self.args.num_heads, 
                                              d_ff=self.args.d_ff, 
                                              rope_theta=self.args.rope_theta, 
                                              device=self.device)
        self.logger.info(f'Model: TransformerLanguageModel, Params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}')
    
    def _init_model_optimizer(self):
        self.optimizer = AdamW(self.model.parameters(), 
                               lr=self.args.learning_rate, 
                               weight_decay=self.args.weight_decay)

    def _load_data(self):
        train_data_path = Path(self.args.data_dir) / self.args.train_file_name
        val_data_path = Path(self.args.data_dir) / self.args.val_file_name
        self.train_data = load_memmap_data(self.logger, str(train_data_path))
        self.val_data = load_memmap_data(self.logger, str(val_data_path))

    def _load_checkpoint_if_needed(self):
        if not self.args.resume_from_checkpoint:
            return 
                
        ckpt_path_str = self.args.resume_from_checkpoint
        if not ckpt_path_str or not os.path.exists(ckpt_path_str):
            self.logger.warning(f"Checkpoint path '{ckpt_path_str}' not found or not specified. Starting fresh.")
            return 

        self.current_iter = load_checkpoint(self.logger, ckpt_path_str, self.model, self.optimizer)

    @torch.no_grad()
    def _estimate_loss(self) -> dict:
        # TODO
        return {}
    
    def _save_checkpoint(self, is_best: bool = False, iter_override: typing.Optional[int] = None):
        iter_to_save = iter_override if iter_override is not None else self.current_iter

        if is_best:
            out_path_str = str(Path(self.args.checkpoint_dir) / f"best_ckpt_val{self.best_val_loss:.4f}_iter{iter_to_save}.pth")
        else:
            out_path_str = str(Path(self.args.checkpoint_dir) / f"ckpt_iter_{iter_to_save}.pth")

        save_checkpoint(self.logger, self.model, self.optimizer, iter_to_save, out_path_str)
        self.logger.info(f'Checkpoint saved to {out_path_str}|Iter: {iter_to_save}|Val Loss: {self.best_val_loss:.4f}')
    
    def train(self):
        self.model.train()
        t_start_training = time.time()
        initial_start_iter = self.current_iter

        try:
            self.logger.info(f'TRAIN....')
            self.current_iter = 10
            self.best_val_loss = 0.234
            # TODO
        except KeyboardInterrupt:
            self.logger.info('Training interrupted by user.')
        finally:
            total_training_time = time.time() - t_start_training
            self.logger.info(f'Total training time: {total_training_time:.2f} seconds.')

            if self.args.always_save_checkpoint and self.current_iter > initial_start_iter:
                self.logger.info(f'Saving final checkpoint at iteration {self.current_iter}...')
                self._save_checkpoint(is_best=False, iter_override=self.current_iter)

            if self.current_iter > initial_start_iter or self.args.resume_from_checkpoint:
                self.logger.info('Running final evaluation for TensorBoard...')
                final_losses = self._estimate_loss()
                self.tb_writer.add_scalar('Loss/final_val', final_losses.get('val_loss', float('nan')), self.current_iter)
                self.tb_writer.add_scalar('Loss/final_train', final_losses.get('train_loss', float('nan')), self.current_iter)

            if self.tb_writer:
                self.tb_writer.close()
                self.logger.info("TensorBoard writer closed.") 

        self.logger.info("✅ Training complete.")