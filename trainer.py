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


class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args

        self.current_iter: int = 0
        self.best_val_loss: float = float('inf')

        self._setup_logging()
        self._setup_environment()

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

    def train(self):
        for i in range(100):
            self.logger.info(f'valod: {i}')