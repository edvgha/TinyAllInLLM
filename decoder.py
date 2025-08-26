import os
import torch
import argparse
import einops
import torch.nn as nn
import numpy as np
from pathlib import Path

from modules import TransformerLanguageModel, softmax
from tokenizer import Tokenizer

class Decoder:
    def __init__(self, args: argparse.Namespace):
        self.args = args

        self.device = torch.device(self.args.device)

        self.end_of_text_token_id = 0

        self._init_model()
        self._init_tokenizer()


    def _init_model(self):
        self.model = TransformerLanguageModel(vocab_size=self.args.vocab_size, 
                                              context_length=self.args.context_length, 
                                              d_model=self.args.embedding_dim, 
                                              num_layers=self.args.num_layers, 
                                              num_heads=self.args.num_heads, 
                                              d_ff=self.args.d_ff, 
                                              rope_theta=self.args.rope_theta, 
                                              device=self.device,
                                              dtype=torch.float32)
        
        if not isinstance(self.model, nn.Module):
            raise TypeError('modle must be a torch.nn.Module')
        
        model_path = Path(self.args.data_dir) / self.args.model_file
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model file not found at {model_path}')

        model_weights = torch.load(model_path, map_location=self.device)

        if 'model_state_dict' not in model_weights:
            raise KeyError(f"Checkpint is missing expected key: 'model_state_dict'")
    
        self.model.load_state_dict(model_weights['model_state_dict'])

    def _init_tokenizer(self):
        self.tokenizer = Tokenizer.from_files(vocab_filepath=Path(self.args.data_dir) / self.args.vocab_file,
                                              merges_filepath=Path(self.args.data_dir) / self.args.merges_file,
                                              special_tokens=[self.args.special_token])

    @torch.no_grad()
    def decode(self) -> str:
        self.model.eval()

        input_ids = self.tokenizer.encode(self.args.prompt)
        input_ids_tr = torch.tensor(np.array(input_ids), dtype=torch.long, device=self.device)
        input_ids_tr = einops.repeat(input_ids_tr, 's -> 1 s')

        for _ in range(self.args.max_tokens):
            logits = self.model(input_ids_tr)[:, -1, :]

            if self.args.temperature > 0:
                logits = logits / self.args.temperature

            probs = softmax(logits, dim=-1)

            if self.args.top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                sorted_indices_to_remove = cumulative_probs > self.args.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

                probs[indices_to_remove] = 0
                probs = probs / probs.sum()

            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() == self.end_of_text_token_id:
                break

            input_ids_tr = torch.cat([input_ids_tr, next_token], dim=-1)

        return self.tokenizer.decode(input_ids_tr[0].tolist())

