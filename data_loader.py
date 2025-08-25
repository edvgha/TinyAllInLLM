import os
import logging
import typing
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import numpy.typing as npt


def data_loading(data: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(data, np.ndarray) or data.ndim != 1:
        raise ValueError("Input must be a 1D NumPy array")
    if not isinstance(data[0], np.integer):
        if len(data) > 0 and not np.issubdtype(data.dtype, np.integer):
            raise ValueError("Input must contain integer token IDs")
        elif len(data) == 0 and (context_length > 0 or batch_size > 0):
            pass
    
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("'batch_size' must be a positive integer")
    if not isinstance(context_length, int) or context_length <= 0:
        raise ValueError("'context_length' must be a positive integer")
    
    n = len(data)

    if n < context_length + 1:
        raise ValueError(
            f"Input array (length {n}) is too short"
            f"Minimum required length is {context_length + 1}"
        )
    
    num_possible_starts = n - context_length
    start_indices = np.random.randint(0, num_possible_starts, size=batch_size)

    input_sequences = []
    target_sequences = []

    for start_idx in start_indices:
        input_seq = data[start_idx : start_idx + context_length]
        target_seq = data[start_idx + 1 : start_idx + context_length + 1]
        input_sequences.append(input_seq)
        target_sequences.append(target_seq)
    
    inputs_tensor = torch.tensor(np.array(input_sequences), dtype=torch.long, device=device)
    targets_tensor = torch.tensor(np.array(target_sequences), dtype=torch.long, device=device)

    del data

    return inputs_tensor, targets_tensor


def save_checkpoint(logger: logging.Logger, 
                    model: nn.Module,
                    optimizer: optim.Optimizer,
                    iteration: int,
                    out: typing.Union[str, os.PathLike, typing.BinaryIO, typing.IO[bytes]]):
    if not isinstance(model, nn.Module):
        raise TypeError('model must be a torch.nn.Module')
    if not isinstance(optimizer, optim.Optimizer):
        raise TypeError('optimizer must be a torch.optim.Optimizer')
    if not isinstance(iteration, int):
        raise TypeError("iteration must be an integer")
    
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    torch.save(checkpoint, out)

    if isinstance(out, (str, os.PathLike)):
        logger.info(f"Checkpoint saved to '{out}' at iteration {iteration}")
    elif hasattr(out, 'name'):
        try:
            logger.info(f"Checkpoint saved to file stream '{out.name}' at iteration {iteration}")
        except AttributeError:
            logger.info(f"Checkpoint saved to file stream at iteration {iteration}")
    else:
        logger.info(f"Checkpoint saved to file stream at iteration {iteration}")


def load_checkpoint(logger: logging.Logger, 
                    src: typing.Union[str, os.PathLike, typing.BinaryIO, typing.IO[bytes]],
                    model: nn.Module,
                    optimizer: optim.Optimizer) -> int:
    if not isinstance(model, nn.Module):
        raise TypeError('modle must be a torch.nn.Module')
    if not isinstance(optimizer, optim.Optimizer):
        raise TypeError('optimizer most be a torch.optim.Optimizer')
    
    try:
        model_device = next(model.parameters()).device
    except:
        logger.warning("Model has no parameters")
        model_device = torch.device('cpu')

    map_location = model_device

    src_display_name = src if isinstance(src, (str, os.PathLike)) else 'file-like object'
    logger.info(f"Loading checkpint from '{src_display_name}' with map_location='{str(map_location)}'")

    checkpoint = torch.load(src, map_location=map_location)

    expected_keys = ['model_state_dict', 'optimizer_state_dict', 'iteration']
    for key in expected_keys:
        if key not in checkpoint:
            raise KeyError(f"Checkpint is missing expected key: '{key}'")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']

    logger.info(f"Checkpint loaded. Resuming from iteration {iteration}")
    return iteration

