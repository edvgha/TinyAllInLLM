import torch

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
        logits: (batch_size, seq_len, vocab_len)
        targets: (batch_size, seq_len)
    """
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    log_sum_exp = torch.logsumexp(logits - max_logits, dim=-1, keepdim=True)
    log_probs = logits - max_logits - log_sum_exp
    selected_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    nll_loss_per_sample = -selected_log_probs
    loss = nll_loss_per_sample.mean()
    return  loss


def perplexity(logits: torch.Tensor, targets: torch.LongTensor) -> torch.Tensor:
    return torch.exp(cross_entropy(logits, targets))