import torch 
import torch.nn as nn
import math
import einops

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype or torch.float32

        self.weight = nn.Parameter(torch.empty((self.out_features, self.in_features), device=self.device, dtype=self.dtype))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        stdv = math.sqrt(2./(self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, std=stdv, a=-3*stdv, b=3*stdv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum('...i,oi->...o', x, self.weight)


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        self.device = device
        self.dtype = dtype or torch.float32

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, dtype=self.dtype, device=self.device))

        torch.nn.init.trunc_normal_(self.weight, mean = 0., std = 1., a = -3., b = 3.)


    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """
        RMSNorm(a_i) = [a_i / RMS(a)] * g_i
        RMS(a) = sqrt((sum{i=1}^{n} a_i^2) / d_model + eps)
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype or torch.float32

        self.weight = nn.Parameter(torch.ones(d_model, device=self.device, dtype=self.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_clone = x.clone()
        in_dtype = x_clone.dtype
        x_fp32 = x_clone.to(torch.float32)

        rms = torch.einsum('...i,...i->...i', x_fp32, x_fp32)
        rms = torch.einsum('...ij->...i', rms).div(self.d_model).add(self.eps).rsqrt()

        x_fp32 = torch.einsum('...ij,...i->...ij', x_fp32, rms)
        x_fp32 = torch.einsum('...i,i->...i', x_fp32, self.weight)
        return x_fp32.to(in_dtype)


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum('...i,...i->...i', x, torch.sigmoid(x))
    

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        d_ff = d_ff or 3 * d_model

        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.einsum('...i,...i->...i', self.silu(self.w1(x)), self.w3(x)))
    

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        self._build_cache(max_seq_len, d_k)

    def _build_cache(self, seq_len: int, dim: int):
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, dim, 2, device=self.device).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, device=inv_freq.device)
        emb = torch.einsum("i,j->ij", t, self.inv_freq)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.LongTensor) -> torch.Tensor:
        # x: (..., seq_len, d_k)
        S, D = x.shape[-2], x.shape[-1]

        if S > self.max_seq_len or D > self.d_k:
            self._build_cache(S, D)

        if token_positions is None:
            token_positions = torch.arange(S, device=x.device)

        cos = self.cos_cached[token_positions, ...]
        sin = self.sin_cached[token_positions, ...]

        # Initial rotation
        R = torch.zeros(S, D, D, device=x.device, dtype=x.dtype)

        # Cos
        cos_expanded = torch.repeat_interleave(cos, repeats=2, dim=-1)
        diag_indices = torch.arange(D, device=x.device)
        R[:, diag_indices, diag_indices] = cos_expanded

        # Sin
        idx_first_half = torch.arange(start=0, end=D, step=2, device=x.device)
        idx_second_half = torch.arange(start=1, end=D, step=2, device=x.device)
        R[:, idx_first_half, idx_second_half] = -sin
        R[:, idx_second_half, idx_first_half] = sin
        
        return torch.einsum('sij,...sj->...si', R, x)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_shifted = x - x_max
    exps = torch.exp(x_shifted)
    sum_exps = torch.sum(exps, dim=dim, keepdim=True)
    return exps / sum_exps


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.BoolTensor | None = None) -> torch.Tensor:
    """
        Q: (batch_size, ..., n, d_k)
        K: (batch_size, ..., m, d_k)
        V: (batch_size, ..., m, d_v)
        mask: (n, m)
        output: (batch_size, ..., n, d_v)
    """
    A = torch.einsum('...nk,...mk->...nm', Q, K)
    A = A * (Q.size(-1) ** -0.5)

    if mask is not None:
        A = A.masked_fill(mask == False, float('-inf'))

    A = softmax(A, dim=-1)
    return torch.einsum('...nm,...mk->...nk', A, V)


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float | None = None, max_seq_len: int | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        self.num_heads = num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.rope = None
        if theta is not None:
            self.rope = RoPE(theta, d_model // num_heads, max_seq_len, device)

    def forward(self, in_features: torch.Tensor, token_positions: torch.LongTensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = in_features.shape

        mask = torch.tril(torch.ones(seq_len, seq_len, device=in_features.device)).bool()
        mask = einops.repeat(mask, 's1 s2 -> b s1 s2', b=batch_size) # [batch_size, seq_len, seq_len]

        Q = einops.rearrange(self.q_proj(in_features),  'b s (h d) -> h b s d', h=self.num_heads)
        K = einops.rearrange(self.k_proj(in_features),  'b s (h d) -> h b s d', h=self.num_heads)
        V = einops.rearrange(self.v_proj(in_features),  'b s (h d) -> h b s d', h=self.num_heads)

        attention_heads = [None] * self.num_heads
        for h in range(self.num_heads):
            Qh, Kh, Vh = Q[h], K[h], V[h]
            if self.rope is not None:
                Qh, Kh = self.rope(Qh, token_positions), self.rope(Kh, token_positions)
            attention_heads[h] = scaled_dot_product_attention(Qh, Kh, Vh, mask) # [batch_size, seq_len, d_model/num_heads]
        attention = torch.cat(attention_heads, dim=-1)
        return self.output_proj(attention) # [batch_size, seq_len, d_model]
    

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float = None, max_seq_len: int = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        """
            in_features: (batch_size, seq_len, dim)
            out: (batch_size, seq_len, dim)
        """
        out = in_features + self.attn(self.ln1(in_features))
        out += self.ffn(self.ln2(out))
        return out
    

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.Sequential(*[TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, theta=rope_theta, max_seq_len=context_length, device=device, dtype=dtype) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        """
            in_indices: (batch_size, seq_len)
            out: (batch_size, seq_len, vocab_len)
        """
        emb = self.token_embeddings(in_indices)
        out = self.layers(emb)
        out = self.ln_final(out)
        out = self.lm_head(out)
        return out