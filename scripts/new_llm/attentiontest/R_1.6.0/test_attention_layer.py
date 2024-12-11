import torch
import torch.nn as nn
import time
from typing import Tuple

def scaled_dot_product_attention(q, k, v):
    """Calculate the attention weights and output using scaled dot-product attention without mask."""
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output, attn_weights

class Attention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("Embedding dimension must be divisible by the number of heads.")

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        # Linear projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention (no mask)
        attn_output, attn_weights = scaled_dot_product_attention(q, k, v)

        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(attn_output)

        return output

def measure_throughput(
    model: nn.Module,
    batch_size: int,
    seq_length: int,
    embed_dim: int,
    num_iterations: int = 100,
) -> Tuple[float, float]:

    model.eval()
    
    query = torch.rand(batch_size, seq_length, embed_dim)
    key = torch.rand(batch_size, seq_length, embed_dim)
    value = torch.rand(batch_size, seq_length, embed_dim)
    
    # Warm-up
    for _ in range(5):
        _ = model(query, key, value)

    # Measure
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(query, key, value)
    end_time = time.time()
    
    total_time = end_time - start_time
    total_tokens = batch_size * seq_length * num_iterations
    tokens_per_second = total_tokens / total_time
    avg_latency_ms = (total_time / num_iterations) * 1000
        
    return tokens_per_second, avg_latency_ms

if __name__ == "__main__":
    # Fixed hyperparameters
    embed_dim = 768
    seq_length = 512
    batch_size = 32
    num_iterations = 100
    num_heads_list = [1, 4, 12, 24]

    print("Running on CPU")
    print("Configuration:")
    print(f"- Batch size: {batch_size}")
    print(f"- Sequence length: {seq_length}")
    print(f"- Embedding dimension: {embed_dim}")

    for num_heads in num_heads_list:
        print(f"\nTesting with num_heads={num_heads}")
        model = Attention(embed_dim, num_heads)
        tokens_per_second, avg_latency_ms = measure_throughput(
            model,
            batch_size,
            seq_length,
            embed_dim,
            num_iterations=num_iterations
        )
        print("Results:")
        print(f"- Throughput: {tokens_per_second:.2f} tokens/second")
        print(f"- Average Latency: {avg_latency_ms:.2f} ms/batch")
