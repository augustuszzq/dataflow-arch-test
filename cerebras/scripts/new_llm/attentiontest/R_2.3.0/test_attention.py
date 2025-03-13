import torch
import time
from cerebras.modelzoo.layers.AttentionLayer import MultiheadAttention

# Function to measure throughput
def measure_throughput(embed_dim, num_heads, seq_length, batch_size, num_trials=10):
    # Initialize MultiheadAttention layer (default on CPU)
    attention_layer = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    # Generate random input tensors on CPU
    q = torch.randn(batch_size, seq_length, embed_dim)
    k = torch.randn(batch_size, seq_length, embed_dim)
    v = torch.randn(batch_size, seq_length, embed_dim)

    # Warm-up runs (not included in timing)
    for _ in range(5):
        _ = attention_layer(q, k, v)

    # Measure throughput
    start_time = time.time()
    for _ in range(num_trials):
        _ = attention_layer(q, k, v)
    end_time = time.time()

    avg_time_per_run = (end_time - start_time) / num_trials
    throughput = batch_size * seq_length / avg_time_per_run  # Tokens processed per second

    return avg_time_per_run, throughput

# Test configurations
embed_dim = 768
seq_length = 512
batch_size = 32

# Test for different numbers of heads
num_heads_list = [1, 4, 12, 24]

print("Testing throughput for different numbers of heads on CPU:\n")
for num_heads in num_heads_list:
    avg_time, throughput = measure_throughput(embed_dim, num_heads, seq_length, batch_size)
    print(f"Num Heads: {num_heads}, Avg Time: {avg_time:.6f} sec/run, Throughput: {throughput:.2f} tokens/sec")
