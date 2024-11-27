import torch
import time
from cerebras.modelzoo.layers.AttentionLayer import MultiheadAttention
from cerebras.pytorch.utils.csdevice import DeviceType, get_device

# Function to measure throughput
def measure_throughput(embed_dim, num_heads, seq_length, batch_size, num_trials=10, device_type="GPU"):
    # Initialize device based on type
    if device_type.upper() == "CSX":
        device = get_device(DeviceType.CSX)
    elif device_type.upper() == "GPU":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # Initialize MultiheadAttention layer
    attention_layer = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, device=device)

    # Generate random input tensors
    q = torch.randn(batch_size, seq_length, embed_dim).to(device)
    k = torch.randn(batch_size, seq_length, embed_dim).to(device)
    v = torch.randn(batch_size, seq_length, embed_dim).to(device)

    # Warm-up runs
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
embed_dim = 512
seq_length = 128
batch_size = 32

# Test for different numbers of heads
num_heads_list = [1, 4, 8, 16]

print("Testing throughput for different numbers of heads on CSX:\n")
for num_heads in num_heads_list:
    avg_time, throughput = measure_throughput(embed_dim, num_heads, seq_length, batch_size, device_type="CSX")
    print(f"Num Heads: {num_heads}, Avg Time: {avg_time:.6f} sec, Throughput: {throughput:.2f} tokens/sec")

print("\nTesting throughput for different numbers of heads on GPU:\n")
for num_heads in num_heads_list:
    avg_time, throughput = measure_throughput(embed_dim, num_heads, seq_length, batch_size, device_type="GPU")
    print(f"Num Heads: {num_heads}, Avg Time: {avg_time:.6f} sec, Throughput: {throughput:.2f} tokens/sec")
