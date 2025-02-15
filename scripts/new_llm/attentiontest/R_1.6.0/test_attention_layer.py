import torch
import time
from modelzoo.common.pytorch.layers.AttentionLayer import MultiheadAttention

def measure_throughput(model: torch.nn.Module, batch_size: int, seq_length: int, embed_dim: int, num_iterations: int = 100):

    model.eval()
    q = torch.randn(batch_size, seq_length, embed_dim)
    k = torch.randn(batch_size, seq_length, embed_dim)
    v = torch.randn(batch_size, seq_length, embed_dim)
    
    with torch.no_grad():
        for _ in range(5):
            _ = model(q, k, v)
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(q, k, v)
    end_time = time.time()

    total_time = end_time - start_time
    total_tokens = batch_size * seq_length * num_iterations
    tokens_per_second = total_tokens / total_time
    avg_latency_ms = (total_time / num_iterations) * 1000
    return tokens_per_second, avg_latency_ms

if __name__ == "__main__":
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
        print(f"\nTesting with num_heads = {num_heads}")
        model = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None,
            use_projection_bias=True,
            use_ffn_bias=False,
            attention_initializer="xavier_uniform",
            output_layer_initializer=None,
            attention_type="scaled_dot_product",
            device=None,
        )
        tokens_per_sec, avg_latency_ms = measure_throughput(model, batch_size, seq_length, embed_dim, num_iterations)
        print("Results:")
        print(f"- Throughput: {tokens_per_sec:.2f} tokens/second")
        print(f"- Average Latency: {avg_latency_ms:.2f} ms/batch")
