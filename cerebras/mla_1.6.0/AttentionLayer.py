# Copyright 2022 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import math
from modelzoo.common.pytorch.model_utils.create_initializer import (
    create_initializer,
)


class MultiheadAttention(nn.Module):
    """Multi-head attention layer. Adapted from:
    https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#MultiheadAttention

    Args:
        embed_dim (int): Number of input units in each projection output
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate for key-query weights. Defaults to 0.0.
        batch_first (bool): If True, then the input and output tensors are 
            provided as (batch, seq, feature), otherwise the format will be
            (seq, batch, feature). Default: True (batch, seq, feature).
        add_bias_kv (bool): If specified, adds bias to the key and value sequences at dim=0. Default: False.
        add_zero_attn (bool): If specified, adds a new batch of zeros to the key and value 
            sequences at dim=1. Default: False
        kdim (int):  Number of output units in key projection
        vdim (int):  Number of output units in  projection
        use_projection_bias (bool): Whether to use bias in the key, query, and
            value projections.
        use_ffn_bias (bool): Whether to use bias in the output projection.
        attention_initializer (str): Projection kernel initializer. Defaults to
            ``xavier_uniform``.
        output_layer_initializer (str or initializer): If not None, use this
            initializer for the output transform layer. Defaults to None.
        attention_type (str): The attention variant to execute. Currently
            accepts ``dot_product`` and ``scaled_dot_product``. Defaults to
            ``scaled_dot_product``.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        batch_first=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        use_projection_bias=None,
        use_ffn_bias=False,
        attention_initializer="xavier_uniform",
        output_layer_initializer=None,
        attention_type="scaled_dot_product",
        device=None,
    ):
        _SUPPORTED_ATTENTION_TYPES = ["dot_product", "scaled_dot_product"]
        assert (
            attention_type in _SUPPORTED_ATTENTION_TYPES
        ), f"Attention type {attention_type} is not supported."
        assert (
            embed_dim % num_heads == 0
        ), "embed_dim must be divisible by num_heads."

        assert batch_first, "Currently, only batch_first=True is supported"
        assert not add_bias_kv, "add_bias_kv=True is not supported."
        assert not add_zero_attn, "add_zero_attn=True is not supported."
        assert kdim is None, "kdim should be set to None."
        assert vdim is None, "vdim should be set to None."
        super(MultiheadAttention, self).__init__()

        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.scale_dot_product = attention_type == "scaled_dot_product"

        self.proj_q_dense_layer = nn.Linear(
            self.embed_dim,
            self.embed_dim,
            bias=use_projection_bias,
            device=device,
        )
        self.proj_k_dense_layer = nn.Linear(
            self.embed_dim,
            self.embed_dim,
            bias=use_projection_bias,
            device=device,
        )
        self.proj_v_dense_layer = nn.Linear(
            self.embed_dim,
            self.embed_dim,
            bias=use_projection_bias,
            device=device,
        )

        self.dropout_layer = nn.Dropout(dropout)

        self.proj_output_dense_layer = nn.Linear(
            self.embed_dim, self.embed_dim, bias=use_ffn_bias, device=device,
        )

        # handle initialization
        output_initializer = attention_initializer
        if output_layer_initializer is not None:
            output_initializer = output_layer_initializer

        self.initializer = create_initializer(attention_initializer)
        self.output_initializer = create_initializer(output_initializer)

        self._reset_parameters()

    def _reset_parameters(self):
        # q, k, v projections
        weight_initializer = self.initializer
        weight_initializer(self.proj_q_dense_layer.weight.data)
        weight_initializer(self.proj_k_dense_layer.weight.data)
        weight_initializer(self.proj_v_dense_layer.weight.data)

        # output projections
        weight_initializer = self.output_initializer
        weight_initializer(self.proj_output_dense_layer.weight.data)

    def forward(
        self,
        q,
        k,
        v,
        attn_mask=None,
        key_padding_mask=None,
        need_weights=False,
        average_attn_weights=True,
        past_kv=None,
        cache_present_kv=False,
        position_bias=None,
        rotary_position_embedding_helper=None,
    ):
        """Applies the attention mechanism to queries ``q``, keys ``k`` and values ``v``.

        Args:
            q (Tensor): Queries, shape ``[batch_size, seq_length, embed_dim]``.
            k (Tensor): Keys, shape ``[batch_size, seq_length, embed_dim]``.
            v (Tensor): Values, shape ``[batch_size, seq_length, embed_dim]``.
            attn_mask (Tensor): Attention mask. Can be 2D of shape
                ``[batch_size, seq_length]``, or 3D of shape
                ``[batch, query_length, seq_length]``.
            key_padding_mask (Tensor): If specified, a mask of shape (N, S) indicating 
                which elements within key to ignore for the purpose of attention
                (i.e. treat as “padding”). Defaults to None.
            need_weights (bool): If specified, returns attn_output_weights in addition
                to attn_outputs. Default: False.
            average_attn_weights (bool): If true, indicates that the returned attn_weights
                should be averaged across heads. Otherwise, attn_weights are provided
                separately per head. Note that this flag only has an effect when
                need_weights=True. Default: True (i.e. average weights across heads)
            past_kv (Tensor): Past keys and values. Has shape
                ``[2, batch_size, num_heads, seq_length, embed_dim / num_heads]``.
                The tensors in ``[0,:,:,:,:]`` and ``[1,:,:,:,:]`` contain the
                past keys and values, respectively. Defaults to ``None``.
            cache_present_kv (bool): Specifies if the present keys and values
                must be cached and returned. Needed to speed up the
                computations when the decoder is called within an
                autoregressive loop. Defaults to ``False``.
            training (bool): Training the model if ``True``. Needed to call the
                ``dropout`` (after softmax) in the appropriate mode.
            position_bias (Tensor): Tensor containing position bias to apply in attention.

        Returns:
            If ``cache_present_kv`` is ``False``, no entry for present keys and values
            is provided.
        """

        assert (
            key_padding_mask is None
        ), "Key-padding mask is not implemented yet."
        assert not (
            rotary_position_embedding_helper and position_bias
        ), "Cannot specify both rotary and relative position embeddings, pick one!"

        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = q.shape[:2]
        real_seq_length = seq_length

        # linearly project the query (q), key (k) and value (v) using different
        # learned projections
        q = self.proj_q_dense_layer(q)
        k = self.proj_k_dense_layer(k)
        v = self.proj_v_dense_layer(v)

        # split q, k, v into heads
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        present_kv = None
        if cache_present_kv:
            present_kv = torch.stack([k, v])

        offset_length = 0
        if past_kv is not None:
            offset_length = past_kv[0].shape[-2]
            real_seq_length += offset_length

        if rotary_position_embedding_helper:
            k = rotary_position_embedding_helper.rotate_tensor(
                k, real_seq_length, offset=offset_length
            )
            q = rotary_position_embedding_helper.rotate_tensor(
                q, real_seq_length, offset=offset_length
            )

        if past_kv is not None:
            k_past, v_past = past_kv[0], past_kv[1]
            k = torch.cat([k_past, k], dim=-2)
            v = torch.cat([v_past, v], dim=-2)

        key_length = real_seq_length if present_kv is None else seq_length
        if self.scale_dot_product:
            depth = self.embed_dim // self.num_heads
            q = q * torch.tensor(1 / float(depth) ** 0.5, dtype=torch.float16,)

        # calculate dot product attention
        logits = torch.matmul(q, k.transpose(-1, -2))

        # apply attention mask
        if attn_mask is not None:
            if (
                attn_mask.dtype == torch.float16
                or attn_mask.dtype == torch.float32
            ):
                logits = logits + attn_mask
            else:
                neg_inf = -1e4
                assert len(mask.shape) in [
                    2,
                    3,
                ], "Only 2D/3D masks are supported"

                if len(mask.shape) == 2:
                    if past_kv is not None:
                        past_mask = torch.zeros(
                            (q.shape[0], past_kv.shape[-2]), dtype=mask.dtype
                        )
                        mask = torch.cat([past_mask, mask], axis=-1)

                    batch_size, seq_length = mask.shape[:2]
                    query_length = 1
                else:
                    if past_kv is not None:
                        past_mask = torch.zeros(
                            (q.shape[0], q.shape[-2], past_kv.shape[-2]),
                            dtype=mask.dtype,
                        )
                        mask = torch.cat([past_mask, mask], axis=-1)

                    batch_size, query_length, seq_length = mask.shape[:3]

                # compute the attention_bias based on the mask.
                # shape: (batch_size, 1, 1, seq_length)
                attention_bias = (
                    mask.view(batch_size, 1, query_length, seq_length) * neg_inf
                )
                logits += attention_bias

        # Add relative position bias, if any
        if position_bias is not None:
            logits += position_bias

        weights = nn.functional.softmax(logits.float(), dim=-1).type_as(logits)
        weights = self.dropout_layer(weights)

        # Shape: (batch_size, num_heads, query_length, embed_dim / num_heads)
        attention_output = torch.matmul(weights, v)

        # Recombine heads --> [batch_size, seq_length, embed_dim].
        attention_output = self._combine_heads(attention_output)

        # Run the combined outputs through another linear projection layer.
        attention_output = self.proj_output_dense_layer(attention_output)

        if cache_present_kv:
            return attention_output, present_kv

        if not need_weights:
            return attention_output
        else:
            if average_attn_weights:
                weights = torch.mean(weights, dim=1).squeeze()
            return (
                attention_output,
                weights,
            )

    def _split_heads(self, x):
        """Split x into different heads, and transpose the resulting value. The
        tensor is transposed to insure the inner dimensions hold the correct
        values during the matrix multiplication.

        Args:
            x: A tensor with shape ``[batch_size, seq_length, embed_dim]``.

        Returns:
            A tensor with shape
            ``[batch_size, num_heads, seq_length, embed_dim/num_heads]``.
        """
        batch_size, seq_length = x.shape[:2]
        depth = self.embed_dim // self.num_heads
        return x.view(batch_size, seq_length, self.num_heads, depth).transpose(
            1, 2
        )

    def _combine_heads(self, x):
        """Combine tensor that has been split.

        Args:
            x: A tensor with shape
            ``[batch_size, num_heads, seq_length, embed_dim/num_heads]``.

        Returns:
            A tensor with shape ``[batch_size, seq_length, embed_dim]``.
        """
        batch_size, seq_length = x.shape[0], x.shape[2]
        return x.transpose(1, 2).reshape(batch_size, seq_length, self.embed_dim)




# =========================
# Helper function and MLA implementation
# =========================

def apply_rope_x(x, cos, sin):
    """
    Simple implementation of RoPE that rotates alternate channels.
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class MLA(nn.Module):
    """
    MLA module for self-attention with internal Q and KV projections,
    RoPE application, head splitting, and output projection.
    """
    def __init__(
        self,
        d_model,
        n_heads,
        max_len=1024,
        rope_theta=10000.0,
        attention_initializer="xavier_uniform",
        output_layer_initializer=None
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dh = d_model // n_heads

        # Internal projection dimensions
        self.q_proj_dim = d_model // 2
        self.kv_proj_dim = (2 * d_model) // 3

        # Split each head channels into non-RoPE and RoPE parts
        self.qk_nope_dim = self.dh // 2
        self.qk_rope_dim = self.dh // 2

        # Create initializers
        self.initializer = create_initializer(attention_initializer)
        if output_layer_initializer is None:
            self.output_initializer = self.initializer
        else:
            self.output_initializer = create_initializer(output_layer_initializer)

        # Q projection: down-projection then up-projection
        self.W_dq = nn.Parameter(torch.empty(d_model, self.q_proj_dim))
        self.W_uq = nn.Parameter(torch.empty(self.q_proj_dim, d_model))
        self.q_layernorm = nn.LayerNorm(self.q_proj_dim)

        # KV projection: down-projection (with extra channels for RoPE), then split into KV parts
        self.W_dkv = nn.Parameter(torch.empty(d_model, self.kv_proj_dim + self.qk_rope_dim))
        self.W_ukv = nn.Parameter(torch.empty(self.kv_proj_dim, d_model + (n_heads * self.qk_nope_dim)))
        self.kv_layernorm = nn.LayerNorm(self.kv_proj_dim)

        # Output projection
        self.W_o = nn.Parameter(torch.empty(d_model, d_model))

        # Precompute cosine and sine caches for RoPE
        self.max_seq_len = max_len
        self.rope_theta = rope_theta
        freqs = 1.0 / (rope_theta ** (torch.arange(0, self.dh, 2).float() / self.dh))
        emb = torch.outer(torch.arange(max_len).float(), freqs)
        cos_cached = emb.cos()[None, None, :, :]  # (1, 1, max_len, dh/2)
        sin_cached = emb.sin()[None, None, :, :]
        self.register_buffer("cos_cached", cos_cached)
        self.register_buffer("sin_cached", sin_cached)

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize parameters using the provided initializers
        self.initializer(self.W_dq.data)
        self.initializer(self.W_uq.data)
        self.initializer(self.W_dkv.data)
        self.initializer(self.W_ukv.data)
        self.output_initializer(self.W_o.data)

    def forward(self, x, kv_cache=None, past_length=0):
        """
        x: Tensor of shape (B, S, d_model)
        kv_cache: previous KV cache (or None)
        past_length: length of past tokens for proper RoPE indexing
        """
        B, S, D = x.size()
        print("检查")
        print(compressed_q.dtype)
        # --- Q projection ---
        compressed_q = x @ self.W_dq               # (B, S, q_proj_dim)
        print("检查")
        print(compressed_q.dtype)
        compressed_q = self.q_layernorm(compressed_q)
        Q = compressed_q @ self.W_uq                 # (B, S, d_model)
        # Reshape to (B, n_heads, S, dh)
        Q = Q.view(B, -1, self.n_heads, self.dh).transpose(1, 2)
        # Split each head into non-RoPE and RoPE parts
        Q, Q_for_rope = torch.split(Q, [self.qk_nope_dim, self.qk_rope_dim], dim=-1)

        # Apply RoPE to Q's RoPE part
        cos_q = self.cos_cached[:, :, past_length:past_length+S, :self.qk_rope_dim//2].repeat(1, 1, 1, 1)
        sin_q = self.sin_cached[:, :, past_length:past_length+S, :self.qk_rope_dim//2].repeat(1, 1, 1, 1)
        Q_for_rope = apply_rope_x(Q_for_rope, cos_q, sin_q)


        # --- KV projection ---
        if kv_cache is None:
            compressed_kv = x @ self.W_dkv          # (B, S, kv_proj_dim + qk_rope_dim)
            KV_for_lora, K_for_rope = torch.split(compressed_kv, [self.kv_proj_dim, self.qk_rope_dim], dim=-1)
            KV_for_lora = self.kv_layernorm(KV_for_lora)
        else:
            new_kv = x @ self.W_dkv
            compressed_kv = torch.cat([kv_cache, new_kv], dim=1)
            new_kv, new_K_for_rope = torch.split(new_kv, [self.kv_proj_dim, self.qk_rope_dim], dim=-1)
            old_kv, old_K_for_rope = torch.split(kv_cache, [self.kv_proj_dim, self.qk_rope_dim], dim=-1)
            new_kv = self.kv_layernorm(new_kv)
            old_kv = self.kv_layernorm(old_kv)
            KV_for_lora = torch.cat([old_kv, new_kv], dim=1)
            K_for_rope = torch.cat([old_K_for_rope, new_K_for_rope], dim=1)

        # Map KV projection to (B, S, d_model + n_heads*qk_nope_dim) and reshape to (B, n_heads, S, ...)
        KV = KV_for_lora @ self.W_ukv
        KV = KV.view(B, -1, self.n_heads, self.dh + self.qk_nope_dim).transpose(1, 2)
        # Split out K (non-RoPE) and V (V remains unchanged)
        K, V = torch.split(KV, [self.qk_nope_dim, self.dh], dim=-1)
        S_full = K.size(2)

        # Apply RoPE to K's RoPE part
        K_for_rope = K_for_rope.view(B, -1, 1, self.qk_rope_dim).transpose(1, 2)
        cos_k = self.cos_cached[:, :, :S_full, :self.qk_rope_dim//2].repeat(1, 1, 1, 1)
        sin_k = self.sin_cached[:, :, :S_full, :self.qk_rope_dim//2].repeat(1, 1, 1, 1)
        K_for_rope = apply_rope_x(K_for_rope, cos_k, sin_k)
        K_for_rope = K_for_rope.repeat(1, self.n_heads, 1, 1)

        # Concatenate Q and K parts
        q_heads = torch.cat([Q, Q_for_rope], dim=-1)
        k_heads = torch.cat([K, K_for_rope], dim=-1)
        v_heads = V

        # Create a lower-triangular attention mask (supports past_length)
        mask = torch.ones((S, S_full), device=x.device)
        mask = torch.tril(mask, diagonal=past_length)
        mask = mask[None, None, :, :]
        sq_mask = mask == 1

        # Compute attention using scaled dot-product attention
        attn_output = scaled_dot_product_attention(q_heads, k_heads, v_heads,
            attn_mask=sq_mask)
        # attn_output = torch.nn.functional.scaled_dot_product_attention(
        #     q_heads, k_heads, v_heads,
        #     attn_mask=sq_mask
        # )
        # Reshape back to (B, S, d_model)
        attn_output = attn_output.transpose(1, 2).reshape(B, S, D)
        output = attn_output @ self.W_o.T
        return output, compressed_kv

# =========================
# MultiheadAttention_MLA with the same interface as the original
# =========================

class MultiheadAttention_MLA(nn.Module):
    """
    This module re-implements the original MultiheadAttention with the same interface,
    but internally uses MLA for attention computation. It currently supports only self-attention
    (i.e., q, k, and v must be identical) and ignores parameters such as attn_mask, key_padding_mask,
    position_bias, and rotary_position_embedding_helper.
    """
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        batch_first=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        use_projection_bias=None,
        use_ffn_bias=False,
        attention_initializer="xavier_uniform",
        output_layer_initializer=None,
        attention_type="scaled_dot_product",
        device=None,
    ):
        super(MultiheadAttention_MLA, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.attention_type = attention_type
        # Ignore add_bias_kv, add_zero_attn, kdim, vdim, use_projection_bias, use_ffn_bias
        self.mla = MLA(embed_dim, num_heads, max_len=1024, rope_theta=10000.0,
                       attention_initializer=attention_initializer,
                       output_layer_initializer=output_layer_initializer)
        self.dropout_layer = nn.Dropout(dropout)

        # Reset parameters for compatibility
        self._reset_parameters()

    def _reset_parameters(self):
        # Reset parameters of the internal MLA module
        self.mla._reset_parameters()

    def forward(
        self,
        q,
        k,
        v,
        attn_mask=None,
        key_padding_mask=None,
        need_weights=False,
        average_attn_weights=True,
        past_kv=None,
        cache_present_kv=False,
        position_bias=None,
        rotary_position_embedding_helper=None,
    ):
        # Check that key_padding_mask and rotary_position_embedding_helper are not provided (not supported)
        if key_padding_mask is not None:
            raise NotImplementedError("key_padding_mask is not supported.")
        if rotary_position_embedding_helper is not None:
            raise NotImplementedError("rotary_position_embedding_helper is not supported.")
        # Only support self-attention: q, k, v must be identical
        if not (torch.equal(q, k) and torch.equal(k, v)):
            raise NotImplementedError("MultiheadAttention_MLA only supports self-attention (q=k=v).")
        
        # Call the MLA module for attention computation
        # Here, past_length is set to 0; extend if caching is needed
        output, new_kv = self.mla(q, kv_cache=past_kv, past_length=0)
        output = self.dropout_layer(output)
        
        if cache_present_kv:
            return output, new_kv
        if need_weights:
            # Return None for attention weights as MLA does not output them
            return output, None
        return output


import torch.nn.functional as F


def scaled_dot_product_attention(
    query, key, value,
    attn_mask=None, dropout_p=0.0,
    is_causal=False, scale=None,
    enable_gqa=False
) -> torch.Tensor:
    
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

    if is_causal:
        # temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        # attn_bias = attn_bias.masked_fill(~temp_mask, float('-inf'))
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(0)

        attn_bias = (1.0 - temp_mask.float()) * float('-inf')
        attn_bias = attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias = attn_bias.masked_fill(~attn_mask, float('-inf'))
        else:
            attn_bias = attn_bias + attn_mask


    if enable_gqa:
        query_groups = query.size(-3)
        key_groups = key.size(-3)
        repeat_times = query_groups // key_groups
        key = key.repeat_interleave(repeat_times, dim=-3)
        value = value.repeat_interleave(repeat_times, dim=-3)


    attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
    attn_weight = attn_weight + attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)

    attn_weight = F.dropout(attn_weight, p=dropout_p, training=True)

    output = torch.matmul(attn_weight, value)

    return output
