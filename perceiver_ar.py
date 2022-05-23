import math
import einops
import torch
import torch.nn as nn
from torch.nn import functional as F

from perceiver import Attention, CrossAttention, SelfAttention, make_cross_attention_mask
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

from typing import List


def make_cross_causal_mask(q_len, kv_len, device):
    """
    Creates a causal mask for the cross attention layer.
    """
    q_index, kv_index = torch.meshgrid(
        torch.arange(q_len, device=device),
        torch.arange(kv_len, device=device),
        indexing='ij')

    causal_mask = q_index < kv_index - kv_len + 1
    causal_mask = causal_mask.unsqueeze(0)
    return causal_mask


def apply_pos_emb(pos_emb, qkv):
    n = qkv[0].shape[-2]
    pos_emb = pos_emb[..., :n, :]
    return tuple(map(lambda t: apply_rotary_emb(pos_emb, t), qkv))


def attend(q, k, v, dropout_prob=0.0, attention_mask=None, rotary_pos_emb=None,
           cache=None, cache_key=None, training=True):
    """Computes multi-head attention using a query, key and value.

    Args:
        q: Query with shape [batch, q_indices, num_heads, head_dim].
        k: Key with shape [batch, kv_indices, num_heads, head_dim].
        v: Value with shape [batch, kv_indices, num_heads, head_dim].
        dropout_prob: dropout probability on the attention weights.
        attention_mask: Array of shape [batch, q_indices, kv_indices] indicating
            which attentions are valid
        rotary_pos_emb: Rotary embedding for positional encoding.
        cache: Cache for the attention.
        cache_key: Key for the cache.

    Returns:
        Output of the attention with shape [batch, q_indices, hiddens]
    """
    batch, q_indices, num_heads, q_head_dim = q.shape
    _, _, _, v_head_dim = v.shape
    hiddens = num_heads * v_head_dim
    offset = cache.get('offset', 0) if cache is not None else 0

    if rotary_pos_emb is not None:
        q, k, v = apply_pos_emb(rotary_pos_emb[..., offset:, :], (q, k, v))

    if offset > 0:
        k_top, v_top = cache[cache_key]
        k = torch.cat([k_top, k], dim=-2)
        v = torch.cat([v_top, v], dim=-2)

    if cache is not None:
        cache[cache_key] = k, v

    attention = torch.einsum('b h t d, b h T d -> b h t T', q, k)

    scale = 1. / math.sqrt(q_head_dim)
    attention = attention * scale

    if attention_mask is not None:
        # Use large_k instead of np.NINF because np.NINF breaks for causal-masked
        # left-padded sampling.
        large_k = 1e4 if attention.dtype == torch.float16 else 1e30
        attention = torch.where(
            attention_mask[:, None, :, :], attention, -large_k)

    normalized = F.softmax(attention, dim=-1)
    if dropout_prob > 0.0:
        normalized = F.dropout(normalized, p=dropout_prob, training=training)

    summed = torch.einsum('b h t T, b h T d -> b h t d', normalized, v)
    summed = summed.view(batch, q_indices, hiddens)

    if attention_mask is not None:
        # If all attended tokens are masked, or for masked tokens
        # some rows of logits gets completely masked, in which case the softmax
        # gives a uniform row and we obtain non-zero outputs where it should be
        # zero. We force zeros.
        wipe_attn = torch.all(
            attention_mask == 0, dim=-1, keepdim=True
        )
        summed = torch.where(wipe_attn, torch.zeros_like(summed), summed)
    return summed


def make_cross_attention_mask(query_mask, kv_mask):
    batch_size, query_len = query_mask.shape
    _, key_len = kv_mask.shape
    mask = torch.outer(query_mask.view(batch_size, -1),
                       kv_mask.view(batch_size, -1))
    assert mask.shape == (batch_size, query_len, key_len)
    return mask


class Attention(nn.Module):
    """Multi-headed {cross, self}-attention."""

    def __init__(self,
                 in_q_channels,
                 in_kv_channels,
                 out_qk_channels,
                 out_v_channels=None,
                 num_heads=8,
                 with_final_bias=True,
                 final_init_scale_multiplier=1.,
                 dropout_prob=0.0,
                 output_channels=None):
        super().__init__()
        self._num_heads = num_heads
        self._with_final_bias = with_final_bias
        self._final_init_scale_multiplier = final_init_scale_multiplier
        self._dropout_prob = dropout_prob
        self._qk_channels = out_qk_channels
        self._v_channels = out_v_channels
        self._output_channels = output_channels

        # Q and K must have the same number of channels.
        # Default to preserving Q's input's shape.
        if self._qk_channels is None:
            self._qk_channels = in_q_channels

        # Q and K must have the same number of channels.
        if self._v_channels is None:
            self._v_channels = self._qk_channels
            # Project the output of QKV attention to a desired number of channels.
            # Default to the same number as the output of the QKV attention operation.
        if self._output_channels is None:
            self._output_channels = self._v_channels

        if self._qk_channels % self._num_heads != 0:
            raise ValueError(f'qk_channels ({self._qk_channels}) must be divisible by'
                             f' num_heads ({self._num_heads}).')
        if self._v_channels % self._num_heads != 0:
            raise ValueError(f'v_channels ({self._v_channels}) must be divisible by'
                             f' num_heads ({self._num_heads}).')

        self.to_q = nn.Linear(in_q_channels, self._qk_channels, bias=True)
        self.to_k = nn.Linear(in_kv_channels, self._qk_channels, bias=True)
        self.to_v = nn.Linear(in_kv_channels, self._v_channels, bias=True)

        self.to_output = nn.Linear(
            self._v_channels, self._output_channels, bias=self._with_final_bias)

    def forward(self, inputs_q, inputs_kv, attention_mask=None,
                rotary_pos_emb=None, cache=None, cache_key=None):
        # Project QKV to a common feature dimension
        qk_channels_per_head = self._qk_channels // self._num_heads
        v_channels_per_head = self._v_channels // self._num_heads

        q = self.to_q(inputs_q)
        k = self.to_k(inputs_kv)
        v = self.to_v(inputs_kv)

        # Reshape channels for multi-head attention
        batch, q_time, _ = q.shape
        _, kv_time, _ = k.shape

        # Modified from original perceiver to support rotary embeddings
        q = einops.rearrange(q, 'b t (h d)', 'b h t d', h=self._num_heads)
        k = einops.rearrange(k, 'b t (h d)', 'b h t d', h=self._num_heads)
        v = einops.rearrange(v, 'b t (h d)', 'b h t d', h=self._num_heads)

        result = attend(q, k, v, self._dropout_prob,
                        attention_mask, rotary_pos_emb, cache, cache_key, self.training)

        return self.to_output(result)


class PerceiverAR(nn.Module):
    """An auto-regressive version of the perceiver encoder."""

    def __init___(
        self,
        # The encoder has a total of
        #   num_self_attends_per_block * num_blocks
        # self-attend layers. We share weights between blocks.
        in_channels,
        max_seq_len,
        num_self_attends_per_block=6,
        num_blocks=8,
        # The latent variable z's dimensionality reflects the
        # the number of most recent tokens at the end of a sequence.
        z_index_dim=512,
        num_z_channels=1024,
        qk_channels=None,
        v_channels=None,
        out_channels=None,
        num_cross_attend_heads=1,
        num_self_attend_heads=8,
        cross_attend_widening_factor=1,
        self_attend_widening_factor=1,
        dropout_prob=0.0,
        z_pos_enc_init_scale=0.02,
        cross_attention_shape_for_attn='kv',
        use_query_residual=True,
            **position_embedding_kwargs):
        super().__init__()

        if num_z_channels % num_self_attend_heads != 0:
            raise ValueError(f'num_z_channels ({num_z_channels}) must be divisible by'
                             f' num_self_attend_heads ({num_self_attend_heads}).')
        if num_z_channels % num_cross_attend_heads != 0:
            raise ValueError(f'num_z_channels ({num_z_channels}) must be divisible by'
                             f' num_cross_attend_heads ({num_cross_attend_heads}).')

        self._input_is_1d = True
        # f TODO: Change z_index_dim to support inference and training
        self._z_index_dim = z_index_dim

        self._num_blocks = num_blocks

        # TODO: extend the input arguments for image / text / audio
        self.rotary_pos_emb = RotaryEmbedding(**position_embedding_kwargs)

        self._input_embs = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=in_channels)

        # Construct the cross attend:
        self.cross_attend = CrossAttention(
            in_q_channels=num_z_channels,
            in_kv_channels=in_channels,
            dropout_prob=dropout_prob,
            num_heads=num_cross_attend_heads,
            widening_factor=cross_attend_widening_factor,
            shape_for_attn=cross_attention_shape_for_attn,
            out_qk_channels=qk_channels or num_z_channels,
            out_v_channels=v_channels or num_z_channels,
            use_query_residual=use_query_residual,
            use_square_relu=True)

        # Construct the block of self-attend layers.
        # We get deeper architectures by applying this block more than once.
        self.self_attends = nn.ModuleList([])
        for _ in range(num_self_attends_per_block):
            self_attend = SelfAttention(
                in_qkv_channels=num_z_channels,
                num_heads=num_self_attend_heads,
                dropout_prob=dropout_prob,
                out_qk_channels=qk_channels or num_z_channels,
                out_v_channels=v_channels or num_z_channels,
                widening_factor=self_attend_widening_factor,
                use_square_relu=True)
            self.self_attends.append(self_attend)

        self._out_channels = out_channels
        if out_channels is not None:
            self.final_layer = nn.Linear(num_z_channels, out_channels)

    def forward(self, inputs, z, *, input_mask=None, rotary_pos_emb=None,
                cache=None, cache_key=None, z_index_dim=1024):
        offset = cache.get('offset', 0) if cache is not None else 0
        attention_mask = None
        if input_mask is not None:
            attention_mask = make_cross_attention_mask(
                query_mask=torch.ones(
                    z.shape[:2], dtype=torch.int32, device=z.device),
                kv_mask=input_mask)

        # Embed the inputs.
        embedded_inputs = self._input_embs(inputs)

        # Get the initial z.
        z = embedded_inputs[:, -z_index_dim:, :]

        # Build the causal mask.
        cross_causal_mask = make_cross_causal_mask(
            q_len=z.shape[1],
            kv_len=embedded_inputs.shape[1], device=z.device)

        if attention_mask is None:
            cross_attention_mask = cross_causal_mask
        else:
            cross_attention_mask = torch.logical_and(
                attention_mask, cross_causal_mask)

        self_causal_mask = torch.ones(
            z.shape[1],
            z.shape[1],
            device=embedded_inputs.device).triu_(z.shapep[1] - z.shapep[1] + 1).bool()

        # Apply the cross-attend.
        z = self.cross_attend(z, inputs,
                              attention_mask=cross_attention_mask, rotary_pos_emb=rotary_pos_emb,
                              cache=cache, cache_key=cache_key)
        for _ in range(self._num_blocks):
            for self_attend in self.self_attends:
                z = self_attend(z, attention_mask=self_causal_mask, rotary_pos_emb=rotary_pos_emb,
                                cache=cache, cache_key=cache_key)

        if self._out_channels is not None:
            z = self.final_layer(z)
        return z
