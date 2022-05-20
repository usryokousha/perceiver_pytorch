# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The code is a direct translation of the original Jax implementation
    into Pytorch code."""

import math
import abc
import torch
import torch.nn as nn
from torch.nn import functional as F

import position_encoding

from typing import List


def attend(q, k, v, dropout_prob=0.0, attention_mask=None, training=True):
    """Computes multi-head attention using a query, key and value.

    Args:
        q: Query with shape [batch, q_indices, num_heads, head_dim].
        k: Key with shape [batch, kv_indices, num_heads, head_dim].
        v: Value with shape [batch, kv_indices, num_heads, head_dim].
        dropout_prob: dropout probability on the attention weights.
        attention_mask: Array of shape [batch, q_indices, kv_indices] indicating
        which attentions are valid
    Returns:
        Output of the attention with shape [batch, q_indices, hiddens]
    """
    batch, q_indices, num_heads, q_head_dim = q.shape
    _, _, _, v_head_dim = v.shape
    hiddens = num_heads * v_head_dim

    attention = torch.einsum('b t h d, b T h d -> b h t T', q, k)

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

    summed = torch.einsum('b h t T, b T h d -> b h t d', normalized, v)
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

    def forward(self, inputs_q, inputs_kv, attention_mask=None):
        # Project QKV to a common feature dimension
        qk_channels_per_head = self._qk_channels // self._num_heads
        v_channels_per_head = self._v_channels // self._num_heads

        q = self.to_q(inputs_q)
        k = self.to_k(inputs_kv)
        v = self.to_v(inputs_kv)

        # Reshape channels for multi-head attention
        batch, q_time, _ = q.shape
        _, kv_time, _ = k.shape

        q = q.view(batch, q_time, self._num_heads, qk_channels_per_head)
        k = k.view(batch, kv_time, self._num_heads, qk_channels_per_head)
        v = v.view(batch, kv_time, self._num_heads, v_channels_per_head)

        result = attend(q, k, v, self._dropout_prob,
                        attention_mask, self.training)

        return self.to_output(result)


class MLP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 widening_factor=4,
                 dropout_prob=0.0,
                 use_square_relu=False) -> None:
        super().__init__()
        self._use_square_relu = use_square_relu
        self.proj1 = nn.Linear(in_channels, out_channels * widening_factor)
        self.proj2 = nn.Linear(out_channels * widening_factor, out_channels)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.proj1(x)
        if self._use_square_relu:
            x = F.relu(x) ** 2
        else:
            x = F.gelu(x)
        x = self.proj2(x)
        return self.dropout(x)


class SelfAttention(nn.Module):
    """A self-attention module, including a dense block."""

    def __init__(self,
                 in_qkv_channels,
                 out_qk_channels,
                 out_v_channels=None,
                 widening_factor=4,
                 dropout_prob=0.0,
                 dropout_attn_prob=0.0,
                 num_heads=8,
                 use_square_relu=False,):
        super().__init__()
        self._use_square_relu = use_square_relu
        self.qkv_norm = nn.LayerNorm(out_qk_channels)
        self.attn_norm = nn.LayerNorm(out_v_channels or out_qk_channels)
        self.attn = Attention(in_qkv_channels, in_qkv_channels, out_qk_channels, out_v_channels,
                              num_heads, dropout_prob=dropout_attn_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.mlp = MLP(out_qk_channels, out_qk_channels,
                       widening_factor, dropout_prob, use_square_relu)

    def forward(self, inputs, attention_mask=None):
        # Project the input to a common feature dimension
        x = inputs
        qkv_inputs = self.qkv_norm(inputs)
        attn = self.attn(qkv_inputs, qkv_inputs, attention_mask)
        x = self.dropout(attn)
        x = x + attn
        x = x + self.mlp(self.attn_norm(attn))
        return x


class CrossAttention(nn.Module):
    def __init__(self,
                 in_q_channels,
                 in_kv_channels,
                 out_qk_channels=None,
                 out_v_channels=None,
                 widening_factor=1,
                 dropout_attn_prob=0.0,
                 dropout_prob=0.0,
                 num_heads=8,
                 shape_for_attn='kv',
                 use_query_residual=True,
                 use_square_relu=False,):
        super().__init__()
        self._use_square_relu = use_square_relu
        self._widening_factor = widening_factor
        self._use_query_residual = use_query_residual
        output_channels = in_q_channels

        if shape_for_attn == 'q':
            qk_channels = in_q_channels
        elif shape_for_attn == 'kv':
            qk_channels = in_kv_channels
        else:
            raise ValueError(f'Unknown value {shape_for_attn} for '
                             'shape_for_attention.')

        v_channels = None
        if out_qk_channels is not None:
            qk_channels = out_qk_channels
        if out_v_channels is not None:
            v_channels = out_v_channels

        self.q_norm = nn.LayerNorm(in_q_channels)
        self.kv_norm = nn.LayerNorm(in_kv_channels)
        self.attn_norm = nn.LayerNorm(out_v_channels or out_qk_channels)

        self.attn = Attention(in_q_channels, in_kv_channels, qk_channels, v_channels,
                                num_heads, dropout_prob=dropout_attn_prob, output_channels=output_channels)

        self.mlp = MLP(v_channels, v_channels, 4, dropout_prob=dropout_prob, use_square_relu=use_square_relu)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs_q, inputs_kv, attention_mask=None):
        attn = self.attn(self.q_norm(inputs_q), self.kv_norm(inputs_kv), attention_mask)
        attn = self.dropout(attn)

        # Optionally include a residual to the query.
        # Consider omitting the residual if the semantics of query and output
        # are different, e.g. if queries are positions and outputs are pixels.
        if self._use_query_residual:
            x = inputs_q + attn
        else:
            x = attn

        x = x + self.mlp(self.attn_norm(attn))

        return x

class PerceiverEncoder(nn.Module):
    """The Perceiver Encoder: a scalable, fully attentional encoder."""

    def __init__(
        self,
        # The encoder has a total of
        #   num_self_attends_per_block * num_blocks
        # self-attend layers. We share weights between blocks.
        in_channels,
        num_self_attends_per_block=6,
        num_blocks=8,
        z_index_dim=512,
        num_z_channels=1024,
        qk_channels=None,
        v_channels=None,
        num_cross_attend_heads=1,
        num_self_attend_heads=8,
        cross_attend_widening_factor=1,
        self_attend_widening_factor=1,
        dropout_prob=0.0,
        z_pos_enc_init_scale=0.02,
        cross_attention_shape_for_attn='kv',
        use_query_residual=True):
        super().__init__()

        # Check that we can use multihead-attention with these shapes.
        if num_z_channels % num_self_attend_heads != 0:
            raise ValueError(f'num_z_channels ({num_z_channels}) must be divisible by'
                        f' num_self_attend_heads ({num_self_attend_heads}).')
        if num_z_channels % num_cross_attend_heads != 0:
            raise ValueError(f'num_z_channels ({num_z_channels}) must be divisible by'
                        f' num_cross_attend_heads ({num_cross_attend_heads}).')

        self._input_is_1d = True

        self._num_blocks = num_blocks

        # Construct the latent array initial state.
        self.z_pos_enc = position_encoding.TrainablePositionEncoding(
            index_dim=z_index_dim,
            num_channels=num_z_channels,
            init_scale=z_pos_enc_init_scale)

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
            use_query_residual=use_query_residual)

        # Construct the block of self-attend layers.
        # We get deeper architectures by applying this block more than once.
        self.self_attends = nn.ModuleList([])
        for _ in range(num_self_attends_per_block):
            self_attend = SelfAttention(
                in_q_channels=num_z_channels,
                in_kv_channels=num_z_channels,
                num_heads=num_self_attend_heads,
                dropout_prob=dropout_prob,
                out_qk_channels=qk_channels or num_z_channels,
                out_v_channels=v_channels or num_z_channels,
                widening_factor=self_attend_widening_factor)
            self.self_attends.append(self_attend)

    def latents(self, inputs):
        # Initialize the latent array for the initial cross-attend.
        return self.z_pos_enc(batch_size=inputs.shape[0])

    def forward(self, inputs, z, *, input_mask=None):
        attention_mask = None
        if input_mask is not None:
            attention_mask = make_cross_attention_mask(
            query_mask=torch.ones(z.shape[:2], dtype=torch.int32, device=z.device),
            kv_mask=input_mask)
        z = self.cross_attend(z, inputs,
                            attention_mask=attention_mask)
        for _ in range(self._num_blocks):
            for self_attend in self.self_attends:
                z = self_attend(z)
        return z

class AbstractPerceiverDecoder(nn.Module, metaclass=abc.ABCMeta):
    """Abstract Perceiver decoder."""

    @abc.abstractmethod
    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None,
                        subsampled_points=None):
        raise NotImplementedError

    @abc.abstractmethod
    def output_shape(self, inputs):
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, query, z, *, is_training, query_mask=None):
        raise NotImplementedError

def unravel_index(flat_index: torch.Tensor, shape: torch.Tensor) -> List[torch.Tensor]:
    # flat_index = operator.index(flat_index)
    result = list()

    if shape == torch.Size([]):
        return 0
    
    for size in shape[::-1]:
        result.append(flat_index % size)
        flat_index = flat_index // size
    
    if len(result) == 1:
        return result[0]
    
    return tuple(result[::-1])

class BasicDecoder(AbstractPerceiverDecoder):
  """Cross-attention-based decoder."""

  def __init__(self,
               output_num_channels,
               position_encoding_type='trainable',
               # Ignored if position_encoding_type == 'none':
               output_index_dims=None,
               subsampled_index_dims=None,
               num_z_channels=1024,
               qk_channels=None,
               v_channels=None,
               use_query_residual=False,
               concat_preprocessed_input=False,
               num_heads=1,
               name='basic_decoder',
               final_project=True,
               **position_encoding_kwargs):
    super().__init__(name=name)
    self._position_encoding_type = position_encoding_type

    # If `none`, the decoder will not construct any position encodings.
    # You should construct your own when quering the decoder.
    self.output_pos_enc = None
    if self._position_encoding_type != 'none':
      self.output_pos_enc = position_encoding.build_position_encoding(
          position_encoding_type,
          index_dims=output_index_dims,
          **position_encoding_kwargs)

    self._output_index_dims = output_index_dims
    if subsampled_index_dims is None:
      subsampled_index_dims = output_index_dims
    self._subsampled_index_dims = subsampled_index_dims
    self._output_num_channels = output_num_channels
    self._final_project = final_project

    self._concat_preprocessed_input = concat_preprocessed_input

    self.decoding_cross_attn = CrossAttention(
        in_q_channels=num_z_channels,
        in_kv_channels=num_z_channels,
        dropout_prob=0.0,
        num_heads=num_heads,
        widening_factor=1.0,
        shape_for_attn='kv',
        qk_channels=qk_channels,
        v_channels=v_channels,
        use_query_residual=use_query_residual)

    self.final_layer = nn.Linear(
        v_channels,
        self._output_num_channels)

  def output_shape(self, inputs):
    return ((inputs[0], self._subsampled_index_dims, self._output_num_channels),
            None)

  def decoder_query(self, inputs, modality_sizes=None,
                    inputs_without_pos=None, subsampled_points=None):
    assert self._position_encoding_type != 'none'  # Queries come from elsewhere
    if subsampled_points is not None:
      # unravel_index returns a tuple (x_idx, y_idx, ...)
      # stack to get the [n, d] tensor of coordinates
      pos = torch.stack(
          unravel_index(subsampled_points, self._output_index_dims),
          dim=1)
      # Map these coordinates to [-1, 1]
      pos = -1 + 2 * pos / self._output_index_dims
      pos = pos.expand(inputs.shape[0], -1, -1)
      pos_emb = self.output_pos_enc(
          batch_size=inputs.shape[0],
          pos=pos)
      pos_emb = pos_emb.view(pos_emb.shape[0], -1, pos_emb.shape[-1])
    else:
      pos_emb = self.output_pos_enc(batch_size=inputs.shape[0])
    if self._concat_preprocessed_input:
      if inputs_without_pos is None:
        raise ValueError('Value is required for inputs_without_pos if'
                         ' concat_preprocessed_input is True')
      pos_emb = torch.concat([inputs_without_pos, pos_emb], dim=-1)

    return pos_emb

  def forward(self, query, z, *,
               query_mask=None):
    # Cross-attention decoding.
    # key, value: B x N x K; query: B x M x K
    # Attention maps -> B x N x M
    # Output -> B x M x K
    # Construct cross attention and linear layer lazily, in case we don't need
    # them.
    attention_mask = None
    if query_mask is not None:
      attention_mask = make_cross_attention_mask(
          query_mask=query_mask,
          kv_mask=torch.ones(z.shape[:2], dtype=torch.int32, device=z.device))
  
    output = self.decoding_cross_attn(query, z,
                                 attention_mask=attention_mask)
    if self._final_project:
      output = self.final_layer(output)
    return output

class Perceiver(nn.Module):
  """The Perceiver: a scalable, fully attentional architecture."""

  def __init__(
      self,
      encoder: PerceiverEncoder,
      decoder: AbstractPerceiverDecoder,
      input_preprocessor=None,
      output_postprocessor=None):
    super().__init__()

    # Feature and task parameters:
    self._input_preprocessor = input_preprocessor
    self._output_postprocessor = output_postprocessor
    self._decoder = decoder
    self._encoder = encoder

  def __call__(self, inputs, *, subsampled_output_points=None,
               pos=None, input_mask=None, query_mask=None):
    if self._input_preprocessor:
      network_input_is_1d = self._encoder._input_is_1d
      inputs, modality_sizes, inputs_without_pos = self._input_preprocessor(
          inputs, pos=pos, 
          network_input_is_1d=network_input_is_1d)
    else:
      modality_sizes = None
      inputs_without_pos = None

    # Get the queries for encoder and decoder cross-attends.
    encoder_query = self._encoder.latents(inputs)
    decoder_query = self._decoder.decoder_query(
        inputs, modality_sizes, inputs_without_pos,
        subsampled_points=subsampled_output_points)

    # Run the network forward:
    z = self._encoder(inputs, encoder_query, input_mask=input_mask)
    _, output_modality_sizes = self._decoder.output_shape(
        inputs)
    output_modality_sizes = output_modality_sizes or modality_sizes

    outputs = self._decoder(
        decoder_query, z, query_mask=query_mask)

    if self._output_postprocessor:
      outputs = self._output_postprocessor(outputs, 
                                           modality_sizes=output_modality_sizes)

    return outputs
