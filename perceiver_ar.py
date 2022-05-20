from turtle import forward
import torch
from torch.nn import functional as F
import torch.nn as nn

from perceiver import Attention, CrossAttention, SelfAttention, make_cross_attention_mask
from position_encoding import ImagePositionEncoding

def make_causal_mask(q_len, kv_len, device):
    """
    Creates a causal mask for the cross attention layer.
    """
    causal_mask = torch.triu(torch.ones(q_len, kv_len, dtype=torch.bool, 
        device=device), diagonal=1)
    causal_mask = causal_mask.unsqueeze(0)
    return causal_mask

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
        **position_encoding_kwargs):
        super().__init__()

        if num_z_channels % num_self_attend_heads != 0:
            raise ValueError(f'num_z_channels ({num_z_channels}) must be divisible by'
                        f' num_self_attend_heads ({num_self_attend_heads}).')
        if num_z_channels % num_cross_attend_heads != 0:
            raise ValueError(f'num_z_channels ({num_z_channels}) must be divisible by'
                        f' num_cross_attend_heads ({num_cross_attend_heads}).')

        self._input_is_1d = True
        self._z_index_dim = z_index_dim

        self._num_blocks = num_blocks

        self.input_pos_enc = ImagePositionEncoding(
            **position_encoding_kwargs)

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

    def forward(self, inputs, z, *, input_mask=None):
        attention_mask = None
        if input_mask is not None:
            attention_mask = make_cross_attention_mask(
            query_mask=torch.ones(z.shape[:2], dtype=torch.int32, device=z.device),
            kv_mask=input_mask)
        
        # Embed the inputs.
        embedded_inputs = self._input_embs(inputs)
        embedded_inputs += self.input_pos_enc(inputs)

        # Get the initial z.
        z = embedded_inputs[:, -self._z_index_dim:, :]

        # Build the causal mask.
        cross_causal_mask = make_causal_mask(
            q_len=z.shape[1],
            kv_len=embedded_inputs.shape[1], device=z.device)

        if attention_mask is None:
            cross_attention_mask = cross_causal_mask
        else:
            cross_attention_mask = torch.logical_and(attention_mask, cross_causal_mask)

        self_causal_mask = make_causal_mask(
            q_len=z.shape[1],
            kv_len=z.shape[1], device=z.device)

        # Apply the cross-attend.
        z = self.cross_attend(z, inputs,
                            attention_mask=cross_attention_mask)
        for _ in range(self._num_blocks):
            for self_attend in self.self_attends:
                z = self_attend(z, attention_mask=self_causal_mask)

        if self._out_channels is not None:
            z = self.final_layer(z)
        return z