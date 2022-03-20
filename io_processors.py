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

"""IO pre and post processors for Perceiver.
    The code is a direct translation of the original Jax implementation."""
import einops
import numpy as np
import torch
import position_encoding

from typing import Tuple, Optional, Sequence, Mapping, Union

ModalitySizeT = Mapping[str, int]
PreprocessorOutputT = Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]

class ImagePreProcessor(torch.nn.Module):
    """Image preprocessing for Perceiver Encoder."""
    def __init__(
        self,
        image_size: tuple,
        position_encoding_type: str = 'fourier',
        n_extra_pos_mlp: int = 0,
        n_extra_pos_dim: int = 128,
        num_channels: int = 64,
        concat_or_add_pos: str = 'concat',
        **position_encoding_kwargs):
        super().__init__()

        if concat_or_add_pos not in ['concat', 'add']:
            raise ValueError(
                f'Invalid value {concat_or_add_pos} for concat_or_add_pos.')

        self._concat_or_add_pos = concat_or_add_pos
        self._num_channels = num_channels
        self._n_extra_pos_mlp = n_extra_pos_mlp
        self._index_dim = image_size

        self._positional_encoding = position_encoding.build_position_encoding(
            index_dims=self._index_dim,
            position_encoding_type=position_encoding_type,
            **position_encoding_kwargs)
        
        # position mlp
        if self._n_extra_pos_mlp > 0:
            self._pos_mlp = torch.nn.Sequential()
            for i in range(n_extra_pos_mlp):
                self._pos_mlp.add_module(
                    f'pos_mlp_{i}',
                    torch.nn.Linear(
                        in_features=n_extra_pos_dim,
                        out_features=n_extra_pos_dim))
                if i < n_extra_pos_mlp - 1:
                    self._pos_mlp.add_module(
                        f'pos_relu_{i}',
                        torch.nn.ReLU())

    def _build_network_inputs(self, inputs: torch.Tensor, pos: torch.Tensor,
        network_input_is_1d: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct the final input, including position encoding."""
        batch_size = inputs.shape[0]
        index_dims = inputs.shape[1:-1]

        # Reshape input features to a 1D index dimension if necessary.
        if len(inputs.shape) > 3 and network_input_is_1d:
            inputs = inputs.view(batch_size, np.prod(index_dims), -1)

        # Construct the position encoding.
        pos_enc = self._positional_encoding(batch_size=batch_size, pos=pos)

        if self._n_extra_pos_mlp:
            pos_enc = self._pos_mlp(pos_enc)

        if not network_input_is_1d:
            # Reshape pos to match the input feature shape
            # if the network takes non-1D inputs
            sh = inputs.shape
            pos_enc = torch.reshape(pos_enc, list(sh)[:-1]+[-1])

        if self._concat_or_add_pos == 'concat':
            inputs_with_pos = torch.concat([inputs, pos_enc], dim=-1)
        elif self._concat_or_add_pos == 'add':
            inputs_with_pos = inputs + pos_enc

        return inputs_with_pos, inputs

    def forward(self, inputs: torch.Tensor, pos: Optional[torch.Tensor] = None,
        network_input_is_1d: bool = True) -> PreprocessorOutputT:
        """Preprocess the input image."""
        inputs, inputs_without_pos = self._build_network_inputs(
            inputs=inputs, pos=pos, network_input_is_1d=network_input_is_1d)
        return inputs, None, inputs_without_pos

class Conv2DUpsample(torch.nn.Module):
    """Simple convolutional auto-encoder."""

    def __init__(self,
                in_channels: int,
                out_channels: int,
                n_space_upsamples: int = 4):

        super().__init__()

        prev_channels = in_channels
        self.upsampler = torch.nn.Sequential()
        for i in range(n_space_upsamples):
            channels = out_channels * pow(2, n_space_upsamples - 1 - i)
            self.upsampler.add_module(
                f'upsample_{i}',
                torch.nn.ConvTranspose2d(
                    in_channels=prev_channels,
                    out_channels=channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False))
            self.upsampler.add_module(
                f'relu_{i}',
                torch.nn.ReLU())
            prev_channels = channels



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsampler(x)

class ImagePostProcessor(torch.nn.Module):
    def __init__(
      self,
      spatial_upsample: int = 1,
      query_outputs: int = 128,
      n_outputs: int = 1,
      input_reshape_size: Optional[Sequence[int]] = None,
      channels_first: bool = False):
        super().__init__()
        self._input_reshape_size = input_reshape_size
        self._channels_first = channels_first
        if spatial_upsample != 1:
            def int_log2(x):
                return int(np.round(np.log(x) / np.log(2)))
        self.convnet = Conv2DUpsample(
            query_outputs, n_outputs, int_log2(spatial_upsample))

    def forward(self, inputs: torch.Tensor,pos: Optional[torch.Tensor] = None,
      modality_sizes: Optional[ModalitySizeT] = None) -> torch.Tensor:
        """Postprocess the input image."""
        inputs = inputs.view(
                [inputs.shape[0]] + list(self._input_reshape_size)
                + [inputs.shape[-1]])
        x = inputs.permute(0, 3, 1, 2)
        x = self.convnet(x)

        if self._channels_first:
            x = x.permute(0, 2, 3, 1)
        return x

        
    