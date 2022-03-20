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
import abc
import torch
import numpy as np
import functools

import torch.nn as nn
from functorch import vmap

def generate_fourier_features(
    pos, num_bands, max_resolution=(224, 224),
    concat_pos=True, sine_only=False):
    """Generate a Fourier frequency position encoding with linear spacing.

    Args:
        pos: The position of n points in d dimensional space.
        A Torch array of shape [n, d].
        num_bands: The number of bands (K) to use.
        max_resolution: The maximum resolution (i.e. the number of pixels per dim).
        A tuple representing resolution for each dimension
        concat_pos: Concatenate the input position encoding to the Fourier features?
        sine_only: Whether to use a single phase (sin) or two (sin/cos) for each
        frequency band.
    Returns:
        embedding: A 1D Torch array of shape [n, n_channels]. If concat_pos is True
        and sine_only is False, output dimensions are ordered as:
            [dim_1, dim_2, ..., dim_d,
            sin(pi*f_1*dim_1), ..., sin(pi*f_K*dim_1), ...,
            sin(pi*f_1*dim_d), ..., sin(pi*f_K*dim_d),
            cos(pi*f_1*dim_1), ..., cos(pi*f_K*dim_1), ...,
            cos(pi*f_1*dim_d), ..., cos(pi*f_K*dim_d)],
        where dim_i is pos[:, i] and f_k is the kth frequency band.
    """
    min_freq = 1.0
    # Nyquist frequency at the target resolution:

    freq_bands = torch.stack([
        torch.linspace(min_freq, res / 2, steps=num_bands)
        for res in max_resolution], dim=0)

    # Get frequency bands for each spatial dimension.
    # Output is size [n, d * num_bands]
    per_pos_features = pos[:, :, None] * freq_bands[None, :, :]
    per_pos_features = per_pos_features.view(-1, np.prod(per_pos_features.shape[1:]))

    if sine_only:
        # Output is size [n, d * num_bands]
        per_pos_features = torch.sin(np.pi * (per_pos_features))
    else:
        # Output is size [n, 2 * d * num_bands]
        per_pos_features = torch.concat(
            [torch.sin(np.pi * per_pos_features),
            torch.cos(np.pi * per_pos_features)], dim=-1)
    
    # Concatenate the raw input positions.
    if concat_pos:
        # Adds d bands to the encoding.
        per_pos_features = torch.concat([pos, per_pos_features], dim=-1)
    return per_pos_features

def build_linear_positions(index_dims, output_range=(-1.0, 1.0)):
  """Generate an array of position indices for an N-D input array.

  Args:
    index_dims: The shape of the index dimensions of the input array.
    output_range: The min and max values taken by each input index dimension.
  Returns:
    A jnp array of shape [index_dims[0], index_dims[1], .., index_dims[-1], N].
  """
  def _linspace(n_xels_per_dim):
    return torch.linspace(
        output_range[0], output_range[1],
        steps=n_xels_per_dim, dtype=torch.float32)

  dim_ranges = [
      _linspace(n_xels_per_dim) for n_xels_per_dim in index_dims]
  array_index_grid = torch.meshgrid(*dim_ranges, indexing='ij')

  return torch.stack(array_index_grid, dim=-1)


class AbstractPositionEncoding(nn.Module, metaclass=abc.ABCMeta):
  """Abstract Perceiver decoder."""

  @abc.abstractmethod
  def forward(self, batch_size, pos):
    raise NotImplementedError

class TrainablePositionEncoding(AbstractPositionEncoding):
    """Trainable position encoding."""
    def __init__(self, index_dim, num_channels=128, init_scale=0.02):
        super(TrainablePositionEncoding, self).__init__()
        self._index_dim = index_dim
        self._num_channels = num_channels
        self._init_scale = init_scale
        self._pos_embs = nn.Parameter(torch.zeros(index_dim, num_channels))
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.trunc_normal_(self._pos_embs.data, std=self._init_scale)

    def forward(self, batch_size: int, pos=None) -> torch.Tensor:
        del pos
        pos_embs = self._pos_embs.unsqueeze(0).expand(batch_size, -1, -1)
        return pos_embs

def _check_or_build_spatial_positions(pos, index_dims, batch_size):
    """Checks or builds spatial position features (x, y, ...).

    Args:
        pos: None, or an array of position features. If None, position features
        are built. Otherwise, their size is checked.
        index_dims: An iterable giving the spatial/index size of the data to be
        featurized.
        batch_size: The batch size of the data to be featurized.
    Returns:
        An array of position features, of shape [batch_size, prod(index_dims)].
    """
    if pos is None:
        pos = build_linear_positions(index_dims)
        pos = pos.unsqueeze(0).expand((batch_size,) + (-1,) * len(pos.shape))
        pos = pos.contiguous().view(batch_size, np.prod(index_dims), -1)

    else:
        # Just a warning label: you probably don't want your spatial features to
        # have a different spatial layout than your pos coordinate system.
        # But feel free to override if you think it'll work!
        assert pos.shape[-1] == len(index_dims)

    return pos

class FourierPositionEncoding(AbstractPositionEncoding):
    """Fourier (Sinusoidal) position encoding."""
    def __init__(self, index_dims, num_bands, concat_pos=True,
                max_resolution=None, sine_only=False):
        super(FourierPositionEncoding, self).__init__()
        self._num_bands = num_bands
        self._concat_pos = concat_pos
        self._sine_only = sine_only
        self._index_dims = index_dims
        # Use the index dims as the maximum resolution if it's not provided.
        self._max_resolution = max_resolution or index_dims

    

    def forward(self, batch_size: int, pos: torch.Tensor=None) -> torch.Tensor:
            pos = _check_or_build_spatial_positions(pos, self._index_dims, batch_size)

            build_ff_fn = functools.partial(generate_fourier_features,
                num_bands=self._num_bands, max_resolution=self._max_resolution,
                concat_pos=self._concat_pos, sine_only=self._sine_only)

            return vmap(build_ff_fn)(pos)

class PositionEncodingProjector(AbstractPositionEncoding):
    """Projects a position encoding to a target size."""

    def __init__(self, output_size: int, base_position_encoding: nn.Module):
        super(PositionEncodingProjector, self).__init__()
        self._output_size = output_size
        self._base_position_encoding = base_position_encoding
        self.proj_pos = nn.LazyLinear(out_features=output_size)

    def forward(self, batch_size: int, pos: torch.Tensor=None) -> torch.Tensor:
        base_pos = self._base_position_encoding(batch_size, pos)
        projected_pos = self.proj_pos(base_pos)
        return projected_pos

def build_position_encoding(
    position_encoding_type,
    index_dims,
    project_pos_dim=-1,
    trainable_position_encoding_kwargs=None,
    fourier_position_encoding_kwargs=None) -> AbstractPositionEncoding:
  """Builds the position encoding."""

  if position_encoding_type == 'trainable':
    assert trainable_position_encoding_kwargs is not None
    output_pos_enc = TrainablePositionEncoding(
        # Construct 1D features:
        index_dim=np.prod(index_dims),
        **trainable_position_encoding_kwargs)
  elif position_encoding_type == 'fourier':
    assert fourier_position_encoding_kwargs is not None
    output_pos_enc = FourierPositionEncoding(
        index_dims=index_dims,
        **fourier_position_encoding_kwargs)
  else:
    raise ValueError(f'Unknown position encoding: {position_encoding_type}.')

  if project_pos_dim > 0:
    # Project the position encoding to a target dimension:
    output_pos_enc = PositionEncodingProjector(
        output_size=project_pos_dim,
        base_position_encoding=output_pos_enc)

  return output_pos_enc



