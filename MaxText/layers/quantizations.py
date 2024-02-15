#  Copyright 2024 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Quantization library."""

import functools

from aqt.jax.v2 import config as aqt_config
from aqt.jax.v2.flax import aqt_flax
from common_types import Config
from dataclasses import dataclass
import jax.numpy as jnp


# NOTE: This config is not effective, it is kept for API reasons
AQT_INT8_CONFIG = aqt_config.config_v3(
  fwd_bits=8,
  dlhs_bits=8,
  drhs_bits=None,
  rng_type='jax.uniform',
  dlhs_local_aqt=None,
  drhs_local_aqt=None,
  fwd_accumulator_dtype=jnp.int32,
  dlhs_accumulator_dtype=jnp.int32,
  drhs_accumulator_dtype=None,
)

def get_aqt_config(local_aqt_shards):
  print(f"called get_aqt_config with shard count {local_aqt_shards}")
  if local_aqt_shards == 0:
      return aqt_config.config_v3(
        fwd_bits=8,
        dlhs_bits=8,
        drhs_bits=None,
        rng_type='jax.uniform',
        dlhs_local_aqt=None,
        drhs_local_aqt=None,
        fwd_accumulator_dtype=jnp.int32,
        dlhs_accumulator_dtype=jnp.int32,
        drhs_accumulator_dtype=None,
      )
  else:
    return aqt_config.config_v3(
      fwd_bits=8,
      dlhs_bits=8,
      drhs_bits=8,
      rng_type='jax.uniform',
      dlhs_local_aqt=None,
      drhs_local_aqt=aqt_config.LocalAqt(local_aqt_shards),
      fwd_accumulator_dtype=jnp.int32,
      dlhs_accumulator_dtype=jnp.int32,
      drhs_accumulator_dtype=jnp.int32,
    )

@dataclass
class AqtQuantization:
  """ Configures AQT quantization github.com/google/aqt. """
  quant_dg: aqt_config.DotGeneral
  quant_mode: aqt_flax.QuantMode = aqt_flax.QuantMode.TRAIN

  def dot_general_cls(self, local_aqt_shards):
    """ Returns dot_general configured with aqt params. """
    aqt_cfg=get_aqt_config(local_aqt_shards)
    aqt_dg_cls = functools.partial(
      aqt_flax.AqtDotGeneral,
      aqt_cfg,
      rhs_quant_mode=self.quant_mode
      )
    return aqt_dg_cls

  def einsum(self, local_aqt_shards):
    """ Returns einsum configured with aqt params """
    aqt_cfg=get_aqt_config(local_aqt_shards)
    aqt_einsum = functools.partial(aqt_flax.AqtEinsum(
      cfg=aqt_cfg,
      lhs_quant_mode=self.quant_mode
      )
    )
    return aqt_einsum

def _get_quant_config(quant_str: str):
  if not quant_str or quant_str.lower()=="none":
    return None
  if quant_str == "int8":
    return AQT_INT8_CONFIG
  raise ValueError(f'Invalid value configured for quantization {quant_str}.')


def configure_quantization(config: Config, quant_mode_str: str = 'train'):
  """ Configure quantization based on user config and quant mode."""
  quant_cfg = _get_quant_config(config.quantization)
  if quant_cfg:
    if quant_mode_str == 'train':
      return AqtQuantization(quant_cfg, aqt_flax.QuantMode.TRAIN)
    elif quant_mode_str == 'serve':
      return AqtQuantization(quant_cfg, aqt_flax.QuantMode.SERVE)
    elif quant_mode_str == 'convert':
      return AqtQuantization(quant_cfg, aqt_flax.QuantMode.CONVERT)
    else:
      raise ValueError(f'Invalid quantization mode {quant_mode_str}.')
  return None


