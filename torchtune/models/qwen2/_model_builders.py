# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List
from functools import partial

from torchtune.models.qwen2._component_builders import qwen2, lora_qwen2

from torchtune.modules import TransformerDecoder
from torchtune.modules.tokenizers import SentencePieceTokenizer
from torchtune.modules.peft import LORA_ATTN_MODULES

"""
Model builders build specific instantiations using component builders. For example
the qwen2_7b model builder uses the qwen2 component builder to create the
qwen2 7B model.
"""


def qwen2_7b() -> TransformerDecoder:
    """
    Builder for creating a Qwen2 model initialized w/ the default 7B parameter values
    from https://arxiv.org/abs/2307.09288

    Returns:
        TransformerDecoder: Instantiation of Qwen2 7B model
    """

    # TODO
