from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .compute_z import get_module_input_output_at_words
from .meet_hparams import MEETHyperParams


def compute_ks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: Dict,
    hparams: MEETHyperParams,
    layer: int,
    context_templates: List[dict],
):
    layer_ks = get_module_input_output_at_words(
        model,
        tok,
        layer,
        context_templates=[request["prompt"] for request in requests],
        words=[request["subject"] for request in requests],
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )[0]

    return layer_ks
