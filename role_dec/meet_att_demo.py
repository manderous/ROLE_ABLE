from typing import Dict, List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from role_enc import MEETHyperParams, apply_meet_to_model
from role_dec import MEET_ATT_HyperParams, apply_meet_att_to_model
from util import nethook
from util.generate import generate_T5
from util.globals import *
import time


def meet_att_demo_model_editing(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request_yes_no: List[Dict],
    stat_yes_yes: List[Dict],
    generation_prompts: List[str],
    generation_targets: List[str],
    alg_name: str = "ROLE_Dec",
) -> Tuple[AutoModelForCausalLM, Dict[str, torch.Tensor]]:
    """
    Applies the selected model editing algorithm. Generates text both before and after
    for comparison of model behavior. Returns the updated model and the original values of
    weights that were changed.
    """

    if alg_name == "ROLE_Enc" or alg_name == "ROLE_Dec":
        model.config._name_or_path = "flan-t5-large"
    else:
        model.config._name_or_path = "gpt2-xl"  # original

    nethook.set_requires_grad(True, model)

    RewritingParamsClass, apply_method, hparams_prefix, hparams_suffix = load_alg(
        alg_name
    )
    neg_params_name = (HPARAMS_DIR/hparams_prefix/f"{model.config._name_or_path}_neg.json")  # Negative sample parameters

    # Retrieving ROLE_Dec hyperparameters
    print("Loading from", neg_params_name)
    neg_hparams = RewritingParamsClass.from_json(neg_params_name)  # Load negative sample parameters
    print(neg_hparams)

    # Generating pre-update text
    pre_update_text = generate_T5(model, tok, generation_prompts, generation_targets, request_yes_no,
                                  max_out_len=6)  # flan-t5-large
    for text in pre_update_text:  # Insert line breaks
        print(text, end="\n")

    # start = time.time()
    # Applying ROLE_Dec to model for negative samples from YES to NO
    model_new, orig_weights = apply_method(
        model,
        tok,
        request_yes_no,
        stat_yes_yes,
        neg_hparams,  # pos_hparams, neg_hparams
        return_orig_weights=True,
    )
    # end = time.time()
    # print(end - start)  # Calculating editing time

    # Generating post-update text
    pre_update_text = generate_T5(model_new, tok, generation_prompts, generation_targets, request_yes_no,
                                  max_out_len=6)  # flan-t5-large
    for text in pre_update_text:  # Insert line breaks
        print(text, end="\n")

    return model_new, orig_weights


def load_alg(alg_name):
    """
    Loads dependencies for the desired algorithm.
    Implementation is slightly awkward to prevent unnecessary imports on Colab.

    The return value is a tuple of the following:
    1. Class for storing hyperparameters
    2. Method for applying rewrites
    3. Location of parameters
    4. Predefined suffix for the param file
    """
    assert alg_name in [
        "FT",
        "FT-L",
        "FT-AttnEdit",
        "MEND",
        "MEND-CF",
        "MEND-zsRE",
        "ROME",
        "MEMIT",
        "MEET",
        "MEET_ATT"
    ]

    if alg_name == "ROLE_Enc":
        return MEETHyperParams, apply_meet_to_model, "ROLE_Enc", ""
    elif alg_name == "ROLE_Dec":
        return MEET_ATT_HyperParams, apply_meet_att_to_model, "ROLE_Dec", ""
