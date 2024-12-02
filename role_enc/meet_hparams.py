from dataclasses import dataclass
from typing import List  # tjy: from typing import List, Literal
from typing_extensions import Literal

from util.hparams import HyperParams


@dataclass
class MEETHyperParams(HyperParams):
    # Method
    layers: List[int]
    layer_selection: Literal["all", "random"]
    fact_token: Literal[
        "last", "subject_first", "subject_last", "subject_first_after_last"
    ]
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool
    mom2_update_weight: float

    # Module templates
    rewrite_module_tmp: str
    rewrite_module_tmp_1: str
    rewrite_module_tmp_2: str
    rewrite_module_tmp_3: str
    enc_layer_module_tmp: str
    enc_mlp_module_tmp: str
    enc_attn_module_tmp: str
    dec_layer_module_tmp: str
    dec_mlp_module_tmp: str
    dec_cross_attn_module_tmp: str
    dec_self_attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str
    lm_head_module_1: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str
