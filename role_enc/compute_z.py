from typing import Dict, List, Tuple
import numpy as np
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from meet import repr_tools
from util import nethook
from .meet_hparams import MEETHyperParams


def compute_z(
    model: T5ForConditionalGeneration,
    tok: T5Tokenizer,
    request: Dict,
    stat_yes_yes: Dict,
    hparams: MEETHyperParams,
    layer: int,
    context_templates: List[dict],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    target_ids = tok(request["target_new"], return_tensors="pt").to("cuda")["input_ids"][0]
    # [465, 1] 1 is a terminator, consider deleting it or not

    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts = [request["prompt"]]
    kl_prompts = [stat_yes_yes['text']]  # Calculate kl loss using SCI_stat_yes_yes.json
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok([prompt for prompt in all_prompts], return_tensors="pt", padding=True).to("cuda")

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [find_fact_lookup_idx(prompt, request["subject"], tok) for i, prompt in enumerate(all_prompts)]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to decoder {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    delta = torch.zeros((model.config.d_model,), requires_grad=True, device="cuda")
    # delta = torch.ones((model.config.d_model,), requires_grad=True, device="cuda")
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == hparams.enc_layer_module_tmp.format(layer):  # Now it's editing the encoder.
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()
                # target_init = cur_out[0, lookup_idxs[0]].detach().clone()

            # Add intervened delta
            for i, idx in enumerate(lookup_idxs):
                cur_out[0][i, idx, :] += delta
                # cur_out[i, idx, :] += delta

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.dec_layer_module_tmp.format(loss_layer),  # The loss is calculated in the decoder.
                hparams.dec_layer_module_tmp.format(layer),  # Now it's editing the encoder.
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:

            # Using the flan-t5 model, generate responses to questions based on the prompts
            decoder_input_ids = tok([""] * len(all_prompts), return_tensors="pt").input_ids  # Setting the input of the decoder
            decoder_input_ids = model._shift_right(decoder_input_ids)  # Input <pad>, id is 0
            logits = model(**input_tok, decoder_input_ids=decoder_input_ids).logits  # [2, 1, 32128]
            # Compute distribution for KL divergence
            kl_logits = logits[1]  # [1, 32128]
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            kl_test = torch.log_softmax(kl_logits, dim=1)  # 看看结果是否一样
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        rewriting_logits = logits[0]  # [1, 32128]
        log_probs = -torch.log_softmax(rewriting_logits, dim=1)  # Negative logarithmic softmax
        nll_loss = log_probs[0][target_ids[0]]  # Choose the negative log probability of the new target position as the loss

        # Aggregate total losses
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss + weight_decay
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']}] "
            f"{torch.exp(-nll_loss).mean().item()}"
        )
        if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta
    print(
        f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
    )

    return target


def get_module_input_output_at_words(
    model: T5ForConditionalGeneration,
    tok: T5Tokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    context_info = dict(
        context_templates=context_templates,
        words=words,
    )
    subtoken = fact_token_strategy[len("subject_"):]
    l_input, l_output = repr_tools.get_reprs_at_word_tokens(
        track="both", subtoken=subtoken, **context_info, **word_repr_args
    )

    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: T5Tokenizer,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """
    fill_idxs = prompt.index(subject)  # Currently concerned subject
    prefixes = prompt[: fill_idxs]  # Prefix
    prefixes_tok = tok(prefixes)
    lookup_id = len(prefixes_tok['input_ids'])-1  # Minus 1 is to subtract the terminator “1”.
    print(f"Lookup index found: {lookup_id} | Sentence: {prompt} | Token:",
          tok.decode(tok(prompt)["input_ids"][lookup_id]),)
    return lookup_id
