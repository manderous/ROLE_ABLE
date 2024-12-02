import unicodedata
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

from util.logit_lens import LogitLens
import torch.utils.data as data
from tqdm import tqdm
import random
from util import nethook


def generate_interactive(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    top_k: int = 5,
    max_out_len: int = 200,
    compare_against: Optional[AutoModelForCausalLM] = None,
    use_logit_lens: bool = False,
    layer_module_tmp: str = "transformer.h.{}",
    ln_f_module: str = "transformer.ln_f",
    lm_head_module: str = "lm_head",
):
    """
    Puts generation in a loop. Allows users to repeatedly provide inputs
    with which text is generated.
    """

    if use_logit_lens:
        llens_gen = LogitLens(
            model,
            tok,
            layer_module_tmp,
            ln_f_module,
            lm_head_module,
            disabled=not use_logit_lens,
        )
        if compare_against:
            llens_vanilla = LogitLens(
                compare_against,
                tok,
                layer_module_tmp,
                ln_f_module,
                lm_head_module,
                disabled=not use_logit_lens,
            )

    while True:
        prompt = input("Enter a prompt: ").strip(" \r\t\n")

        print(
            f"Argument Model: "
            f"{generate_fast(model, tok, [prompt], n_gen_per_prompt=1, top_k=top_k, max_out_len=max_out_len)}"
        )
        if compare_against:
            print(
                f"Baseline Model: "
                f"{generate_fast(compare_against, tok, [prompt], n_gen_per_prompt=1, top_k=top_k, max_out_len=max_out_len)}"
            )

        if use_logit_lens:
            inp_prompt = tok([prompt], padding=True, return_tensors="pt").to(
                next(model.parameters()).device
            )

            with llens_gen:
                model(**inp_prompt)
            print("\n--- Argument Model Logit Lens ---")
            llens_gen.pprint()

            if compare_against:
                with llens_vanilla:
                    compare_against(**inp_prompt)
                print("--- Baseline Model Logit Lens ---")
                llens_vanilla.pprint()

        print()


def generate_T5(
    model: T5ForConditionalGeneration,
    tok: T5Tokenizer,
    prompts: List[str],
    y_true: List[str],
    request_yes_no: List[str],
    n_gen_per_prompt: int = 1,
    top_k: int = 5,
    max_out_len: int = 200,
):  # flan-t5

    inp = tok(prompts, padding=True, return_tensors="pt").to("cuda")  # right padding for T5
    torch_dataset = data.TensorDataset(inp['input_ids'], inp['attention_mask'])
    loader = data.DataLoader(
        dataset=torch_dataset,
        batch_size=8,
        shuffle=None,
        num_workers=0
    )

    out_preds = []
    for input_ids, attention_mask in tqdm(loader):
        decoder_input_ids = tok([""] * input_ids.shape[0], return_tensors="pt").input_ids
        decoder_input_ids = model._shift_right(decoder_input_ids)
        logits = model(input_ids, attention_mask, decoder_input_ids=decoder_input_ids).logits  # [batch_size, 1, 32128]
        logits = logits[:, 0, :]  # [batch_size, 32128]
        out_prob = torch.softmax(logits, dim=1)
        p, preds = torch.max(out_prob, dim=1)
        out_preds.append(preds)
    out_preds = torch.cat(out_preds)

    y_pred = []
    for out_i in out_preds:
        prediction = tok.decode(out_i)
        if prediction == "Yes" or prediction == "yes":
            pred_label = 1
        elif prediction == "No" or prediction == "no":
            pred_label = 0
        y_pred.append(pred_label)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="binary")
    recall = recall_score(y_true, y_pred, average="binary")
    f1 = f1_score(y_true, y_pred, average="binary")

    print("Acc: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(accuracy, precision, recall, f1))

    y_true_pos_idx = [idx for idx, y_each in enumerate(y_true) if y_each == 1]
    y_true_neg_idx = [idx for idx, y_each in enumerate(y_true) if y_each == 0]
    random.shuffle(y_true_pos_idx)
    random.shuffle(y_true_neg_idx)
    pos_max_len = len(y_true_pos_idx) if len(y_true_pos_idx) < 10 else 10
    neg_max_len = len(y_true_neg_idx) if len(y_true_neg_idx) < 10 else 10
    shuffle_order = y_true_pos_idx[:pos_max_len] + y_true_neg_idx[:neg_max_len]

    txt = []
    for idx in shuffle_order[:20]:
        prompt = prompts[idx]
        real_target = "Yes" if y_true[idx] == 1 else "No"
        predict_target = tok.decode(out_preds[idx])
        txt.append("{}: real_answer:{}, predict_answer:{}".format(prompt, real_target, predict_target))
    return txt


def generate_T5_multi(
    model: T5ForConditionalGeneration,
    tok: T5Tokenizer,
    prompts: List[str],
    y_true: List[str],
    n_gen_per_prompt: int = 1,
    top_k: int = 5,
    max_out_len: int = 200,
):  # flan-t5

    inp = tok(prompts, padding=True, return_tensors="pt").to("cuda")  # right padding for T5

    torch_dataset = data.TensorDataset(inp['input_ids'], inp['attention_mask'])
    loader = data.DataLoader(
        dataset=torch_dataset,
        batch_size=32,
        shuffle=None,
        num_workers=0
    )

    out_preds = []
    for input_ids, attention_mask in tqdm(loader):
        decoder_input_ids = tok([""] * input_ids.shape[0], return_tensors="pt").input_ids
        decoder_input_ids = model._shift_right(decoder_input_ids)
        logits = model(input_ids, attention_mask, decoder_input_ids=decoder_input_ids).logits  # [batch_size, 1, 32128]
        logits = logits[:, 0, :]  # [batch_size, 32128]
        out_prob = torch.softmax(logits, dim=1)
        p, preds = torch.max(out_prob, dim=1)
        out_preds.append(preds)
    out_preds = torch.cat(out_preds)

    y_pred = []
    for out_i in out_preds:
        prediction = tok.decode(out_i)
        if prediction == "Yes" or prediction == "yes":
            pred_label = 1
        elif prediction == "No" or prediction == "no":
            pred_label = 0
        y_pred.append(pred_label)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="binary")
    recall = recall_score(y_true, y_pred, average="binary")
    f1 = f1_score(y_true, y_pred, average="binary")
    print("Acc: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(accuracy, precision, recall, f1))

    y_true_pos_idx = [idx for idx, y_each in enumerate(y_true) if y_each == 1]
    y_true_neg_idx = [idx for idx, y_each in enumerate(y_true) if y_each == 0]
    random.shuffle(y_true_pos_idx)
    random.shuffle(y_true_neg_idx)
    pos_max_len = len(y_true_pos_idx) if len(y_true_pos_idx) < 10 else 10
    neg_max_len = len(y_true_neg_idx) if len(y_true_neg_idx) < 10 else 10
    shuffle_order = y_true_pos_idx[:pos_max_len] + y_true_neg_idx[:neg_max_len]

    txt = []
    for idx in shuffle_order[:20]:
        prompt = prompts[idx]
        real_target = "Yes" if y_true[idx] == 1 else "No"
        predict_target = tok.decode(out_preds[idx])
        txt.append("{}: real_answer:{}, predict_answer:{}".format(prompt, real_target, predict_target))

    return txt
