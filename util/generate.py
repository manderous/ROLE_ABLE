import unicodedata
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration  # tjy
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix  # tjy

from util.logit_lens import LogitLens
import torch.utils.data as data
from tqdm import tqdm
import random
from util import nethook
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def generate_fast(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    n_gen_per_prompt: int = 1,
    top_k: int = 5,
    max_out_len: int = 200,
):
    """
    Fast, parallelized auto-regressive text generation with top-k sampling.
    Our custom implementation.
    """

    # Unroll prompts and tokenize
    inp = [prompt for prompt in prompts for _ in range(n_gen_per_prompt)]
    inp_tok = tok(inp, padding=True, return_tensors="pt").to(
        next(model.parameters()).device
    )
    input_ids, attention_mask = inp_tok["input_ids"], inp_tok["attention_mask"]
    batch_size = input_ids.size(0)

    # Setup storage of fast generation with attention caches.
    # `cur_context` is used to define the range of inputs that are not yet
    # stored in `past_key_values`. At each step, we are generating the
    # next token for the index at `cur_context.stop + 1`.
    past_key_values, cur_context = None, slice(0, attention_mask.sum(1).min().item())

    with torch.no_grad():
        while input_ids.size(1) < max_out_len:  # while not exceeding max output length
            model_out = model(
                input_ids=input_ids[:, cur_context],
                attention_mask=attention_mask[:, cur_context],
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits, past_key_values = model_out.logits, model_out.past_key_values
            softmax_out = torch.nn.functional.softmax(logits[:, -1, :], dim=1)

            # Top-k sampling
            tk = torch.topk(softmax_out, top_k, dim=1).indices
            softmax_out_top_k = torch.gather(softmax_out, 1, tk)
            softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]
            new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
            new_toks = torch.gather(tk, 1, new_tok_indices)

            # If we're currently generating the continuation for the last token in `input_ids`,
            # create a new index so we can insert the new token
            if cur_context.stop == input_ids.size(1):
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_zeros(batch_size, 1)], dim=1
                )
                input_ids = torch.cat(
                    [
                        input_ids,
                        input_ids.new_ones(batch_size, 1) * tok.pad_token_id,
                    ],
                    dim=1,
                )

            last_non_masked = attention_mask.sum(1) - 1
            for i in range(batch_size):
                new_idx = last_non_masked[i] + 1
                if last_non_masked[i].item() + 1 != cur_context.stop:
                    continue

                # Stop generating if we've already maxed out for this prompt
                if new_idx < max_out_len:
                    input_ids[i][new_idx] = new_toks[i]
                    attention_mask[i][new_idx] = 1

            cur_context = slice(cur_context.stop, cur_context.stop + 1)

    txt = [tok.decode(x) for x in input_ids.detach().cpu().numpy().tolist()]
    txt = [
        unicodedata.normalize("NFKD", x)
        .replace("\n\n", " ")
        .replace("<|endoftext|>", "")
        for x in txt
    ]

    return txt


def generate_T5(
    model: T5ForConditionalGeneration,
    tok: T5Tokenizer,
    prompts: List[str],
    y_true: List[str],
    request_yes_no: List[str],
    n_gen_per_prompt: int = 1,
    top_k: int = 5,
    max_out_len: int = 200,
):  # tjy: flan-t5

    inp = tok(prompts, padding=True, return_tensors="pt").to("cuda")  # t5的tokenizer默认是right padding
    # 之前可视化的代码作者用的是left padding应对gpt2-xl，这个left/right padding的问题需要特别注意

    torch_dataset = data.TensorDataset(inp['input_ids'], inp['attention_mask'])
    loader = data.DataLoader(
        dataset=torch_dataset,
        batch_size=8,  # 每批提取的数量  ESL:32, CTB, SCI:128
        shuffle=None,  # 要不要打乱数据（打乱比较好）
        num_workers=0  # 多少线程来读取数据
    )

    out_preds = []
    for input_ids, attention_mask in tqdm(loader):
        # 利用flan-t5模型，根据提示，生成对于的问题回答
        decoder_input_ids = tok([""] * input_ids.shape[0], return_tensors="pt").input_ids  # 设置decoder的输入
        decoder_input_ids = model._shift_right(decoder_input_ids)  # 输入<pad>，id是0
        logits = model(input_ids, attention_mask, decoder_input_ids=decoder_input_ids).logits  # [batch_size, 1, 32128]
        logits = logits[:, 0, :]  # [batch_size, 32128]
        out_prob = torch.softmax(logits, dim=1)
        p, preds = torch.max(out_prob, dim=1)
        out_preds.append(preds)
    out_preds = torch.cat(out_preds)

    # out_preds = []
    # for input_ids, attention_mask in tqdm(loader):
    #     # 利用flan-t5模型，根据提示，生成对于的问题回答
    #     out = model.generate(input_ids, attention_mask=attention_mask, max_length=max_out_len, return_dict_in_generate=True, output_scores=True)
    #     out_prob = torch.softmax(out[1][0], dim=1)
    #     p, preds = torch.max(out_prob, dim=1)
    #     out_preds.append(preds)
    # out_preds = torch.cat(out_preds)

    y_pred = []
    for out_i in out_preds:
        prediction = tok.decode(out_i)
        if prediction == "Yes" or prediction == "yes":
            pred_label = 1
        elif prediction == "No" or prediction == "no":
        # else:
            pred_label = 0
        y_pred.append(pred_label)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="binary")
    recall = recall_score(y_true, y_pred, average="binary")
    f1 = f1_score(y_true, y_pred, average="binary")
    # cm_plot(y_true, y_pred)  # 画混淆矩阵

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

    # txt = copy.deepcopy(prompts)  # 深拷贝列表，前拷贝列表会导致prompts会随着txt的变化而变化
    # for out_i in out_preds:
    #     for prompt_id, prompt_out_i in enumerate(out_i):
    #         txt[prompt_id] += (" " + tok.decode(prompt_out_i))  # 把T5生成的结果附在每个prompt后面

    # print('—————————————————————————————— TEST: request_yes_no ——————————————————————————————')
    # request_yes_no = [one_row['prompt'] for one_row in request_yes_no]
    # inp = tok(request_yes_no, padding=True, return_tensors="pt").to("cuda")  # t5的tokenizer默认是right padding
    #
    # decoder_input_ids = tok([""] * len(request_yes_no), return_tensors="pt").input_ids  # 设置decoder的输入
    # decoder_input_ids = model._shift_right(decoder_input_ids)  # 输入<pad>，id是0
    # logits = model(**inp, decoder_input_ids=decoder_input_ids).logits  # [batch_size, 1, 32128]
    #
    # logits = logits[:, 0, :]  # [batch_size, 32128]
    # out_prob = torch.softmax(logits, dim=1)
    # p, preds = torch.max(out_prob, dim=1)
    # for idx, out_i in enumerate(preds):
    #     yes_or_no = tok.decode(out_i)
    #     print(yes_or_no, ' :', out_prob[idx][465])
    # print('—————————————————————————————— OVER ——————————————————————————————')

    return txt


def generate_T5_multi(
    model: T5ForConditionalGeneration,
    tok: T5Tokenizer,
    prompts: List[str],
    y_true: List[str],
    n_gen_per_prompt: int = 1,
    top_k: int = 5,
    max_out_len: int = 200,
):  # tjy: flan-t5

    inp = tok(prompts, padding=True, return_tensors="pt").to(device)  # t5的tokenizer默认是right padding
    # 之前可视化的代码作者用的是left padding应对gpt2-xl，这个left/right padding的问题需要特别注意

    torch_dataset = data.TensorDataset(inp['input_ids'], inp['attention_mask'])
    loader = data.DataLoader(
        dataset=torch_dataset,
        batch_size=32,  # 每批提取的数量  ESL:32, CTB, SCI:128
        shuffle=None,  # 要不要打乱数据（打乱比较好）
        num_workers=0  # 多少线程来读取数据
    )

    out_preds = []
    for input_ids, attention_mask in tqdm(loader):
        # 利用flan-t5模型，根据提示，生成对于的问题回答
        decoder_input_ids = tok([""] * input_ids.shape[0], return_tensors="pt").input_ids  # 设置decoder的输入
        decoder_input_ids = model._shift_right(decoder_input_ids)  # 输入<pad>，id是0
        logits = model(input_ids.to(device), attention_mask.to(device), decoder_input_ids=decoder_input_ids.to(device)).logits  # [batch_size, 1, 32128]
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
        else:
            a = 0  # 这个结果需要debug
        y_pred.append(pred_label)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="binary")
    recall = recall_score(y_true, y_pred, average="binary")
    f1 = f1_score(y_true, y_pred, average="binary")
    # cm_plot(y_true, y_pred)  # 画混淆矩阵

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
