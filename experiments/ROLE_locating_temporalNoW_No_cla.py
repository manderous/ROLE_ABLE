import argparse
import json
import os
import re
from collections import defaultdict

import numpy
import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration  # tjy

from dsets import KnownsDataset_t5_xl_MAVEN_temporalNo_cla  # tjy
from rome.tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)
from util import nethook
from util.globals import DATA_DIR
from util.runningstats import Covariance, tally

from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.cuda.device_count()


def main():
    parser = argparse.ArgumentParser(description="Causal Tracing")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    def parse_noise_rule(code):
        if code in ["m", "s"]:
            return code
        elif re.match("^[uts][\d\.]+", code):
            return code
        else:
            return float(code)

    aa(
        "--model_name",
        default="/home/jingyao2/PycharmProjects/rome-main/flan-t5-large",  # tjy flan-t5-large
        choices=[
            "gpt2-xl",
            "EleutherAI/gpt-j-6B",
            "EleutherAI/gpt-neox-20b",
            "gpt2-large",
            "gpt2-medium",
            "gpt2",
            "flan-t5-large",  # tjy
            "flan-t5-xl",  # tjy
        ],
    )
    aa("--fact_file", default=None)
    aa("--output_dir", default="results/{model_name}/MAVEN/temporalNoW_No/classification")  # tjy: 这个文件夹记得随时更改，用于分类不同的任务
    aa("--noise_level", default="s3", type=parse_noise_rule)
    aa("--replace", default=0, type=int)
    args = parser.parse_args()

    modeldir = f'r{args.replace}_{args.model_name.replace("/", "_")}'
    modeldir = f"n{args.noise_level}_" + modeldir
    output_dir = args.output_dir.format(model_name=modeldir)
    result_dir = f"{output_dir}/cases"
    pdf_dir = f"{output_dir}/pdfs"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    # Half precision to let the 20b model fit.
    torch_dtype = torch.float16 if "20b" in args.model_name else None

    mt = ModelAndTokenizer(args.model_name, torch_dtype=torch_dtype)

    if args.fact_file is None:
        knowns = KnownsDataset_t5_xl_MAVEN_temporalNo_cla(DATA_DIR)  # tjy
    else:
        with open(args.fact_file) as f:
            knowns = json.load(f)

    noise_level = args.noise_level
    uniform_noise = False
    if isinstance(noise_level, str):
        if noise_level.startswith("s"):
            # Automatic spherical gaussian
            factor = float(noise_level[1:]) if len(noise_level) > 1 else 1.0
            noise_level = factor * collect_embedding_std(
                mt, [k["subject"] for k in knowns]
            )
            print(f"Using noise_level {noise_level} to match model times {factor}")
        elif noise_level == "m":
            # Automatic multivariate gaussian
            noise_level = collect_embedding_gaussian(mt)
            print(f"Using multivariate gaussian to match model noise")
        elif noise_level.startswith("t"):
            # Automatic d-distribution with d degrees of freedom
            degrees = float(noise_level[1:])
            noise_level = collect_embedding_tdist(mt, degrees)
        elif noise_level.startswith("u"):
            uniform_noise = True
            noise_level = float(noise_level[1:])

    for knowledge in tqdm(knowns):
        known_id = knowledge["known_id"]
        for kind in None, "layer.0.SelfAttention", "layer.1.DenseReluDense", "layer.1.EncDecAttention", "layer.2.DenseReluDense":
            kind_suffix = f"_{kind}" if kind else ""
            filename = f"{result_dir}/knowledge_{known_id}{kind_suffix}.npz"
            if not os.path.isfile(filename):
                result = calculate_hidden_flow(
                    mt,
                    knowledge["prompt"],
                    knowledge["subject"],
                    expect="Yes",  # 现在我想看负样本预测为Yes的结果可视化
                    kind=kind,
                    noise=noise_level,
                    uniform_noise=uniform_noise,
                    replace=args.replace,
                )
                numpy_result = {
                    k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                    for k, v in result.items()
                }
                numpy.savez(filename, **numpy_result)
            else:
                numpy_result = numpy.load(filename, allow_pickle=True)
            if not numpy_result["correct_prediction"]:
                tqdm.write(f"Skipping {knowledge['prompt']}")
                continue
            plot_result = dict(numpy_result)
            plot_result["kind"] = kind
            pdfname = f'{pdf_dir}/{str(numpy_result["answer"]).strip()}_{known_id}{kind_suffix}.pdf'
            if known_id > 500:
                continue
            plot_trace_heatmap(plot_result, savepdf=pdfname)


def trace_with_patch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    uniform_noise=False,
    replace=False,  # True to replace with instead of add noise
    trace_layers=None,  # List of traced outputs to return
):
    """
    Runs a single causal trace.  Given a model and a batch input where
    the batch size is at least two, runs the batch in inference, corrupting
    a the set of runs [1...n] while also restoring a set of hidden states to
    the values from an uncorrupted run [0] in the batch.

    The convention used by this function is that the zeroth element of the
    batch is the uncorrupted run, and the subsequent elements of the batch
    are the corrupted runs.  The argument tokens_to_mix specifies an
    be corrupted by adding Gaussian noise to the embedding for the batch
    inputs other than the first element in the batch.  Alternately,
    subsequent runs could be corrupted by simply providing different
    input tokens via the passed input batch.

    Then when running, a specified set of hidden states will be uncorrupted
    by restoring their values to the same vector that they had in the
    zeroth uncorrupted run.  This set of hidden states is listed in
    states_to_patch, by listing [(token_index, layername), ...] pairs.
    To trace the effect of just a single state, this can be just a single
    token/layer pair.  To trace the effect of restoring a set of states,
    any number of token indices and layers can be listed.
    """

    rs = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)

    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)

    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    if isinstance(noise, float):
        noise_fn = lambda x: noise * x
    else:
        noise_fn = noise

    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                noise_data = noise_fn(
                    torch.from_numpy(prng(x.shape[0] - 1, e - b, x.shape[2]))
                ).to(x.device)
                if replace:  # tjy: original
                    x[1:, b:e] = noise_data
                elif x.shape[1] < 7:  # tjy: 解码层输出最大长度为6
                    x = x  # tjy: 若为解码层的输入，保持不变，不加噪声
                else:  # tjy: original
                    x[1:, b:e] += noise_data
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            if layer.split('.')[0] == 'encoder':  # tjy: 对于编码层，直接用原始值恢复被毁值即可
                h[1:, t] = h[0, t]
            if layer.split('.')[0] == 'decoder' and t < h.shape[1]:  # tjy: 对于解码层，只能恢复解码层输入token长度以内的模块
                h[1:, t] = h[0, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,
        edit_output=patch_rep,
    ) as td:
        # outputs_exp = model(**inp)  # tjy: original
        outputs_exp = model.generate(**inp, max_length=6, return_dict_in_generate=True, output_scores=True)  # tjy: flan-t5

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp[1][:-1][0][1:], dim=1).mean(dim=0)[answers_t]  # tjy: flan-t5
    # probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]  # tjy: original

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack([untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2)
        return probs, all_traced

    return probs


def trace_with_repatch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    states_to_unpatch,  # A list of (token index, layername) triples to re-randomize
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    uniform_noise=False,
):
    rs = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    unpatch_spec = defaultdict(list)
    for t, l in states_to_unpatch:
        unpatch_spec[l].append(t)

    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if first_pass or (layer not in patch_spec and layer not in unpatch_spec):
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec.get(layer, []):
            h[1:, t] = h[0, t]
        for t in unpatch_spec.get(layer, []):
            h[1:, t] = untuple(first_pass_trace[layer].output)[1:, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    for first_pass in [True, False] if states_to_unpatch else [False]:
        with torch.no_grad(), nethook.TraceDict(
            model,
            [embed_layername] + list(patch_spec.keys()) + list(unpatch_spec.keys()),
            edit_output=patch_rep,
        ) as td:
            outputs_exp = model(**inp)
            if first_pass:
                first_pass_trace = td

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    return probs


def calculate_hidden_flow(
    mt,
    prompt,
    subject,
    samples=10,
    noise=0.1,
    token_range=None,
    uniform_noise=False,
    replace=False,
    window=10,
    kind=None,
    expect=None,
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))
    No_id = 465  # "No"在Flan-T5-large中词典的id
    with torch.no_grad():
        answer_t, base_score, No_score = [d[0] for d in predict_from_input(mt.model, inp, No_id)]  # tjy: flan-t5
    [answer] = decode_tokens(mt.tokenizer, [answer_t])
    if expect is not None and answer.strip() != expect:  # tjy: 筛选预测结果为expect(Yes)的样本进行后面的可视化计算
        return dict(correct_prediction=False)
    answer_t = No_id  # tjy: 替换Yes预测值为No
    base_score = No_score  # tjy: 替换Yes预测值为No
    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
    if token_range == "subject_last":
        token_range = [e_range[1] - 1]
    elif token_range is not None:
        raise ValueError(f"Unknown token_range: {token_range}")
    low_score = trace_with_patch(
        mt.model, inp, [], answer_t, e_range, noise=noise, uniform_noise=uniform_noise
    ).item()
    if not kind:
        differences = trace_important_states(
            mt.model,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            token_range=token_range,
        )
    else:
        differences = trace_important_window(
            mt.model,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            window=window,
            kind=kind,
            token_range=token_range,
        )
    differences = differences.detach().cpu()
    return dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        subject_range=e_range,
        answer=answer,
        window=window,
        correct_prediction=True,
        kind=kind or "",
    )


def trace_important_states(
    model,
    num_layers,
    inp,
    e_range,
    answer_t,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    token_range=None,
):
    ntoks = inp["input_ids"].shape[1]
    table = []

    if token_range is None:
        token_range = range(ntoks)
    for tnum in tqdm(token_range):
        row = []
        for layer in range(num_layers):
            r = trace_with_patch(
                model,
                inp,
                [(tnum, layername(model, layer))],
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )
            row.append(r)
        table.append(torch.stack(row))
    print('完成trace_important_states')
    return torch.stack(table)


def trace_important_window(
    model,
    num_layers,
    inp,
    e_range,
    answer_t,
    kind,
    window=10,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    token_range=None,
):
    ntoks = inp["input_ids"].shape[1]
    table = []

    if token_range is None:
        token_range = range(ntoks)
    for tnum in tqdm(token_range):
        # if tnum > 3 and (kind == "layer.1.EncDecAttention" or kind == "layer.2.DenseReluDense"):
        #     break  # tjy: 在探索flan-T5-large的decoder时，我们仅需追踪第一个输入的token即可，因为decoder仅输出Yes仅停止了
        row = []
        for layer in range(num_layers):
            layerlist = [
                (tnum, layername(model, L, kind))
                for L in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                )
            ]
            continue_flag = False  # tjy: 初始化continue标识符
            for layer_tuple in layerlist:  # tjy: 判断layer列表中是否有None
                if 'None' in layer_tuple:  # tjy: 若有None，就跳过当前追踪，这主要考虑到encoder和decoder的模块命名不一样
                    continue_flag = True  # tjy
                    break  # tjy
            if continue_flag:  # tjy
                continue  # tjy: 若有None，就跳过当前追踪，这主要考虑到encoder和decoder的模块命名不一样
            r = trace_with_patch(
                model,
                inp,
                layerlist,
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )
            row.append(r)
        table.append(torch.stack(row))
    print('完成trace_important_window')
    return torch.stack(table)


class ModelAndTokenizer:
    """
    An object to hold on to (or automatically download and hold)
    a GPT-style language model and tokenizer.  Counts the number
    of layers.
    """

    def __init__(
        self,
        model_name=None,
        model=None,
        tokenizer=None,
        low_cpu_mem_usage=False,
        torch_dtype=None,
    ):
        if tokenizer is None:
            assert model_name is not None
            # tokenizer = AutoTokenizer.from_pretrained(model_name)  # tjy
            tokenizer = T5Tokenizer.from_pretrained(model_name)  # tjy 使用flan-t5-large
        if model is None:
            assert model_name is not None
            # model = AutoModelForCausalLM.from_pretrained(
            #     model_name, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype
            # )  # tjy
            model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")  # tjy 使用flan-t5-large
            nethook.set_requires_grad(False, model)
            model.eval().cuda()
        self.tokenizer = tokenizer
        self.model = model
        self.layer_names = [
            n
            for n, m in model.named_modules()
            if (re.match(r"^(transformer|gpt_neox|encoder|decoder)\.(h|layers|block)\.\d+$", n))
        ]  # encoder|decoder, block是flan-t5的参数名
        self.num_layers = len(self.layer_names)

    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )


def layername(model, num, kind=None):
    if hasattr(model, "shared"):
        if kind == "embed":
            return "shared"
        if num < 24:  # tjy: 0-23层为编码层
            if kind is None or kind == "layer.0.SelfAttention" or kind == "layer.1.DenseReluDense":  # tjy: flan-t5
                return f'encoder.block.{num}{"" if kind is None else "." + kind}'  # tjy: flan-t5
            else:  # tjy: flan-t5
                return "None"  # tjy: flan-t5
        else:  # tjy: 24-47为解码层
            if kind is None or kind == "layer.0.SelfAttention" or kind == "layer.1.EncDecAttention" or kind == "layer.2.DenseReluDense":  # tjy: flan-t5
                return f'decoder.block.{num-24}{"" if kind is None else "." + kind}'  # tjy: flan-t5
            else:  # tjy: flan-t5
                return "None"  # tjy: flan-t5
    if hasattr(model, "transformer"):
        if kind == "embed":
            return "transformer.wte"
        return f'transformer.h.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "gpt_neox"):
        if kind == "embed":
            return "gpt_neox.embed_in"
        if kind == "attn":
            kind = "attention"
        return f'gpt_neox.layers.{num}{"" if kind is None else "." + kind}'
    assert False, "unknown transformer structure"


def guess_subject(prompt):
    return re.search(r"(?!Wh(o|at|ere|en|ich|y) )([A-Z]\S*)(\s[A-Z][a-z']*)*", prompt)[
        0
    ].strip()


def plot_hidden_flow(
    mt,
    prompt,
    subject=None,
    samples=10,
    noise=0.1,
    uniform_noise=False,
    window=10,
    kind=None,
    savepdf=None,
):
    if subject is None:
        subject = guess_subject(prompt)
    result = calculate_hidden_flow(
        mt,
        prompt,
        subject,
        samples=samples,
        noise=noise,
        uniform_noise=uniform_noise,
        window=window,
        kind=kind,
    )
    plot_trace_heatmap(result, savepdf)


def plot_trace_heatmap(result, savepdf=None, title=None, xlabel=None, modelname=None):
    differences = result["scores"]
    low_score = result["low_score"]
    answer = result["answer"]
    kind = (
        None
        if (not result["kind"] or result["kind"] == "None")
        else str(result["kind"])
    )
    window = result.get("window", 10)
    labels = list(result["input_tokens"])
    for i in range(*result["subject_range"]):
        labels[i] = labels[i] + "*"

    with plt.rc_context(rc={"font.family": "Times New Roman"}):
        fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
        h = ax.pcolor(
            differences,
            cmap={None: "Purples", "None": "Purples", "layer.0.SelfAttention": "Greens",   # tjy: flan-t5
                  "layer.1.DenseReluDense": "Reds", "layer.1.EncDecAttention": "Oranges",   # tjy: flan-t5
                  "layer.2.DenseReluDense": "Blues"}[kind],  # tjy: flan-t5
            vmin=low_score,
        )
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
        ax.set_yticklabels(labels, fontsize=2)  # tjy: 修改y轴的刻度值大小
        if not modelname:
            modelname = "GPT"
        if not kind:
            ax.set_title("Impact of restoring state after corrupted input")
            ax.set_xlabel(f"single restored layer within {modelname}")
        else:
            # kindname = "MLP" if kind == "mlp" else "Attn"  # tjy: original
            kindname = kind  # tjy: flan-t5
            ax.set_title(f"Impact of restoring {kindname} after corrupted input")
            ax.set_xlabel(f"center of interval of {window} restored {kindname} layers")
        cb = plt.colorbar(h)
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        elif answer is not None:
            # The following should be cb.ax.set_xlabel, but this is broken in matplotlib 3.5.1.
            cb.ax.set_title(f"p({str(answer).strip()})", y=-0.16, fontsize=10)
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def plot_all_flow(mt, prompt, subject=None):
    for kind in ["mlp", "attn", None]:
        plot_hidden_flow(mt, prompt, subject, kind=kind)


# Utilities for dealing with tokens
def make_inputs(tokenizer, prompts, device="cuda"):
    token_lists = [tokenizer.encode(p) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    # position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        #    position_ids=torch.tensor(position_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )


def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def find_token_range(tokenizer, token_array, substring):
    toks = decode_tokens(tokenizer, token_array)  # tjy: flan-t5
    toks_substring = decode_tokens(tokenizer, tokenizer.encode(substring))[:-1]  # tjy: flan-t5
    tok_start, tok_end = None, None  # tjy: flan-t5
    for i in range(len(toks)-len(toks_substring)+1):  # tjy: flan-t5
        if toks_substring == toks[i:i+len(toks_substring)]:  # tjy: flan-t5
            tok_start = i  # tjy: flan-t5
            tok_end = tok_start + len(toks_substring)  # tjy: flan-t5
    return (tok_start, tok_end)  # tjy: flan-t5
    # whole_string = " ".join(toks)  # tjy: original
    # char_loc = whole_string.index(substring)  # tjy: original
    # loc = 0  # tjy: original
    # tok_start, tok_end = None, None  # tjy: original
    # for i, t in enumerate(toks):  # tjy: original
    #     loc += len(t)  # tjy: original
    #     if tok_start is None and loc > char_loc:  # tjy: original
    #         tok_start = i  # tjy: original
    #     if tok_end is None and loc >= char_loc + len(substring):  # tjy: original
    #         tok_end = i + 1  # tjy: original
    #         break  # tjy: original
    # return (tok_start, tok_end)  # tjy: original


# def predict_token(mt, prompts, return_p=False):
#     inp = make_inputs(mt.tokenizer, prompts)
#     preds, p, _, _ = predict_from_input(mt.model, inp)  # tjy: flan-t5
#     result = [mt.tokenizer.decode(c) for c in preds]
#     if return_p:
#         result = (result, p)
#     return result


def predict_from_input(model, inp, No_id):
    out = model.generate(**inp, max_length=6, return_dict_in_generate=True, output_scores=True)  # tjy: flan-t5
    out_ps = []  # tjy: flan-t5
    out_preds = []  # tjy: flan-t5
    out_probs = []
    for out_row in out[1]:  # tjy: flan-t5
        out_prob = torch.softmax(out_row, dim=1)  # tjy: flan-t5
        p, preds = torch.max(out_prob, dim=1)  # tjy: flan-t5
        out_ps.append(p)  # tjy: flan-t5
        out_preds.append(preds)  # tjy: flan-t5
        out_probs.append(out_prob)  # tjy: flan-t5

    return out_preds[:-1][0], out_ps[:-1][0], out_probs[0][:, No_id]
    # return out_preds[:-1][0], out_ps[:-1][0], No_preds[:-1][0], No_ps[:-1][0]  # tjy: flan-t5
    # out = model(**inp)["logits"]  # tjy: original
    # probs = torch.softmax(out[:, -1], dim=1)  # tjy: original
    # p, preds = torch.max(probs, dim=1)  # tjy: original
    # return preds, p  # tjy: original


def collect_embedding_std(mt, subjects):
    alldata = []
    inp = make_inputs(mt.tokenizer, [subjects[0]])
    with nethook.Trace(mt.model, layername(mt.model, 0, "embed")) as t:
        # mt.model(**inp)  # tjy: original
        mt.model.generate(**inp, max_new_tokens=6)  # tjy: flan-t5
        alldata.append(t.output[0])
    alldata = torch.cat(alldata)
    alldata = alldata.repeat(1, len(subjects))
    noise_level = alldata.std().item()
    return noise_level


def get_embedding_cov(mt):
    model = mt.model
    tokenizer = mt.tokenizer

    def get_ds():
        ds_name = "wikitext"
        raw_ds = load_dataset(
            ds_name,
            dict(wikitext="wikitext-103-raw-v1", wikipedia="20200501.en")[ds_name],
        )
        try:
            maxlen = model.config.n_positions
        except:
            maxlen = 100  # Hack due to missing setting in GPT2-NeoX.
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    ds = get_ds()
    sample_size = 1000
    batch_size = 5
    filename = None
    batch_tokens = 100

    progress = lambda x, **k: x

    stat = Covariance()
    loader = tally(
        stat,
        ds,
        cache=filename,
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=0,
    )
    with torch.no_grad():
        for batch_group in loader:
            for batch in batch_group:
                batch = dict_to_(batch, "cuda")
                del batch["position_ids"]
                with nethook.Trace(model, layername(mt.model, 0, "embed")) as tr:
                    model(**batch)
                feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                stat.add(feats.cpu().double())
    return stat.mean(), stat.covariance()


def make_generator_transform(mean=None, cov=None):
    d = len(mean) if mean is not None else len(cov)
    device = mean.device if mean is not None else cov.device
    layer = torch.nn.Linear(d, d, dtype=torch.double)
    nethook.set_requires_grad(False, layer)
    layer.to(device)
    layer.bias[...] = 0 if mean is None else mean
    if cov is None:
        layer.weight[...] = torch.eye(d).to(device)
    else:
        _, s, v = cov.svd()
        w = s.sqrt()[None, :] * v
        layer.weight[...] = w
    return layer


def collect_embedding_gaussian(mt):
    m, c = get_embedding_cov(mt)
    return make_generator_transform(m, c)


def collect_embedding_tdist(mt, degree=3):
    # We will sample sqrt(degree / u) * sample, where u is from the chi2[degree] dist.
    # And this will give us variance is (degree / degree - 2) * cov.
    # Therefore if we want to match the sample variance, we should
    # reduce cov by a factor of (degree - 2) / degree.
    # In other words we should be sampling sqrt(degree - 2 / u) * sample.
    u_sample = torch.from_numpy(
        numpy.random.RandomState(2).chisquare(df=degree, size=1000)
    )
    fixed_sample = ((degree - 2) / u_sample).sqrt()
    mvg = collect_embedding_gaussian(mt)

    def normal_to_student(x):
        gauss = mvg(x)
        size = gauss.shape[:-1].numel()
        factor = fixed_sample[:size].reshape(gauss.shape[:-1] + (1,))
        student = factor * gauss
        return student

    return normal_to_student


if __name__ == "__main__":
    main()
