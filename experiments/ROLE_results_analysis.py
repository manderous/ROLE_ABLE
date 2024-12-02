import numpy, os
from matplotlib import pyplot as plt
import copy
from tqdm import tqdm
import math

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams.update({'font.size': 20})
arch = "flan-t5-large"  # for flan-T5-xl
archname = "Flan-T5-large"  # for flan-t5-large


class Avg:
    def __init__(self):
        self.d = []

    def add(self, v):
        self.d.append(v[None])

    def add_all(self, vv):
        self.d.append(vv)

    def avg(self):
        return numpy.concatenate(self.d).mean(axis=0)

    def std(self):
        return numpy.concatenate(self.d).std(axis=0)

    def size(self):
        return sum(datum.shape[0] for datum in self.d)


def read_knowlege(multi_task_name, count=150, kind=None, arch="gpt2-xl"):
    print('Encoder ongoing task:', multi_task_name, ", module: ", kind)
    dirname = f"results/{arch}/{multi_task_name}/cases/"
    kindcode = "" if not kind else f"_{kind}"
    (avg_fe, avg_ee, avg_le, avg_cc, avg_fa, avg_ea, avg_la, avg_hs, avg_ls, avg_fs, avg_fle, avg_fla,) = [Avg() for _ in range(12)]
    relation_count = 0  # 统计数量，初始化
    for i in tqdm(range(count)):
        try:
            data = numpy.load(f"{dirname}/knowledge_{i}{kindcode}.npz")
        except:
            continue
        # Only consider cases where the model begins with the correct prediction
        if "correct_prediction" in data and not data["correct_prediction"]:
            continue
        relation_count += 1
        scores = data["scores"]
        first_e, first_a = data["subject_range"]
        last_e = first_a - 1
        last_a = len(scores) - 1
        # original prediction
        avg_hs.add(data["high_score"])
        # prediction after subject is corrupted
        avg_ls.add(data["low_score"])
        avg_fs.add(scores.max())
        # some maximum computations
        avg_fle.add(scores[last_e].max())
        avg_fla.add(scores[last_a].max())

        avg_fe.add_all(scores[:9])  # Answering the following yes/no question.
        avg_ee.add_all(scores[9:11])  # Is
        avg_le.add_all(scores[11:14])  # there a
        sentence_index = data['input_tokens'].tolist().index('sentence')  # "sentence"的位置
        if "causal" in multi_task_name.split('/')[1]:
            avg_cc.add(scores[14])  # causal
            avg_fa.add(scores[15])  # relation
            avg_ea.add_all(scores[16:sentence_index+1])  # (between A and B) in sentence
            avg_la.add_all(scores[sentence_index+1:])  # <context>?
        elif "temporal" in multi_task_name.split('/')[1]:
            avg_cc.add_all(scores[14:16])  # temporal
            avg_fa.add(scores[16])  # relation
            avg_ea.add_all(scores[17:sentence_index + 1])  # (between A and B) in sentence
            avg_la.add_all(scores[sentence_index + 1:])  # <context>?
        elif "subevent" in multi_task_name.split('/')[1]:
            avg_cc.add_all(scores[14:17])  # subevent
            avg_fa.add(scores[17])  # relation
            avg_ea.add_all(scores[18:sentence_index + 1])  # (between A and B) in sentence
            avg_la.add_all(scores[sentence_index + 1:])  # <context>?

    result = numpy.stack([avg_fe.avg(), avg_ee.avg(), avg_le.avg(), avg_cc.avg(), avg_fa.avg(), avg_ea.avg(), avg_la.avg(),])
    result_std = numpy.stack([avg_fe.std(), avg_ee.std(), avg_le.std(), avg_cc.std(), avg_fa.std(), avg_ea.std(), avg_la.std(),])
    print("Task", multi_task_name, "has", relation_count, "samples to satisfy the requirement")
    return dict(low_score=avg_ls.avg(), result=result, result_std=result_std, size=avg_fe.size())


def read_knowlege_decoder(multi_task_name, count=150, kind=None, arch="gpt2-xl"):
    print('Decoder ongoing task:', multi_task_name, ", module: ", kind)
    dirname = f"results/{arch}/{multi_task_name}/cases/"
    kindcode = "" if not kind else f"_{kind}"
    (avg_fe, avg_hs, avg_ls,) = [Avg() for _ in range(3)]
    for i in tqdm(range(count)):
        try:
            data = numpy.load(f"{dirname}/knowledge_{i}{kindcode}.npz")
        except:
            continue
        # Only consider cases where the model begins with the correct prediction
        if "correct_prediction" in data and not data["correct_prediction"]:
            continue
        scores = data["scores"]
        # original prediction
        avg_hs.add(data["high_score"])
        # prediction after subject is corrupted
        avg_ls.add(data["low_score"])

        avg_fe.add(scores[0])  # start token

    result = numpy.stack([avg_fe.avg()])
    result_std = numpy.stack([avg_fe.std()])
    return dict(low_score=avg_ls.avg(), result=result, result_std=result_std, size=avg_fe.size())


if __name__ == "__main__":
    the_count = 500  # flan-t5-large: TBD: 2020, CNC: 1603, CTB:318, MAVEN:500
    high_score = None  # Scale all plots according to the y axis of the first plot

    # choise：
    # The code simultaneously visualizes the following tasks:
    # multi_task_names = [A, B, C, D, E, F]
    # multi_task_names = ["causal_trace_TBD_inference_2", "causal_trace_TBD_2", "causal_trace_CNC_2", "causal_trace_CTB_2"]
    # multi_task_names = ["MAVEN/temporal/classification", "MAVEN/temporal/extraction", "MAVEN/causal/classification",
    #                     "MAVEN/causal/extraction", "MAVEN/subevent/classification", "MAVEN/subevent/extraction"]
    multi_task_names = ["MAVEN/temporalNo/classification", "MAVEN/temporalNo/extraction", "MAVEN/causalNo/classification",
                        "MAVEN/causalNo/extraction", "MAVEN/subeventNo/classification", "MAVEN/subeventNo/extraction"]
    # multi_task_names = ["MAVEN/temporalW_Yes/classification", "MAVEN/temporalW_Yes/extraction", "MAVEN/causalW_Yes/classification",
    #                     "MAVEN/causalW_Yes/extraction", "MAVEN/subeventW_Yes/classification", "MAVEN/subeventW_Yes/extraction"]
    # multi_task_names = ["MAVEN/temporalNoW_No/classification", "MAVEN/temporalNoW_No/extraction", "MAVEN/causalNoW_No/classification",
    #                     "MAVEN/causalNoW_No/extraction", "MAVEN/subeventNoW_No/classification", "MAVEN/subeventNoW_No/extraction"]
    # Remember to modify it before each run !
    # choise:[results_No, results_W_Yes, results_NoW_No, results, _another]
    request_type = "results_No"

    ########################## Image of T5-Encoder, heat map ##########################
    task_layers = [None, "layer.0.SelfAttention", "layer.1.DenseReluDense"]  # T5的Encoder考虑三个模块
    multi_task_results = []  # Initialize List
    for task_i, multi_task_name in enumerate(multi_task_names):
        task_results = []  # Initialize the list for each task
        for layer_j, kind in enumerate(task_layers):
            d = read_knowlege(multi_task_name, the_count, kind, arch)
            # Split the data by encoder and decoder and assign flag labels
            d_Enc = dict()
            d_Dec = dict()
            if kind == None or kind == "layer.0.SelfAttention":
                d_Enc = copy.deepcopy(d)
                d_Dec = copy.deepcopy(d)
                # Splitting the encoder (first 24 layers) and decoder (last 24 layers) of flan-t5-large
                # To align with MLP, I've also selected only the first 20 layers to visualize
                d_Enc['result'], d_Dec['result'] = numpy.split(d['result'], [20], axis=1)
                d_Dec['result'] = d_Dec['result'][0:1, :]  # result
                d_Enc['result_std'], d_Dec['result_std'] = numpy.split(d['result_std'], [20], axis=1)
                d_Dec['result_std'] = d_Dec['result_std'][0:1, :]  # result_std
            elif kind == "layer.1.DenseReluDense":
                d_Enc = copy.deepcopy(d)

            if "No" in request_type:
                # For negative samples, reverse the AIE calculation,
                # because negative samples should have a higher probability of predicting “No” with added noise.
                result = d_Enc["low_score"] - d_Enc["result"]
            else:
                result = d_Enc["result"] - d_Enc["low_score"]

            task_results.append(result)
        multi_task_results.append(task_results)

    task_layers = [None, "layer.0.SelfAttention", "layer.1.DenseReluDense"]  # Encoder for T5 considers three modules
    if "No" in request_type:
        fig, axes = plt.subplots(6, 3, figsize=(10, 12), sharey=False, sharex=True, dpi=200)
    else:
        fig, axes = plt.subplots(6, 3, figsize=(12, 12), sharey=False, sharex=True, dpi=200)
    title_dict = {
        None: "Encoder-Total",
        "layer.0.SelfAttention": "Encoder-Self-Attn",
        "layer.1.DenseReluDense": "Encoder-MLP",
    }
    for task_i, multi_task_name in enumerate(multi_task_names):
        for layer_j, kind in enumerate(task_layers):
            result = multi_task_results[task_i][layer_j]  # Get the results under the current task and layer type
            ax = axes[task_i, layer_j]
            h = ax.pcolor(
                result,
                cmap={None: "Purples", "None": "Purples", "layer.0.SelfAttention": "Greens",  # flan-t5
                      "layer.1.DenseReluDense": "Reds", "layer.1.EncDecAttention": "Oranges",  # flan-t5
                      "layer.2.DenseReluDense": "Blues"}[kind],  # flan-t5
                vmin=0.025,  # minimum value: vmin=0, 0.02, 0.025
                vmax=result.max(),  # maximum value: vmax=result.max(), 0.2
            )
            ax.invert_yaxis()
            if task_i == 0:
                ax.set_title(title_dict[kind])
            if task_i == 5:
                ax.set_xticks([0.5 + i for i in range(0, result.shape[1]-4, 5)])
                ax.set_xticklabels(list(range(0, result.shape[1]-4, 5)))

            ax.set_yticks([0.5 + i for i in range(len(result))])
            if layer_j == 0:
                if "No" in request_type:
                    ax.set_yticklabels(["", "", "", "", "", "", ""])  # show when negative
                else:
                    ax.set_yticklabels(["Answering ... question.",
                                        "Is", "there a", "CTS", "relation",
                                        "... in sentence",
                                        "< context >?", ])  # show when positive
            elif layer_j == 2:
                ax.set_yticklabels(["", "", "", "", "", "", ""])  # show when negative
                if "No" in request_type:
                    ax.set_ylabel(f"{multi_task_name.split('/')[1].split('N')[0]}-{multi_task_name.split('/')[2][:3]}",
                                  labelpad=-225, fontsize=15)  # show when positive
            else:
                ax.set_yticklabels(["", "", "", "", "", "", ""])

    savepdf = f"results/{arch}/MAVEN/{request_type}/summary/rollup_Enc_all_token_0.025.pdf"
    os.makedirs(os.path.dirname(savepdf), exist_ok=True)
    print("画图中")
    plt.tight_layout()  # 自动调节子图间距
    plt.savefig(savepdf, bbox_inches="tight")
    ########################## Drawing Completion ##########################

    ######################### Exploring the Decoder of T5: Plot line graph ##########################
    # To make confidence intervals visible, we plot the data as line graphs below.
    task_layers = [(None, "Decoder-Total"), ("layer.0.SelfAttention", "Decoder-Self-Attn"),
                   ("layer.1.EncDecAttention", "Decoder-Cross-Attn"),
                   ("layer.2.DenseReluDense", "Decoder-MLP")]

    if "No" in request_type:
        fig, axes = plt.subplots(6, 4, figsize=(12, 15), sharey=False, sharex=True, dpi=200)  # flan-t5-large
    else:
        fig, axes = plt.subplots(6, 4, figsize=(13, 15), sharey=False, sharex=True, dpi=200)  # flan-t5-large

    for task_i, multi_task_name in enumerate(multi_task_names):
        color_order = [0, 1, 2, 4, 5, 3]
        for layer_j, (kind, title) in enumerate(task_layers):
            print(f"Reading {kind}")
            d = read_knowlege_decoder(multi_task_name, the_count, kind, arch)
            # Start by selecting only layers 0-23 of the encoder of flan-t5-large for visualization
            d["result"] = d["result"][:, -24:][0:1, :]
            d["result_std"] = d["result_std"][:, -24:][0:1, :]
            if kind == None or kind == "layer.0.SelfAttention":
                d["result"] = d["result"][:, :20]  # Array length aligned to MLP and cross-attention
                d["result_std"] = d["result_std"][:, :20]  # Array length aligned to MLP and cross-attention
            count = d["size"]
            labels = ["s"]
            ax = axes[task_i, layer_j]
            for i, label in list(enumerate(labels)):
                if "No" in request_type:
                    # For negative samples, reverse the AIE calculation,
                    # because negative samples should have a higher probability of predicting “No” with added noise.
                    y = d["low_score"] - d["result"][i]
                else:
                    y = d["result"][i] - d["low_score"]

                x = list(range(len(y)))
                std = d["result_std"][i]
                error = std * 1.96 / math.sqrt(count)
                if "temporal" in multi_task_name.split('/')[1]:
                    color_name = "darkorange"
                elif "causal" in multi_task_name.split('/')[1]:
                    color_name = "deeppink"
                elif "subevent" in multi_task_name.split('/')[1]:
                    color_name = "forestgreen"

                ax.fill_between(x, y - error, y + error, alpha=0.3, color=color_name)
                ax.plot(x, y, label=label, color=color_name)

            if task_i == 0:
                ax.set_title(f"{title}")
            ax.set_ylim(-0.1, 0.4)  # (-0.1, 0.5), (-0.1, 0.6)
            ax.set_yticks([0.0, 0.1, 0.2, 0.3])
            if layer_j == 0:
                if "No" in request_type:
                    ax.set_yticklabels(["", "", "", ""])  # for negative samples
                else:
                    ax.set_ylabel(f"{multi_task_name.split('/')[1]}-{multi_task_name.split('/')[2][:3]}")  # show when positive
                    ax.set_yticklabels([0.0, 0.1, 0.2, 0.3])  # for positive samples
            else:
                ax.set_yticklabels(["", "", "", ""])

    plt.tight_layout()
    print("in drawing")
    plt.savefig(f"results/{arch}/MAVEN/{request_type}/summary/lineplot_Dec.pdf")
    ###################################################

