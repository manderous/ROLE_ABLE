import numpy, os
from matplotlib import pyplot as plt
from tqdm import tqdm
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams.update({'font.size': 16})
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


def read_knowlege(count=150, multi_task_name=None, kind=None, arch="gpt2-xl"):
    print('Encoder ongoing task:', multi_task_name, ", module: ", kind)
    dirname = f"results/{arch}/{multi_task_name}/cases/"
    kindcode = "" if not kind else f"_{kind}"
    (avg_fe, avg_ee, avg_le, avg_cc, avg_fa, avg_ea, avg_la, avg_hs, avg_ls, avg_fs, avg_fle, avg_fla,) = [Avg() for _ in range(12)]
    for i in tqdm(range(count)):
        try:
            data = numpy.load(f"{dirname}/knowledge_{i}{kindcode}.npz")
        except:
            continue
        # Only consider cases where the model begins with the correct prediction
        if "correct_prediction" in data and not data["correct_prediction"]:
            continue
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

        avg_fe.add(scores[9])  # I
        avg_ee.add(scores[10])  # s
        avg_le.add(scores[11])  # there
        avg_cc.add(scores[12])  # " " new variable
        avg_fa.add(scores[13])  # a

        # avg_fe.add_all(scores[9:11])  # Is
        # avg_ee.add(scores[11])  # there
        # avg_le.add(scores[12])  #
        # avg_fa.add(scores[13])  # a

        if "causal" in multi_task_name.split('/')[1]:
            avg_ea.add(scores[14])  # causal
            avg_la.add(scores[15])  # relation
        elif "temporal" in multi_task_name.split('/')[1]:
            avg_ea.add_all(scores[14:16])  # temporal
            avg_la.add(scores[16])  # relation
        elif "subevent" in multi_task_name.split('/')[1]:
            avg_ea.add_all(scores[14:17])  # subevent
            avg_la.add(scores[17])  # relation

    result = numpy.stack([avg_fe.avg(), avg_ee.avg(), avg_le.avg(), avg_cc.avg(), avg_fa.avg(), avg_ea.avg(), avg_la.avg(),])
    result_std = numpy.stack([avg_fe.std(), avg_ee.std(), avg_le.std(), avg_cc.std(), avg_fa.std(), avg_ea.std(), avg_la.std(),])
    return dict(low_score=avg_ls.avg(), result=result, result_std=result_std, size=avg_fe.size())


def read_knowlege_decoder(count=150, multi_task_name=None, kind=None, arch="gpt2-xl"):
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
    multi_task_names = ["MAVEN/temporal/classification", "MAVEN/temporal/extraction",  # True positive samples
                        "MAVEN/causal/classification", "MAVEN/causal/extraction",  # True positive samples
                        "MAVEN/subevent/classification", "MAVEN/subevent/extraction",  # True positive samples
                        "MAVEN/temporalNo/classification", "MAVEN/temporalNo/extraction",  # True negative samples
                        "MAVEN/causalNo/classification", "MAVEN/causalNo/extraction",  # True negative samples
                        "MAVEN/subeventNo/classification", "MAVEN/subeventNo/extraction"  # True negative samples
                        ]
    # Remember to modify it before each run !
    # choise:[results_No, results_W_Yes, results_NoW_No, results]
    request_type = "results_and_resultsNo"
    task_num = 6

    ############  Consider layer.1.DenseReluDense of T5-Encoder  ############
    kind_encoder = "layer.1.DenseReluDense"  # Modules for studying encoders: layer.1.DenseReluDense
    kind_decoder = "layer.1.EncDecAttention"  # Modules for studying decoders: layer.1.EncDecAttention
    multi_task_results_encoder, multi_task_results_decoder = [], []  # Initialize List
    for multi_task_name in multi_task_names:
        d_encoder = read_knowlege(the_count, multi_task_name, kind_encoder, arch)
        d_decoder = read_knowlege_decoder(the_count, multi_task_name, kind_decoder, arch)
        if "No" in multi_task_name.split('/')[1]:
            # For negative samples, reverse the AIE calculation,
            # because negative samples should have a higher probability of predicting “No” with added noise.
            task_value_encoder = d_encoder["low_score"] - d_encoder["result"]  # Encoder
            task_value_decoder = d_decoder["low_score"] - d_decoder["result"][0, :]  # Decoder
        else:
            task_value_encoder = d_encoder["result"] - d_encoder["low_score"]  # Encoder
            task_value_decoder = d_decoder["result"][0, :] - d_decoder["low_score"]  # Decoder
        multi_task_results_encoder.append(task_value_encoder)  # Encoder
        multi_task_results_decoder.append(task_value_decoder)  # Decoder

    # Visualize the difference between all tokens under Extract and Classify tasks (qualitative analysis)
    # token_list = ['I', 's', 'there', ' ', 'a', 'causal/temporal/sub-event', 'relation']
    token_ids = [1, 2, 5, 6]
    token_num = len(token_ids)
    token_list = ['Encoder-"Is"', 'Encoder-"there"', 'Encoder-"causal/temporal/subevent"', 'Encoder-"relation"', 'Decoder-"</s>"']
    task_labels = ['Positive samples: Temporal relation', 'Positive samples: Causal relation', 'Positive samples: Sub-event relation',
                   'Negative samples: Temporal relation', 'Negative samples: Causal relation', 'Negative samples: Sub-event relation']
    title_labels = ['Positive samples \n Temporal relation \n Classification-Extraction',
                    'Positive samples \n Causal relation \n Classification-Extraction',
                    'Positive samples \n Subevent relation \n Classification-Extraction',
                   'Negative samples \n Temporal relation \n Classification-Extraction',
                    'Negative samples \n Causal relation \n Classification-Extraction',
                    'Negative samples \n Subevent relation \n Classification-Extraction']
    fig, axes = plt.subplots(task_num, token_num+1, figsize=(4*(token_num+1), 14), sharey=False, sharex=True, dpi=200)
    for task_id in range(task_num):

        # determines the shape of a line
        if task_id % 3 == 0:  # Temporal
            line_char = "-"
        elif task_id % 3 == 1:  # Causal
            line_char = "--"
        else:  # 子事件
            line_char = "-."

        for t_id in range(token_num):
            yy = multi_task_results_encoder[2 * task_id] - multi_task_results_encoder[2 * task_id + 1]  # Classify-Extract
            yy_token = yy[token_ids[t_id], :]
            x = list(range(len(yy_token)))
            ax = axes[task_id, t_id]
            # 决定颜色
            if task_id < 3:  # Positive samples
                color_char = "forestgreen"
            else:  # Negative samples
                color_char = "darkorchid"
            ax.plot(x, yy_token, linestyle=line_char, linewidth =2.0, color=color_char, label=task_labels[task_id]+'-Encoder')
            if task_id == 0:
                ax.set_title(f"{token_list[t_id]}")
            if t_id == 0:
                ax.set_ylabel(title_labels[task_id], rotation=0, labelpad=100, fontsize=18)

        yy = multi_task_results_decoder[2 * task_id] - multi_task_results_decoder[2 * task_id + 1]  # Classify-Extract
        x = list(range(len(yy)))
        ax = axes[task_id, 4]
        # decide on a color
        if task_id < 3:  # Positive samples
            color_char = "darkorange"
        else:  # Negative samples
            color_char = "deeppink"
        ax.plot(x, yy, linestyle=line_char, linewidth =2.0, color=color_char, label=task_labels[task_id]+'-Decoder')
        if task_id == 0:
            ax.set_title(f"{token_list[4]}")

    savepdf_Enc = f"results/{arch}/MAVEN/{request_type}/differences/lineplot-differences_Enc_Dec_paper.pdf"
    os.makedirs(os.path.dirname(savepdf_Enc), exist_ok=True)
    print("in drawing")
    plt.tight_layout()  # Automatically adjusts subgraph spacing
    plt.savefig(savepdf_Enc, bbox_inches="tight")
