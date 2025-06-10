import numpy, os
from matplotlib import pyplot as plt
from tqdm import tqdm
from matplotlib.font_manager import FontProperties


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams.update({'font.size': 18})
zhfont = FontProperties(fname='simsun.ttc')  # 替换为你的宋体字体文件路径
arch = "ns3_r0__share_project_whm_jingyao_rome-main_flan-t5-large"  # tjy: for flan-T5-xl
archname = "Flan-T5-large"  # tjy: for flan-t5-large


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
    print('Encoder正在进行的任务：', multi_task_name, "，模块：", kind)
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
        avg_cc.add(scores[12])  # " " tjy新增变量
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
    print('Decoder正在进行的任务：', multi_task_name, "，模块：", kind)
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
    the_count = 500  # tjy: flan-t5-large: TBD: 2020, CNC: 1603, CTB:318, MAVEN:500
    high_score = None  # Scale all plots according to the y axis of the first plot
    multi_task_names = ["MAVEN/temporal/classification", "MAVEN/temporal/extraction",  # 正确的正样本
                        "MAVEN/causal/classification", "MAVEN/causal/extraction",  # 正确的正样本
                        "MAVEN/subevent/classification", "MAVEN/subevent/extraction",  # 正确的正样本
                        "MAVEN/temporalNo/classification", "MAVEN/temporalNo/extraction",  # 正确的负样本
                        "MAVEN/causalNo/classification", "MAVEN/causalNo/extraction",  # 正确的负样本
                        "MAVEN/subeventNo/classification", "MAVEN/subeventNo/extraction"  # 正确的负样本
                        ]
    request_type = "results_and_resultsNo"  # tjy: 记得每次运行前修改 ！！！   # results_No, results_W_Yes, results_NoW_No, results
    task_num = 6

    ############  考虑T5-Encoder的layer.1.DenseReluDense  ############
    kind_encoder = "layer.1.DenseReluDense"  # 编码器研究的模块: layer.1.DenseReluDense
    kind_decoder = "layer.1.EncDecAttention"  # 解码器研究的模块:layer.1.EncDecAttention
    multi_task_results_encoder, multi_task_results_decoder = [], []  # 初始化列表
    for multi_task_name in multi_task_names:
        d_encoder = read_knowlege(the_count, multi_task_name, kind_encoder, arch)
        d_decoder = read_knowlege_decoder(the_count, multi_task_name, kind_decoder, arch)
        if "No" in multi_task_name.split('/')[1]:  # 若为负样本
            # 对于负样本，要反过来计算AIE，因为负样本在加噪声的情况下，选择No的概率应该会更大
            task_value_encoder = d_encoder["low_score"] - d_encoder["result"]  # 编码器
            task_value_decoder = d_decoder["low_score"] - d_decoder["result"][0, :]  # 解码器
        else:  # 若为正样本
            # 对于正样本，这样计算AIE
            task_value_encoder = d_encoder["result"] - d_encoder["low_score"]  # 编码器
            task_value_decoder = d_decoder["result"][0, :] - d_decoder["low_score"]  # 解码器
        multi_task_results_encoder.append(task_value_encoder)  # 编码器
        multi_task_results_decoder.append(task_value_decoder)  # 解码器

    # 可视化所有token在Extract和Classify任务下的差值（定性分析）
    # token_list = ['I', 's', 'there', ' ', 'a', 'causal/temporal/sub-event', 'relation']
    token_ids = [1, 2, 5, 6]
    token_num = len(token_ids)
    token_list = ['编码器-"Is"', '编码器-"there"', '编码器-"causal/temporal/subevent"', '编码器-"relation"', '解码器-"</s>"']
    task_labels = ['Positive samples: Temporal relation', 'Positive samples: Causal relation', 'Positive samples: Sub-event relation',
                   'Negative samples: Temporal relation', 'Negative samples: Causal relation', 'Negative samples: Sub-event relation']
    title_labels = ['正样本 \n 时序(分类-抽取)',
                    '正样本 \n 因果(分类-抽取)',
                    '正样本 \n 子事件(分类-抽取)',
                   '负样本 \n 时序(分类-抽取)',
                    '负样本 \n 因果(分类-抽取)',
                    '负样本 \n 子事件(分类-抽取)']
    fig, axes = plt.subplots(task_num, token_num+1, figsize=(4*(token_num+1), 14), sharey=False, sharex=True, dpi=200)
    legend_lines = []  # 用于绘制图例
    legend_labels = []  # 用于绘制图例
    for task_id in range(task_num):

        # 决定线型
        if task_id % 3 == 0:  # 时序
            line_char = "-"
        elif task_id % 3 == 1:  # 因果
            line_char = "--"
        else:  # 子事件
            line_char = "-."

        for t_id in range(token_num):
            yy = multi_task_results_encoder[2 * task_id] - multi_task_results_encoder[2 * task_id + 1]  # Classify-Extract
            yy_token = yy[token_ids[t_id], :]
            x = list(range(len(yy_token)))
            ax = axes[task_id, t_id]
            # 决定颜色
            if task_id < 3:  # 正样本
                color_char = "forestgreen"
            else:  # 负样本
                color_char = "darkorchid"
            ax.plot(x, yy_token, linestyle=line_char, linewidth =2.0, color=color_char, label=task_labels[task_id]+'-Encoder')
            if task_id == 0:
                ax.set_title(f"{token_list[t_id]}", fontproperties=zhfont, size=18)
            if t_id == 0:
                ax.set_ylabel(title_labels[task_id], rotation=0, labelpad=100, fontsize=18, fontproperties=zhfont)

        # axLine, axLabel = ax.get_legend_handles_labels()  # 用于绘制图例
        # legend_lines.extend(axLine)  # 用于绘制图例
        # legend_labels.extend(axLabel)  # 用于绘制图例

        yy = multi_task_results_decoder[2 * task_id] - multi_task_results_decoder[2 * task_id + 1]  # Classify-Extract
        x = list(range(len(yy)))
        ax = axes[task_id, 4]
        # 决定颜色
        if task_id < 3:  # 正样本
            color_char = "darkorange"
        else:  # 负样本
            color_char = "deeppink"
        ax.plot(x, yy, linestyle=line_char, linewidth =2.0, color=color_char, label=task_labels[task_id]+'-Decoder')
        if task_id == 0:
            ax.set_title(f"{token_list[4]}", fontproperties=zhfont, size=18)
        # axLine, axLabel = ax.get_legend_handles_labels()  # 用于绘制图例
        # legend_lines.extend(axLine)  # 用于绘制图例
        # legend_labels.extend(axLabel)  # 用于绘制图例

    savepdf_Enc = f"results/{arch}/MAVEN/{request_type}/differences/lineplot-differences_Enc_Dec_paper.pdf"
    os.makedirs(os.path.dirname(savepdf_Enc), exist_ok=True)
    print("画图中")
    # fig.legend(legend_lines, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)  # 绘制图例
    plt.tight_layout()  # 自动调节子图间距
    plt.savefig(savepdf_Enc, bbox_inches="tight")
