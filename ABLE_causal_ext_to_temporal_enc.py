# Tjy write
import torch
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration  # tjy
import json
from util import nethook
from util.generate import generate_T5_multi
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.device_count()

model_name = '/home/jingyao2/PycharmProjects/memit-main/flan-t5-large'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = T5Tokenizer.from_pretrained(model_name)  # 使用flan-t5-large
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)  # tjy 使用flan-t5-large
print(model.config)


if __name__=="__main__":
    # 加载测试集
    # SCI_path = './experiments/dataset/SCI_zero.json'
    SCI_path = './experiments/dataset/ESL_zero.json'
    # SCI_path = './experiments/dataset/CTB_zero.json'
    # SCI_path = './experiments/EventStoryLine-master/ESL_729_130_599_zero.json'
    # SCI_path = './experiments/EventStoryLine-master/ESL_729_zero.json'
    # SCI_path = './experiments/EventStoryLine-master/ESL_zero.json'
    # SCI_path = './experiments/EventStoryLine-master/ESL_inter_zero.json'
    # SCI_path = './experiments/EventStoryLine-master/ESL_intra_zero.json'
    # SCI_path = './experiments/EventStoryLine-master/ESL_zero_top20.json'
    # SCI_path = './experiments/EventStoryLine-master/ESL_inter_zero_top20.json'
    # SCI_path = './experiments/EventStoryLine-master/ESL_intra_zero_top20.json'
    # SCI_path = './experiments/Causal-TimeBank-main/CTB_zero.json'
    # SCI_path = './experiments/MAVEN_ERE/MAVEN_intra_zero.json'
    # SCI_path = './experiments/MAVEN_ERE/MAVEN_intra_more_neg_zero.json'
    # SCI_path = './experiments/CausalNewsCorpus-master/CNC_zero.json'
    # SCI_path = './experiments/ALTLEX/ALTLEX_zero.json'
    # SCI_path = './experiments/MAVEN_ERE/MAVEN_intra_subevent_zero.json'
    # SCI_path = './experiments/HiEve_Dataset-master/HiEve_zero.json'

    SCI_f = open(SCI_path, 'r')
    SCI_data = json.load(SCI_f)

    generation_prompts = []  # 测试集：所有SCI数据集
    generation_targets = []
    for data in SCI_data:
        generation_prompts.append(data['prompt'])
        if data['attribute'] == "Yes" or data['attribute'] == "yes":
            y_true = 1
        elif data['attribute'] == "No" or data['attribute'] == "no":
            y_true = 0
        generation_targets.append(y_true)

    # Execute rewrite
    model.config._name_or_path = "flan-t5-large"  # tjy
    nethook.set_requires_grad(False, model)  # True

    # print_loud("Generating pre-update text")
    # pre_update_text = generate_T5_multi(model, tok, generation_prompts, generation_targets, max_out_len=6)  # tjy: flan-t5-large
    # for text in pre_update_text:  # tjy: 增加换行符
    #     print(text, end="\n")

    # tjy: 类比编辑
    # 加载三个任务的参数：

    TASK_NUM = "1"

    # start = time.time()
    # 时序任务
    # MEET_Enc_MLP_MAVEN_temporal_cla_requests_yes_no_1
    # MEET_Dec_ATT_MAVEN_temporal_cla_requests_yes_no_1
    A_task_vector = torch.load("MEET_Enc_MLP_MAVEN_temporal_cla_requests_yes_no_{}.pth".format(TASK_NUM))  # 1, 2
    A_task_w_names = list(A_task_vector.keys())
    A_task_layers = np.array([int(one_name.split('.')[2]) for one_name in A_task_w_names])
    # MEET_Enc_MLP_MAVEN_temporal_ext_requests_yes_no_1
    # MEET_Dec_ATT_MAVEN_temporal_ext_requests_yes_no_1
    B_task_vector = torch.load("MEET_Enc_MLP_MAVEN_temporal_ext_requests_yes_no_{}.pth".format(TASK_NUM))  # 1, 2
    B_task_w_names = list(B_task_vector.keys())
    B_task_layers = np.array([int(one_name.split('.')[2]) for one_name in B_task_w_names])

    # # 子事件任务
    # # MEET_Enc_MLP_MAVEN_subevent_cla_requests_yes_no_1
    # # MEET_Dec_ATT_MAVEN_subevent_cla_requests_yes_no_1
    # A_task_vector = torch.load("MEET_Dec_ATT_MAVEN_subevent_cla_requests_yes_no_{}.pth".format(TASK_NUM))  # 1, 2
    # A_task_w_names = list(A_task_vector.keys())
    # A_task_layers = np.array([int(one_name.split('.')[2]) for one_name in A_task_w_names])
    # # MEET_Enc_MLP_MAVEN_subevent_ext_requests_yes_no_1
    # # MEET_Dec_ATT_MAVEN_subevent_ext_requests_yes_no_1
    # B_task_vector = torch.load("MEET_Dec_ATT_MAVEN_subevent_ext_requests_yes_no_{}.pth".format(TASK_NUM))  # 1, 2
    # B_task_w_names = list(B_task_vector.keys())
    # B_task_layers = np.array([int(one_name.split('.')[2]) for one_name in B_task_w_names])

    # # 因果任务
    # # MEET_Enc_MLP_MAVEN_subevent_cla_requests_yes_no_1
    # # MEET_Dec_ATT_MAVEN_subevent_cla_requests_yes_no_1
    # A_task_vector = torch.load("MEET_Enc_MLP_MAVEN_causal_cla_requests_yes_no_{}.pth".format(TASK_NUM))  # 1, 2
    # A_task_w_names = list(A_task_vector.keys())
    # A_task_layers = np.array([int(one_name.split('.')[2]) for one_name in A_task_w_names])
    # # MEET_Enc_MLP_MAVEN_subevent_ext_requests_yes_no_1
    # # MEET_Dec_ATT_MAVEN_subevent_ext_requests_yes_no_1
    # B_task_vector = torch.load("MEET_Enc_MLP_MAVEN_causal_ext_requests_yes_no_{}.pth".format(TASK_NUM))  # 1, 2
    # B_task_w_names = list(B_task_vector.keys())
    # B_task_layers = np.array([int(one_name.split('.')[2]) for one_name in B_task_w_names])

    ###########################################################################################
    # MEET_Enc_MLP_MAVEN_causal_cla_requests_yes_no_1.pth
    # MEET_Dec_ATT_MAVEN_causal_cla_requests_yes_no_1.pth
    # MEET_Enc_MLP_MAVEN_subevent_cla_requests_yes_no_1.pth
    # MEET_Dec_ATT_MAVEN_subevent_cla_requests_yes_no_1.pth
    C_task_vector = torch.load("MEET_Enc_MLP_MAVEN_causal_cla_requests_yes_no_{}.pth".format(TASK_NUM))  # 1, 2
    C_task_w_names = list(C_task_vector.keys())
    C_task_layers = np.array([int(one_name.split('.')[2]) for one_name in C_task_w_names])

    # A - B = C - D, so D = C - (A-B)
    D_task_layers = C_task_layers - A_task_layers + B_task_layers

    new_weights = {}  # 更新的weights
    orig_weights = {}  # 存储原始的weights
    D_task_w_names = []
    total_params = 0
    for layer in D_task_layers:
        # w_name = 'decoder.block.{}.layer.1.EncDecAttention.o.weight'.format(layer)
        w_name = 'encoder.block.{}.layer.1.DenseReluDense.wo.weight'.format(layer)
        D_task_w_names.append(w_name)
        new_weights[w_name] = nethook.get_parameter(model, w_name)
        orig_weights[w_name] = new_weights[w_name].detach().clone()
        total_params += new_weights[w_name].numel()

    param_count_m = total_params / 1e6
    print(f"Total model parameters: {param_count_m:.2f}M")

    alpha = 1.8
    print("alpha: ", alpha)
    D_task_vector = {}
    for A_name, B_name, C_name, D_name in zip(A_task_w_names, B_task_w_names, C_task_w_names, D_task_w_names):
        D_task_vector[D_name] = C_task_vector[C_name] - (A_task_vector[A_name] - B_task_vector[B_name]) * alpha
        new_weights[D_name][...] = orig_weights[D_name] + D_task_vector[D_name]  # 避免循环被叠加赋值
        # analogy_total_params += (A_task_vector[A_name].numel() + B_task_vector[B_name].numel() +
        #                          C_task_vector[C_name].numel() + D_task_vector[D_name].numel())
    # finish up editing

    # end = time.time()
    # print(end - start)
    # analogy_param_count_m = analogy_total_params / 1e6
    # print(f"Analogy total model parameters: {analogy_param_count_m:.2f}M")

    # print_loud("Generating post-update text")
    post_update_text = generate_T5_multi(model, tok, generation_prompts, generation_targets, max_out_len=6)  # tjy: flan-t5-large
    # for text in pre_update_text:  # tjy: 增加换行符
    #     print(text, end="\n")

