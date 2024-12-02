import torch
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
import json
from util import nethook
from util.generate import generate_T5_multi
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.cuda.device_count()
model_name = 'flan-t5-large'
tok = T5Tokenizer.from_pretrained(model_name)  # flan-t5-large
model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")  # flan-t5-large
print(model.config)


if __name__=="__main__":
    # Load Test Set
    SCI_path = './experiments/Causal-TimeBank-main/CTB_zero.json'
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
    model.config._name_or_path = "flan-t5-large"
    nethook.set_requires_grad(False, model)  # True

    pre_update_text = generate_T5_multi(model, tok, generation_prompts, generation_targets, max_out_len=6)  # lan-t5-large
    for text in pre_update_text:
        print(text, end="\n")

    # ABLE
    # Load the parameters of the three tasks:
    TASK_NUM = "3"

    A_task_vector = torch.load("MEET_Dec_ATT_MAVEN_temporal_cla_requests_yes_no_{}.pth".format(TASK_NUM))  # 1, 2
    A_task_w_names = list(A_task_vector.keys())
    A_task_layers = np.array([int(one_name.split('.')[2]) for one_name in A_task_w_names])

    B_task_vector = torch.load("MEET_Dec_ATT_MAVEN_temporal_ext_requests_yes_no_{}.pth".format(TASK_NUM))  # 1, 2
    B_task_w_names = list(B_task_vector.keys())
    B_task_layers = np.array([int(one_name.split('.')[2]) for one_name in B_task_w_names])

    D_task_vector = torch.load("MEET_Dec_ATT_MAVEN_causal_ext_requests_yes_no_{}.pth".format(TASK_NUM))  # 1, 2
    D_task_w_names = list(D_task_vector.keys())
    D_task_layers = np.array([int(one_name.split('.')[2]) for one_name in D_task_w_names])

    # # A - B = C - D, so D = C - (A-B)
    # D_task_layers = C_task_layers - A_task_layers + B_task_layers
    # A - B = C - D, so C = (A-B) + D
    C_task_layers = A_task_layers - B_task_layers + D_task_layers

    new_weights = {}  # Updated weights
    orig_weights = {}  # Storing raw weights
    # D_task_w_names = []
    C_task_w_names = []
    for layer in C_task_layers:
        w_name = 'decoder.block.{}.layer.1.EncDecAttention.o.weight'.format(layer)
        # w_name = 'encoder.block.{}.layer.1.DenseReluDense.wo.weight'.format(layer)
        # D_task_w_names.append(w_name)
        C_task_w_names.append(w_name)
        new_weights[w_name] = nethook.get_parameter(model, w_name)
        orig_weights[w_name] = new_weights[w_name].detach().clone()

    alpha = 1.0
    print("alpha: ", alpha)
    # D_task_vector = {}
    C_task_vector = {}
    for A_name, B_name, C_name, D_name in zip(A_task_w_names, B_task_w_names, C_task_w_names, D_task_w_names):
        # D_task_vector[D_name] = C_task_vector[C_name] - (A_task_vector[A_name] - B_task_vector[B_name]) * alpha
        C_task_vector[C_name] = (A_task_vector[A_name] - B_task_vector[B_name]) * alpha + D_task_vector[D_name]
        new_weights[C_name][...] = orig_weights[C_name] + C_task_vector[C_name]  # 避免循环被叠加赋值
    # finish up editing

    post_update_text = generate_T5_multi(model, tok, generation_prompts, generation_targets, max_out_len=6)  # flan-t5-large
    for text in pre_update_text:
        print(text, end="\n")
