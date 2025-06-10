import torch
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
from role_enc.meet_demo import meet_demo_model_editing
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.device_count()

model_name = 'flan-t5-large'
tok = T5Tokenizer.from_pretrained(model_name)  # flan-t5-large
model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")  # flan-t5-large
model.eval()
print(model.config)

TRAIN_NUMBER = 10

# For false negative sample requests, expect the prediction to be changed from YES to NO
# 对于负样本的request，希望预测由YES改为NO
# request_yes_no_path = './experiments/MAVEN_ERE/MAVEN_temporal_cla_requests_yes_no.json'
# request_yes_no_path = './experiments/MAVEN_ERE/MAVEN_temporal_ext_requests_yes_no.json'
# request_yes_no_path = './experiments/MAVEN_ERE/MAVEN_causal_cla_requests_yes_no.json'
# request_yes_no_path = './experiments/MAVEN_ERE/MAVEN_causal_ext_requests_yes_no.json'
# request_yes_no_path = './experiments/MAVEN_ERE/MAVEN_subevent_cla_requests_yes_no.json'
# request_yes_no_path = './experiments/MAVEN_ERE/MAVEN_subevent_ext_requests_yes_no.json'
# request_yes_no_path = './experiments/dataset/SCI_requests_yes_no.json'
# request_yes_no_path = './experiments/dataset/ESL_requests_yes_no.json'
# request_yes_no_path = './experiments/EventStoryLine-master/ESL_729_requests_yes_no.json'
# request_yes_no_path = './experiments/dataset/CTB_requests_yes_no.json'
request_yes_no_path = './experiments/Causal-TimeBank-main/CTB_requests_yes_no.json'
# request_yes_no_path = './experiments/MAVEN_ERE/MAVEN_intra_requests_yes_no.json'
# request_yes_no_path = './experiments/MAVEN_ERE/MAVEN_intra_subevent_requests_yes_no.json'
# request_yes_no_path = './experiments/CausalNewsCorpus-master/CNC_requests_yes_no.json'
# request_yes_no_path = './experiments/ALTLEX/ALTLEX_requests_yes_no.json'
request_yes_no_f = open(request_yes_no_path, 'r')
request_yes_no_data = json.load(request_yes_no_f)
request_yes_no = request_yes_no_data[:TRAIN_NUMBER]  # Training set: pick the top 10 from the false negative samples

# For true positive samples of stat, expect to maintain the YES status
# 对于正样本的stat，希望维持YES的状态
# stat_yes_yes_path = './experiments/MAVEN_ERE/MAVEN_temporal_cla_stat_yes_yes.json'
# stat_yes_yes_path = './experiments/MAVEN_ERE/MAVEN_temporal_ext_stat_yes_yes.json'
# stat_yes_yes_path = './experiments/MAVEN_ERE/MAVEN_causal_cla_stat_yes_yes.json'
# stat_yes_yes_path = './experiments/MAVEN_ERE/MAVEN_causal_ext_stat_yes_yes.json'
# stat_yes_yes_path = './experiments/MAVEN_ERE/MAVEN_subevent_cla_stat_yes_yes.json'
# stat_yes_yes_path = './experiments/MAVEN_ERE/MAVEN_subevent_ext_stat_yes_yes.json'
# stat_yes_yes_path = './experiments/dataset/SCI_stat_yes_yes.json'
stat_yes_yes_path = './experiments/Causal-TimeBank-main/CTB_stat_yes_yes.json'
# stat_yes_yes_path = './experiments/EventStoryLine-master/ESL_729_stat_yes_yes.json'
# stat_yes_yes_path = './experiments/MAVEN_ERE/MAVEN_intra_stat_yes_yes.json'
# stat_yes_yes_path = './experiments/MAVEN_ERE/MAVEN_intra_subevent_stat_yes_yes.json'
# stat_yes_yes_path = './experiments/CausalNewsCorpus-master/CNC_stat_yes_yes.json'
# stat_yes_yes_path = './experiments/ALTLEX/ALTLEX_stat_yes_yes.json'
stat_yes_yes_f = open(stat_yes_yes_path, 'r')
stat_yes_yes_data = json.load(stat_yes_yes_f)
stat_yes_yes = stat_yes_yes_data[:TRAIN_NUMBER]  # Used to calculate kl loss

# Load the test set
# 加载测试集
# SCI_path = './experiments/MAVEN_ERE/MAVEN_temporal_cla_zero.json'
# SCI_path = './experiments/MAVEN_ERE/MAVEN_temporal_cla_zero_pos.json'
# SCI_path = './experiments/MAVEN_ERE/MAVEN_temporal_ext_zero.json'
# SCI_path = './experiments/MAVEN_ERE/MAVEN_temporal_ext_zero_pos.json'
# SCI_path = './experiments/MAVEN_ERE/MAVEN_causal_cla_zero.json'
# SCI_path = './experiments/MAVEN_ERE/MAVEN_causal_cla_zero_pos.json'
# SCI_path = './experiments/MAVEN_ERE/MAVEN_causal_ext_zero.json'
# SCI_path = './experiments/MAVEN_ERE/MAVEN_causal_ext_zero_pos.json'
# SCI_path = './experiments/MAVEN_ERE/MAVEN_subevent_cla_zero.json'
# SCI_path = './experiments/MAVEN_ERE/MAVEN_subevent_cla_zero_pos.json'
# SCI_path = './experiments/MAVEN_ERE/MAVEN_subevent_ext_zero.json'
# SCI_path = './experiments/MAVEN_ERE/MAVEN_subevent_ext_zero_pos.json'
# SCI_path = './experiments/dataset/SCI_zero.json'
# SCI_path = './experiments/dataset/ESL_zero.json'
# SCI_path = './experiments/dataset/CTB_zero.json'
SCI_path = './experiments/Causal-TimeBank-main/CTB_zero.json'
# SCI_path = './experiments/EventStoryLine-master/ESL_729_240_489_zero.json'
# SCI_path = './experiments/EventStoryLine-master/ESL_729_140_589_zero.json'
# SCI_path = './experiments/EventStoryLine-master/ESL_729_130_599_zero.json'
# SCI_path = './experiments/EventStoryLine-master/ESL_729_109_620_zero.json'
# SCI_path = './experiments/EventStoryLine-master/ESL_729_zero.json'
# SCI_path = './experiments/EventStoryLine-master/ESL_zero.json'
# SCI_path = './experiments/EventStoryLine-master/ESL_inter_zero.json'
# SCI_path = './experiments/EventStoryLine-master/ESL_intra_zero.json'
# SCI_path = './experiments/MAVEN_ERE/MAVEN_intra_zero.json'
# SCI_path = './experiments/MAVEN_ERE/MAVEN_intra_subevent_zero.json'
# SCI_path = './experiments/CausalNewsCorpus-master/CNC_zero.json'
# SCI_path = './experiments/ALTLEX/ALTLEX_zero.json'
SCI_f = open(SCI_path, 'r')
SCI_data = json.load(SCI_f)


generation_prompts = []  # 测试集, Test set
generation_targets = []
for data in SCI_data:
    generation_prompts.append(data['prompt'])
    if data['attribute'] == "Yes" or data['attribute'] == "yes":
        y_true = 1
    elif data['attribute'] == "No" or data['attribute'] == "no":
        y_true = 0
    generation_targets.append(y_true)

past_req = []
for req in request_yes_no + stat_yes_yes:
    if 'prompt' in req:
        req_prompt = req['prompt']
    elif 'text' in req:
        req_prompt = req['text']
    if req_prompt not in past_req:
        past_req.append(req_prompt)
        req_idx = generation_prompts.index(req_prompt)
        # Test set: remove labels from training set
        # 测试集：删除训练集的标签
        del generation_targets[req_idx]
        generation_prompts.remove(req_prompt)  # 测试集：删除训练集

ALG_NAME = "ROLE_Enc"

# Execute rewrite
model_new, orig_weights = meet_demo_model_editing(
    model, tok, request_yes_no, stat_yes_yes, generation_prompts, generation_targets, alg_name=ALG_NAME
)
