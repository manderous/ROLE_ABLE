import torch
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
from role_dec.meet_att_demo import meet_att_demo_model_editing
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
request_yes_no_path = './experiments/dataset/CTB_requests_yes_no.json'
request_yes_no_f = open(request_yes_no_path, 'r')
request_yes_no_data = json.load(request_yes_no_f)
request_yes_no = request_yes_no_data[:TRAIN_NUMBER]  # Training set: pick the top 10 from the false negative samples

# For true positive samples of stat, expect to maintain the YES status
stat_yes_yes_path = './experiments/dataset/CTB_stat_yes_yes.json'
stat_yes_yes_f = open(stat_yes_yes_path, 'r')
stat_yes_yes_data = json.load(stat_yes_yes_f)
stat_yes_yes = stat_yes_yes_data[:TRAIN_NUMBER]  # Used to calculate kl loss

# Load the test set
SCI_path = './experiments/Causal-TimeBank-main/CTB_zero.json'
SCI_f = open(SCI_path, 'r')
SCI_data = json.load(SCI_f)


generation_prompts = []  # Test set
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
        del generation_targets[req_idx]  # Test set: remove labels from training set
        generation_prompts.remove(req_prompt)

ALG_NAME = "ROLE_Dec"

# Execute rewrite
model_new, orig_weights = meet_att_demo_model_editing(
    model, tok, request_yes_no, stat_yes_yes, generation_prompts, generation_targets, alg_name=ALG_NAME
)
