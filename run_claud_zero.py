import requests
import json
from typing import List
import time
from tqdm import tqdm

BASE_URL = "XXX"
SECRET_KEY = "XXX"


def process_prompts(prompts, batch_size):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SECRET_KEY}"
    }

    data = {
        "model": "claude-3-5-sonnet-20240620",
        # "max_tokens": 30,
        "messages": [
            {"role": "user", "content": "Answer only Yes or No to the following "+str(batch_size)+" prompts:\n"+"\n".join(prompts)}
        ]
    }

    response = requests.post(f"{BASE_URL}/v1/chat/completions", headers=headers, json=data)

    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

    result = response.json()["choices"][0]["message"]["content"]

    # 假设API返回的是一个包含所有回复的列表
    responses = result.split("\n")
    if len(responses) > batch_size:
        responses = responses[-batch_size:]
    elif len(responses) < batch_size:
        print('数量不够')
    return [{"prompt": prompt, "response": response} for prompt, response in zip(prompts, responses)]


def save_results(results, model):
    # filename = f"./results/HiEve/claude-3.5-sonnet-0620/{model}_results.txt"
    # filename = f"./results/HiEve/{model}_results.txt"
    # filename = f"./results/SCI/claude-3.5-sonnet-0620/{model}_results.txt"
    # filename = f"./results/CTB/claude-3.5-sonnet-0620/{model}_results.txt"
    # filename = f"./results/ESL/claude-3.5-sonnet-0620/{model}_results.txt"
    # filename = f"./results/ALTLEX/claude-3.5-sonnet-0620/{model}_results.txt"
    filename = f"./results/CNC/claude-3.5-sonnet-0620/{model}_results.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    # SCI_path = './experiments/data/MAVEN_ERE/MAVEN_intra_subevent_zero.json'
    # SCI_path = './experiments/data/HiEve_Dataset-master/HiEve_zero.json'
    # SCI_path = './experiments/data/SCI/SCI_zero.json'
    # SCI_path = './experiments/data/CTB/CTB_zero.json'
    # SCI_path = './experiments/data/ESL/ESL_729_zero.json'
    # SCI_path = './experiments/data/ALTLEX/ALTLEX_zero.json'
    SCI_path = './experiments/CNC/CNC_zero.json'
    BATCH_SIZE = 64  # 64

    SCI_f = open(SCI_path, 'r')
    SCI_data = json.load(SCI_f)
    all_prompts = []  # 测试集：所有SCI数据集
    for data in SCI_data:
        all_prompts.append(data['prompt'])

    results = []
    # for i in tqdm(range(2496, len(all_prompts), BATCH_SIZE)):
    for i in tqdm(range(0, len(all_prompts), BATCH_SIZE)):
        batch = all_prompts[i:i + BATCH_SIZE]
        batch_results = process_prompts(batch, BATCH_SIZE)
        save_results(batch_results, f"claude-3.5-sonnet-0620_batch_{i // BATCH_SIZE}")
        results.extend(batch_results)
        time.sleep(1.5)  # 添加延迟以避免超过API速率限制

    # save_results(results, "claude-3.5-sonnet-0620")
    print(f"Completed processing for claude-3.5-sonnet-0620")
