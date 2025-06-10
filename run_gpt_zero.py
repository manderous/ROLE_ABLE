import openai
import json
from time import sleep
from tqdm import tqdm
import requests

# 设置自定义的基础地址和 API 密钥
base_url = "XXX"
api_key = "XXX"

# 定义请求头
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}


# 定义要使用的模型列表
# models = ["text-davinci-002", "text-davinci-003", "gpt-3.5-turbo", "gpt-4"]
models = ["gpt-4"]  # davinci-002, gpt-3.5-turbo-0125, gpt-4

def call_openai_api_batch(prompts, model):
    try:
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt} for prompt in prompts],
            "max_tokens": 10,
            "n": len(prompts)
        }
        response = requests.post(url=f"{base_url}/chat/completions",
                                 headers=headers,
                                 json=data)
        return [choice['message']['content'] for choice in response.json()['choices']]
    except Exception as e:
        print(f"Error calling API for model {model}: {str(e)}")
        return [None] * len(prompts)


def process_prompts_batch(prompts, model):
    results = []
    responses = call_openai_api_batch(prompts, model)
    for prompt, response in zip(prompts, responses):
        if response:
            results.append({"prompt": prompt, "response": response})
    return results


def save_results(results, model):
    filename = f"./results/SCI/gpt-4/{model}_results.txt"
    # filename = f"./results/SCI/gpt-3.5-turbo-0125/{model}_results.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {filename}")


def remove_sensitive_phrases(text):
    phrases = ["Moharebeh (waging war against God)", "the Zionist regime", "Mossad"]
    for phrase in phrases:
        text = text.replace(phrase, "")
    return text


def main():
    # SCI_path = './experiments/data/MAVEN_ERE/MAVEN_intra_subevent_zero.json'
    # SCI_path = './experiments/data/HiEve_Dataset-master/HiEve_zero.json'
    # SCI_path = './experiments/data/SCI/SCI_zero.json'
    # SCI_path = './experiments/data/CTB/CTB_zero.json'
    # SCI_path = './experiments/data/ESL/ESL_729_zero.json'
    # SCI_path = './experiments/data/ALTLEX/ALTLEX_zero.json'
    SCI_path = './experiments/SCI/SCI_zero.json'
    batch_size = 64

    SCI_f = open(SCI_path, 'r')
    SCI_data = json.load(SCI_f)
    all_prompts = []  # 测试集：所有SCI数据集
    for data in SCI_data:
        # all_prompts.append(data['prompt'].split('question. ')[1] + " Only answer yes/no.")  # for davinci-002
        all_prompts.append(data['prompt'])

    for model in models:
        print(f"Processing model: {model}")
        all_results = []
        # for i in tqdm(range(27328, 27392, batch_size), desc=f"Batches for {model}"):
        for i in tqdm(range(0, len(all_prompts), batch_size), desc=f"Batches for {model}"):
            batch = all_prompts[i:i+batch_size]
            # batch = [remove_sensitive_phrases(one_b) for one_b in batch]

            results = process_prompts_batch(batch, model)
            save_results(results, f"{model}_batch_{i // batch_size}")
            all_results.extend(results)

            sleep(1)  # 为了避免超过API速率限制
        save_results(all_results, model)
        print(f"Completed processing for {model}")


if __name__ == "__main__":
    main()
