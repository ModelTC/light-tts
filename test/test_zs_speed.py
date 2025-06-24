######## lora 语音速度测试 ###############
from datetime import datetime
import uuid
import json
import time
import requests
import json
from concurrent.futures import ProcessPoolExecutor
import soundfile as sf
import numpy as np
import numpy as np
import random
from tqdm import tqdm
import os

# config
server_url = "http://localhost:8080/inference_zero_shot"
num_workers = [1, 2, 4, 8]
num_test = 50
path = "../cosyvoice/asset/zero_shot_prompt.wav"
sample_rate = 24000
# get all inputs
with open("test_texts.json", "r") as f:
    all_inputs = json.load(f)

failed = 0
os.makedirs("./outs", exist_ok=True)

def get_file(index):
    url = f"{server_url}/"
    inputs = all_inputs[index % len(all_inputs)]
    files = {
        "prompt_wav": ("sample.wav", open(path, "rb"), "audio/wav")
    }
    data = {
        "tts_text": inputs,
        "prompt_text": "希望你以后能够做的比我还好呦。",
    }
    global failed

    start = time.time()
    response = requests.post(url, files=files, data=data, stream=True)
    cost_time = time.time() - start
    
    try:
        if response.status_code == 200:
            audio_data = bytearray()
            for chunk in response.iter_content(chunk_size=4096):  # 分批读取
                if chunk:
                    audio_data.extend(chunk)
        else:
            print(f"{index} {inputs} \n error:{response.text}")
            failed += 1
    except Exception as e:
        print(f'An exception occurred\ntext: {inputs}\n{e}\n{response.text}')
        failed += 1
    finally:
        return cost_time

results = []
for num_worker in num_workers:
    print(f"Concurrency {num_worker} test starts")

    last_failed = failed

    start_time = time.time()
    with ProcessPoolExecutor(max_workers=num_worker) as executor:
        all_cost_times = list(
            tqdm(executor.map(get_file, range(num_test)), total=num_test, desc="running tests")
        )
    total_time = time.time() - start_time

    # 假设是你的数据数组
    data = np.array(all_cost_times)
    result = {
        "num_workers": num_worker
    }
    # 计算分位数
    if failed - last_failed > 0:
        print(f"Failed {failed - last_failed}")
    for percentile in [50, 90, 99]:
        percentile_data = np.percentile(data, percentile)
        result[f"{percentile}%"] = round(percentile_data, 2)
    result["total_cost_time"] = round(total_time, 2)
    result["qps"] = round(num_test / total_time, 2)
    results.append(result)

    for k, v in result.items():
        print(f"{k}: {v}")

    print(f"Concurrency {num_worker} test ends")
    print("\n" + "=" * 80 + "\n")

if failed == 0:
    print("Test passed!")
else:
    print(f"Total failed {failed}")

result_path = f"./summary_triton_flashdecoding_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.md"

heads = results[0].keys()
with open(result_path, "w") as md_file:
    md_file.write("|")
    for head in heads:
        md_file.write(head + "|")
    md_file.write("\r\n")
    md_file.write("|")
    for _ in range(len(heads)):
        md_file.write("------|")
    md_file.write("\r\n")
    for result in results:
        md_file.write("|")
        for head in heads:
            md_file.write(str(result[head]) + "|")
        md_file.write("\r\n")