import requests
import time
import soundfile as sf
import numpy as np
import os
import threading
import json

url = "http://0.0.0.0:8080/inference_zero_shot"
num = 5
# 准备要发送的文本和音频文件
path = "../cosyvoice/asset/zero_shot_prompt.wav"
stream = True  # 是否使用流式推理
with open("test_texts.json", "r") as f:
    all_inputs = json.load(f)
res_list = []
os.makedirs("./outs", exist_ok=True)
def get_file(index):
    files = {
        "prompt_wav": ("sample.wav", open(path, "rb"), "audio/wav")
    }
    # inputs = random.choice(all_inputs)
    inputs = all_inputs[0]
    # inputs = all_inputs[2]
    data = {
        "tts_text": inputs,
        "prompt_text": "希望你以后能够做的比我还好呦。",
        "stream": stream
    }
    start_time = time.time()
    chunk_time = time.time()

    response = requests.post(url, files=files, data=data, stream=True)
    sample_rate = 24000
    dtype = "int16"
    first = True
    ttft = 0
    audio_data = bytearray()
    try:
        for chunk in response.iter_content(chunk_size=4096):  # 分批读取
            # if a_time > 0.01:
            if chunk:
                if first:
                    first = False
                    ttft = time.time() - start_time
                # print(f"Received {len(chunk)} bytes, Cost {(time.time() - chunk_time) * 1000} ms")
                chunk_time = time.time()
                audio_data.extend(chunk)
    except Exception as e:
        print(f"Exception: {e}")
        print(f"Error: {response.status_code}, {response.text}")
        return

    cost_time = time.time() - start_time
    speech_len = len(audio_data) / 2 / 24000
    # 将字节数据转换为 NumPy 数组
    audio_np = np.frombuffer(audio_data, dtype=np.int16)

    # 输出服务器响应
    if response.status_code == 200:
        # with open("output_audio.wav", "wb") as f:
        #     f.write(response.content)
        output_wav = f"./outs/output{'_stream' if stream else ''}_{index}.wav"
        sf.write(output_wav, audio_np, samplerate=sample_rate, subtype="PCM_16")
        print(f"{inputs} saved as {output_wav}, time cost: {cost_time:.2f} s, rtf: {cost_time / speech_len}, ttft: {ttft:.2f} s")
    else:
        print("Error:", response.status_code, response.text)

st = time.time()
for index in range(num):
    print("start index", index)
    thread = threading.Thread(target=get_file, args=(index,))
    thread.start()
