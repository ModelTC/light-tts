import requests
import time
import json
import soundfile as sf
import io
import numpy as np
import os
import threading
import random
import websockets
import base64
import asyncio

url = "ws://localhost:8080/inference_zero_shot_bistream"
num = 1
# 准备要发送的文本和音频文件
path = "../cosyvoice/asset/zero_shot_prompt.wav"
res_list = []
os.makedirs("./outs", exist_ok=True)
def text_generator(index):
    for text in [
        '收到好友从远方寄来的生日礼物，',
        '那份意外的惊喜与深深的祝福',
        '让我心中充满了甜蜜的快乐，',
        '笑容如花儿般绽放。',
    ]:
        yield text

async def send_texts(websocket: websockets.WebSocketClientProtocol, index):
    for text in text_generator(index):
        await websocket.send(json.dumps({"tts_text": text}))
        print(f"Index {index}, Sent text: {text}")
        await asyncio.sleep(0.1)  # 模拟发送文本的时间
    await websocket.send(json.dumps({"finish": True}))  # 发
    print(f"Index {index}, Text sending completed.")

async def receive_audio(websocket: websockets.WebSocketClientProtocol, index):
    audio_data = bytearray()
    sample_rate = 24000
    output_wav = f"./outs/output_bistream_{index}.wav"
    try:
        while True:
            response_audio = await websocket.recv()
            if isinstance(response_audio, bytes):
                audio_data.extend(response_audio)
                print(f"Index {index}, Received {len(response_audio)} bytes.")
            else:
                print(f"Index {index}, Received {response_audio}")
    except websockets.exceptions.ConnectionClosed:
        print(f"Index {index}, Connection closed.")
    finally:
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        sf.write(output_wav, audio_np, samplerate=sample_rate, subtype="PCM_16")
        print(f"Index {index}, Audio saved as {output_wav}.")
    
async def main(index):
    async with websockets.connect(url) as websocket:
        # Step 1: 发送初始化参数
        init_params = {
            "prompt_text": "希望你以后能够做的比我还好呦。",
        }
        await websocket.send(json.dumps(init_params))
        print("Sent initialization data.")

        # Step 2: 发送 `prompt_wav` 文件
        with open(path, "rb") as f:
            await websocket.send(f.read())
        print(f"Sent prompt_wav file: {path}")

        # Step 3: 并行执行 **发送任务** 和 **接收任务**
        send_task = asyncio.create_task(send_texts(websocket, index))
        receive_task = asyncio.create_task(receive_audio(websocket, index))

        # 等待两个任务完成
        await asyncio.gather(send_task, receive_task)

async def run_multiple():
    tasks = [main(index) for index in range(num)]
    await asyncio.gather(*tasks)

# 运行主程序
asyncio.run(run_multiple())