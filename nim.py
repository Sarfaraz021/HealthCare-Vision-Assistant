
import os
import requests
import base64
from dotenv import load_dotenv
load_dotenv("var.env")
os.getenv("NVIDIA_API_KEY") == os.environ["NVIDIA_API_KEY"]

invoke_url = "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-11b-vision-instruct/chat/completions"
stream = True

with open("h.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

assert len(image_b64) < 180_000, \
    "To upload larger images, use the assets API (see docs)"


headers = {
    "Authorization": "Bearer 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",
    "Accept": "text/event-stream" if stream else "application/json"
}

payload = {
    "model": 'meta/llama-3.2-11b-vision-instruct',
    "messages": [
        {
            "role": "user",
            "content": f'What is in this image? <img src="data:image/png;base64,{image_b64}" />'
        }
    ],
    "max_tokens": 512,
    "temperature": 1.00,
    "top_p": 1.00,
    "stream": stream
}

response = requests.post(invoke_url, headers=headers, json=payload)

if stream:
    for line in response.iter_lines():
        if line:
            print(line.decode("utf-8"))
else:
    print(response.json())
