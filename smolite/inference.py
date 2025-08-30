import json
import numpy as np
import tensorflow as tf
import sentencepiece as spm
import requests
import pyarrow.parquet as pq
from smolite.model import Smolite
from smolite.download_file import download_file
from generate-tokenizer import generate_max, ids_to_text, text_to_ids
# =======================
# 0) 파일 다운로드 함수
# =======================
def download_file(url, save_path):
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(save_path, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    print(f"✅ {save_path} 저장됨")


download_file(
    "https://huggingface.co/datasets/Yuchan5386/Smolwrite-dataset/resolve/main/unigram.model?download=true",
    "ko_unigram.model"
)

max_len = 100
model = Smolite(vocab_size, seq_len=max_len, d_model=256, n_layers=12, d_ff=1024, num_heads=8, dropout_rate=0.1)
dummy_input = np.zeros((1, max_len), dtype=np.int32)
model(dummy_input)

model.load_weights('model1.weights.h5')
print("모델 가중치 로드 완료!")


last_len = 0
text = ""  # 생성된 전체 텍스트를 저장

# 종료 조건: 사용자가 'quit' 입력하거나 모델이 <end> 토큰 생성 시
while True:
    # 사용자 입력 받기
    user_input = input("\n사용자: ")
    if user_input.lower() in ["quit", "exit"]:
        print("대화를 종료합니다.")
        break

    # 모델 입력에 사용자 메시지 추가
    prompt = user_input
    for new_text in generate_max(model, prompt):
        # 새로 생긴 부분만 출력
        print(new_text[last_len:], end="", flush=True)
        text = new_text
        last_len = len(text)

    # 출력 후 마지막 길이 초기화
    last_len = 0
    print("\n")
