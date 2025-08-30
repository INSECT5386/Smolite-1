import json
import numpy as np
import tensorflow as tf
import sentencepiece as spm
import requests
import pyarrow.parquet as pq
from smolite.model import Smolite
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

def generate_text_streaming_fixed(model, prompt, max_len=100, max_gen=100, p=0.9, temperature=0.5, min_len=10, repetition_penalty=1.1):
    model_input = text_to_ids(f"<start> {prompt} <sep>")
    model_input = model_input[:max_len]
    generated = list(model_input)
    vocab_size = sp.get_piece_size()

    for step in range(max_gen):
        if len(generated) > max_len:
            input_seq = generated[-max_len:]
        else:
            input_seq = generated

        input_padded = np.pad(input_seq, (0, max_len - len(input_seq)), constant_values=pad_id)
        input_tensor = tf.convert_to_tensor([input_padded])

        logits = model(input_tensor, training=False)
        next_token_logits = logits[0, len(input_seq) - 1].numpy()

        # 반복 패널티
        unique, counts = np.unique(generated, return_counts=True)
        for u, c in zip(unique, counts):
            next_token_logits[u] /= repetition_penalty ** c

        # pad / end 제한
        if end_id < vocab_size:
            next_token_logits[end_id] -= 5.0
        if pad_id < vocab_size:
            next_token_logits[pad_id] -= 10.0

        # top-p 샘플링
        probs = tf.nn.softmax(next_token_logits / temperature).numpy()
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumulative_probs, p, side='right')
        cutoff = min(cutoff, len(sorted_indices) - 1)
        top_indices = sorted_indices[:cutoff + 1]
        top_probs = sorted_probs[:cutoff + 1]
        top_probs /= np.sum(top_probs)

        next_token_id = int(np.random.choice(top_indices, p=top_probs))
        generated.append(next_token_id)

        # 스트리밍용: 지금까지 생성된 토큰을 디코딩 (띄어쓰기 적용)
        partial_text = sp.decode(generated)
        # 마지막 출력과 차이만 스트리밍
        yield partial_text

        if next_token_id == end_id and len(generated) >= min_len:
            break


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
    for new_text in generate_text_streaming_fixed(model, prompt):
        # 새로 생긴 부분만 출력
        print(new_text[last_len:], end="", flush=True)
        text = new_text
        last_len = len(text)

    # 출력 후 마지막 길이 초기화
    last_len = 0
    print("\n")
