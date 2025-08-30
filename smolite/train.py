import json
import numpy as np
import tensorflow as tf
import sentencepiece as spm
import requests
import pyarrow.parquet as pq
from smolite.model import Smolite
from generate-tokenizer import generate, ids_to_text, text_to_ids
from smolite.loss-acc import smoothed_loss_keras, masked_accuracy, masked_perplexity

def download_file(url, save_path):
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(save_path, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    print(f"✅ {save_path} 저장됨")

download_file(
    "https://huggingface.co/datasets/Yuchan5386/Smolwrite-dataset/resolve/main/VeTrans.jsonl?download=true",
    "converted.jsonl"
)
download_file(
    "https://huggingface.co/datasets/Yuchan5386/Smolwrite-dataset/resolve/main/kolig_unigram.model?download=true",
    "ko_unigram.model"
)

max_len = 100
batch_size = 64

def jsonl_stream(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            conversations = data.get("conversations", [])
            
            # 두 개씩 짝지어서 human->GPT만
            for i in range(0, len(conversations) - 1, 2):
                human_msg = conversations[i]
                gpt_msg = conversations[i + 1]

                if human_msg.get("from") != "human" or gpt_msg.get("from") != "gpt":
                    continue

                prompt = human_msg.get("value", "").replace("\n", " ").strip()
                response = gpt_msg.get("value", "").replace("\n", " ").strip()
                full = f"<start> {prompt} <sep> {response} <end>"

                if "<sep>" not in full:
                    continue

                sep_index = full.index("<sep>")
                input_text = full[:sep_index + len("<sep>")].strip()
                target_text = full[sep_index + len("<sep>"):].strip()

                input_ids = text_to_ids(input_text)
                target_ids = text_to_ids(target_text + " <end>")

                available_len = max_len - len(input_ids)
                if available_len <= 0:
                    input_ids = input_ids[-max_len:]
                    target_ids = []
                    target_mask = [0] * len(input_ids)
                else:
                    target_ids = target_ids[:available_len]
                    target_mask = [0] * len(input_ids) + [1] * len(target_ids)

                full_input = input_ids + target_ids
                pad_len = max_len - len(full_input)
                full_input += [pad_id] * pad_len
                target_mask += [0] * pad_len

                target_seq = full_input[1:] + [end_id]
                target_seq = target_seq[:max_len]

                masked_target = [
                    t if m == 1 else pad_id
                    for t, m in zip(target_seq, target_mask)
                ]

                yield (
                    tf.convert_to_tensor(full_input, dtype=tf.int32),
                    tf.convert_to_tensor(masked_target, dtype=tf.int32)
                )

dataset = tf.data.Dataset.from_generator(
    lambda: jsonl_stream("converted.jsonl"),
    output_signature=(
        tf.TensorSpec(shape=(max_len,), dtype=tf.int32),
        tf.TensorSpec(shape=(max_len,), dtype=tf.int32)
    )
)

dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
print("✅ 스트리밍 TF Dataset 준비 완료!")
 
model = Smolite(vocab_size=vocab_size, seq_len=max_len, d_model=256, d_ff=1024, n_layers=6)    
dummy_input = tf.zeros((1, max_len), dtype=tf.int32)  max_len  
_ = model(dummy_input) 
model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.95, epsilon=1e-8, clipnorm=1.0)
model.compile(
    optimizer=optimizer,
    loss=smoothed_loss_keras,
    metrics=[masked_accuracy, masked_perplexity],
    run_eagerly=False
)
history = model.fit(
    dataset,
    epochs=1,
    verbose=1
)

model.save_weights("model1.weights.h5")
print("✅ 모델 가중치 저장 완료!")

print("\n\n===== 생성 결과 =====")  
prompt = '"안녕하세요! 한국 밴드에 대해 궁금한 것이 있어요!"'
print(generate(model, prompt, max_len=100, max_gen=98, p=0.9, temperature=0.2, min_len=20))



