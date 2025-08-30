import json
import numpy as np
import tensorflow as tf
import sentencepiece as spm
import requests
import pyarrow.parquet as pq

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

# =======================
# 1) 데이터 및 토크나이저 다운로드
# =======================
download_file(
    "https://huggingface.co/datasets/Yuchan5386/Smolwrite-dataset/resolve/main/VeTrans.jsonl?download=true",
    "converted.jsonl"
)
download_file(
    "https://huggingface.co/datasets/Yuchan5386/Smolwrite-dataset/resolve/main/kolig_unigram.model?download=true",
    "ko_unigram.model"
)


# =======================
# 토크나이저 로드
# =======================
sp = spm.SentencePieceProcessor()
sp.load("ko_unigram.model")

pad_id = sp.piece_to_id("<pad>") if sp.piece_to_id("<pad>") != -1 else 0
start_id = sp.piece_to_id("<start>")
sep_id = sp.piece_to_id("<sep>")
end_id = sp.piece_to_id("<end>")
unk_id = sp.piece_to_id("<unk>")
vocab_size = sp.get_piece_size()
print(f"✅ Vocabulary size: {vocab_size}")

max_len = 100
batch_size = 64

def text_to_ids(text):
    return sp.encode(text, out_type=int)
def ids_to_text(ids):
    return sp.decode(ids)
# =======================
# JSONL 스트리밍 generator
# =======================
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

# =======================
# TF Dataset 스트리밍
# =======================
dataset = tf.data.Dataset.from_generator(
    lambda: jsonl_stream("converted.jsonl"),
    output_signature=(
        tf.TensorSpec(shape=(max_len,), dtype=tf.int32),
        tf.TensorSpec(shape=(max_len,), dtype=tf.int32)
    )
)

dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

print("✅ 스트리밍 TF Dataset 준비 완료!")


class RotaryPositionalEmbedding(tf.keras.layers.Layer):  
    def __init__(self, dim):  
        super().__init__()  
        inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))  
        self.inv_freq = tf.constant(inv_freq, dtype=tf.float32)  
  
    def call(self, x):  
        batch, heads, seq_len, depth = tf.unstack(tf.shape(x))  
        t = tf.range(seq_len, dtype=tf.float32)  
        freqs = tf.einsum('i,j->ij', t, self.inv_freq)  
        emb_sin = tf.sin(freqs)  
        emb_cos = tf.cos(freqs)  
        emb_cos = tf.reshape(emb_cos, [1, 1, seq_len, -1])  
        emb_sin = tf.reshape(emb_sin, [1, 1, seq_len, -1])  
        x1 = x[..., ::2]  
        x2 = x[..., 1::2]  
        x_rotated = tf.stack([  
            x1 * emb_cos - x2 * emb_sin,  
            x1 * emb_sin + x2 * emb_cos  
        ], axis=-1)  
        x_rotated = tf.reshape(x_rotated, tf.shape(x))  
        return x_rotated

class SwiGLU(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.proj = tf.keras.layers.Dense(d_ff * 2)
        self.out = tf.keras.layers.Dense(d_model)

    def call(self, x):
        x_proj = self.proj(x)
        x_val, x_gate = tf.split(x_proj, 2, axis=-1)
        return self.out(x_val * tf.nn.silu(x_gate))
        
class GPTBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, num_heads=8, dropout_rate=0.1, adapter_dim=64):  
        super().__init__()  
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)  
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)  
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate) 
        self.adapter_down = tf.keras.layers.Dense(adapter_dim, activation='gelu') 
        self.adapter_up = tf.keras.layers.Dense(d_model)  
  
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)  
        self.ffn = SwiGLU(d_model, d_ff)  
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate) 
        self.rope = RotaryPositionalEmbedding(d_model // num_heads)  
  
    def call(self, x, training=False):  
        x_norm = self.ln1(x)  
        b, s, _ = tf.shape(x_norm)[0], tf.shape(x_norm)[1], tf.shape(x_norm)[2]  
        h = self.mha.num_heads  
        d = x_norm.shape[-1] // h  
  
        qkv = tf.reshape(x_norm, [b, s, h, d])  
        qkv = tf.transpose(qkv, [0, 2, 1, 3])  
        q = self.rope(qkv)  
        k = self.rope(qkv)  
        q = tf.reshape(tf.transpose(q, [0, 2, 1, 3]), [b, s, h * d])  
        k = tf.reshape(tf.transpose(k, [0, 2, 1, 3]), [b, s, h * d])  
  
        attn_out = self.mha(query=q, value=x_norm, key=k, use_causal_mask=True, training=training)  
        attn_out = self.dropout1(attn_out, training=training)  

        adapter_out = self.adapter_up(self.adapter_down(attn_out))
        attn_out = attn_out + adapter_out  
  
        x = x + attn_out  
        ffn_out = self.ffn(self.ln2(x))  
        x = x + self.dropout2(ffn_out, training=training)  
        return x

class InteractGPT(tf.keras.Model):  
    def __init__(self, vocab_size, seq_len, d_model, d_ff, n_layers, num_heads=8, dropout_rate=0.1):  
        super().__init__()  
        self.token_embedding = tf.keras.layers.Embedding(vocab_size, d_model)  
        self.blocks = [GPTBlock(d_model, d_ff, num_heads, dropout_rate) for _ in range(n_layers)]  
        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=1e-5)  
  
    def call(self, x, training=False):  
        x = self.token_embedding(x)  
        for block in self.blocks:  
            x = block(x, training=training)  
        x = self.ln_f(x)  
        logits = tf.matmul(x, self.token_embedding.embeddings, transpose_b=True)  
        return logits  

model = InteractGPT(vocab_size=vocab_size, seq_len=max_len, d_model=256, d_ff=1024, n_layers=6)    
dummy_input = tf.zeros((1, max_len), dtype=tf.int32)  # 배치1, 시퀀스길이 max_len  
_ = model(dummy_input)  # 모델이 빌드됨  
model.summary()
print("모델 가중치 로드 완료!")  
def smoothed_loss_keras(y_true, y_pred, eps=0.1):
    y_true = tf.cast(y_true, tf.int32)   # ← 여기 추가
    mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
    vocab = tf.shape(y_pred)[-1]
    y_true_oh = tf.one_hot(y_true, depth=vocab, dtype=tf.float32)
    y_true_ls = (1.0 - eps) * y_true_oh + eps / tf.cast(vocab, tf.float32)
    log_probs = tf.nn.log_softmax(y_pred, axis=-1)
    per_tok = -tf.reduce_sum(y_true_ls * log_probs, axis=-1)
    per_tok = per_tok * mask
    return tf.reduce_sum(per_tok) / (tf.reduce_sum(mask)+1e-8)

def masked_accuracy(y_true, y_pred):
    # y_true를 int로 변환
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)  # pad 토큰 제외
    pred_id = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    acc = tf.cast(tf.equal(y_true, pred_id), tf.float32) * mask
    return tf.reduce_sum(acc) / (tf.reduce_sum(mask) + 1e-8)

def masked_perplexity(y_true, y_pred, eps=0.1):
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
    vocab = tf.shape(y_pred)[-1]
    y_true_oh = tf.one_hot(y_true, depth=vocab, dtype=tf.float32)
    y_true_ls = (1.0 - eps) * y_true_oh + eps / tf.cast(vocab, tf.float32)
    log_probs = tf.nn.log_softmax(y_pred, axis=-1)
    per_tok = -tf.reduce_sum(y_true_ls * log_probs, axis=-1)
    per_tok = per_tok * mask
    mean_loss = tf.reduce_sum(per_tok) / (tf.reduce_sum(mask) + 1e-8)
    return tf.exp(mean_loss)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.95, epsilon=1e-8, clipnorm=1.0)
model.compile(
    optimizer=optimizer,
    loss=smoothed_loss_keras,
    metrics=[masked_accuracy, masked_perplexity],
    run_eagerly=False
)
# =======================
# 8) 학습
# =======================
history = model.fit(
    dataset,
    epochs=1,
    verbose=1
)

# =======================
# 9) 가중치 저장
# =======================
model.save_weights("model1.weights.h5")
print("✅ 모델 가중치 저장 완료!")

def generate_text_topp(model, prompt, max_len=100, max_gen=98, p=0.9, temperature=0.2, min_len=20):
    model_input = text_to_ids(f"<start> {prompt} <sep>")
    model_input = model_input[:max_len]
    generated = list(model_input)
    for step in range(max_gen):
        if len(generated) > max_len:
            input_seq = generated[-max_len:]
        else:
            input_seq = generated
        input_padded = np.pad(input_seq, (0, max_len - len(input_seq)), constant_values=pad_id)
        input_tensor = tf.convert_to_tensor([input_padded])
        logits = model(input_tensor, training=False)
        next_token_logits = logits[0, len(input_seq) - 1].numpy()
        next_token_logits[end_id] -= 5.0
        next_token_logits[pad_id] -= 10.0
        probs = tf.nn.softmax(next_token_logits / temperature).numpy()
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumulative_probs, p)
        top_indices = sorted_indices[:cutoff + 1]
        top_probs = sorted_probs[:cutoff + 1]
        top_probs /= np.sum(top_probs)
        next_token_id = np.random.choice(top_indices, p=top_probs)
        if next_token_id == end_id and len(generated) >= min_len:
            break
        generated.append(int(next_token_id))
    return ids_to_text(generated)

print("\n\n===== 생성 결과 =====")  
print(generate_text_topp(model, "안녕하세요! 한국 밴드에 대해 궁금한 것이 있어요!", p=0.9))
