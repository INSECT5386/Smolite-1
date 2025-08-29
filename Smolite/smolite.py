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
    "https://huggingface.co/datasets/Yuchan5386/Smolwrite-dataset/resolve/main/unigram.model?download=true",
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
batch_size = 96

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

class SwiGLU(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.proj = tf.keras.layers.Dense(d_ff)
        self.adapter = tf.keras.layers.Dense(d_model // 2, activation='gelu')
        self.out = tf.keras.layers.Dense(d_model)
    def call(self, x):
        x = self.adapter(x)
        x_proj = self.proj(x)
        x_val, x_gate = tf.split(x_proj, 2, axis=-1)
        return self.out(x_val * tf.nn.silu(x_gate))

class LightweightAttn(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads=8, dropout_rate=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = d_model // num_heads
        self.q_proj = tf.keras.layers.Dense(d_model)
        self.k_proj = tf.keras.layers.Dense(d_model)
        self.v_proj = tf.keras.layers.Dense(d_model)
        self.o_proj = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def apply_rope(self, x):
        # x: [B, H, L, D_head]
        seq_len = tf.shape(x)[2]
        inv_freq = 1.0 / (10000 ** (np.arange(0, self.d_head, 2) / self.d_head))
        freqs = tf.einsum('i,j->ij', tf.range(seq_len, dtype=tf.float32), inv_freq)
        sin = tf.reshape(tf.sin(freqs), [1, 1, seq_len, -1])
        cos = tf.reshape(tf.cos(freqs), [1, 1, seq_len, -1])
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x_rot = tf.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
        return tf.reshape(x_rot, tf.shape(x))

    def call(self, x, training=False):
        # LayerNorm + Residual
        x_norm = self.ln(x)
        B, L, _ = tf.shape(x_norm)[0], tf.shape(x_norm)[1], tf.shape(x_norm)[2]
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)
        q = tf.reshape(q, [B, L, self.num_heads, self.d_head])
        k = tf.reshape(k, [B, L, self.num_heads, self.d_head])
        v = tf.reshape(v, [B, L, self.num_heads, self.d_head])
        q = tf.transpose(q, [0,2,1,3])  # [B,H,L,D_head]
        k = tf.transpose(k, [0,2,1,3])
        v = tf.transpose(v, [0,2,1,3])
        q = self.apply_rope(q)
        k = self.apply_rope(k)
        attn_scores = tf.einsum('bhid,bhjd->bhij', q, k) / tf.math.sqrt(tf.cast(self.d_head, tf.float32))
        mask = tf.linalg.band_part(tf.ones((L, L)), -1, 0)  # lower triangular
        mask = tf.reshape(mask, [1, 1, L, L])
        attn_scores = attn_scores * mask + (1.0 - mask) * (-1e9)
        attn_probs = tf.nn.softmax(attn_scores, axis=-1)
        attn_probs = self.dropout(attn_probs, training=training)
        out = tf.einsum('bhij,bhjd->bhid', attn_probs, v)
        out = tf.transpose(out, [0,2,1,3])
        out = tf.reshape(out, [B, L, self.d_model])
        out = self.o_proj(out)
        return x + out

class Block(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, num_heads=8, dropout_rate=0.1):
        super().__init__()
        self.attn1 = LightweightAttn(d_model, num_heads=num_heads, dropout_rate=dropout_rate)
        self.ffn = SwiGLU(d_model, d_ff)
        self.ln_f1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)  
        self.ln_f2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)  
        self.ln_f3 = tf.keras.layers.LayerNormalization(epsilon=1e-5)  
    def call(self, x):
        skip = x
        x = self.ln_f1(x)
        x = self.attn1(x) 
        x = self.ln_f2(x)
        x = self.ffn(x)
        x = self.ln_f3(x)
        return x + skip
    
class Smolite(tf.keras.Model):  
    def __init__(self, vocab_size, seq_len, d_model,n_layers, d_ff, num_heads=8, dropout_rate=0.1):  
        super().__init__()  
        self.token_embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.blocks = [Block(d_model, d_ff, num_heads=8, dropout_rate=0.1) for _ in range(n_layers)]
        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=1e-5)  
    def call(self, x, training=False):  
        x = self.token_embedding(x)
        for blk in self.blocks:
            x = blk(x, training=training)
        logits = tf.matmul(x, self.token_embedding.embeddings, transpose_b=True)  
        return logits
    
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

# =======================
# 7) 모델 초기화 및 컴파일
# =======================
model = Smolite(vocab_size, seq_len=max_len, d_model=256, n_layers=12, d_ff=1024, num_heads=8, dropout_rate=0.1)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.95, epsilon=1e-8, clipnorm=1.0)
model.compile(
    optimizer=optimizer,
    loss=smoothed_loss_keras,
    metrics=[masked_accuracy, masked_perplexity],
    run_eagerly=False
)
# 더미 입력으로 모델 초기화
dummy_input = np.zeros((1, max_len), dtype=np.int32)
model(dummy_input)
model.summary()

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
model.save_weights("model.weights.h5")
print("✅ 모델 가중치 저장 완료!")

def generate_text_topp(model, prompt, max_len=100, max_gen=98, p=0.9, temperature=0.8, min_len=20):
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
