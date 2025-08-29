import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import sentencepiece as spm
import json
from tqdm import tqdm
import requests

# =======================
# 0) 파일 다운로드
# =======================
def download_file(url, save_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✅ 파일 저장됨: {save_path}")

download_file('https://huggingface.co/datasets/Yuchan5386/Smolwrite-dataset/resolve/main/VeTrans.jsonl?download=true', 'VeTrans.jsonl')
download_file('https://huggingface.co/datasets/Yuchan5386/Smolwrite-dataset/resolve/main/unigram.model?download=true', 'ko_unigram.model')
# =======================
# 1) 토크나이저 로드 & 특수토큰 안전화
# =======================
sp = spm.SentencePieceProcessor()
sp.load('ko_unigram.model')
vocab_size = sp.get_piece_size()

pad_id = sp.piece_to_id("<pad>") or 0
start_id = sp.piece_to_id("<start>") or sp.bos_id() or 1
end_id = sp.piece_to_id("<end>") or sp.eos_id() or 2

print("TOKENS:", {"pad": pad_id, "start": start_id, "end": end_id, "vocab": vocab_size})

max_len = 100
batch_size = 64

def data_generator(file_path, max_len=100, pad_id=0, start_id=1, end_id=2, max_samples=None):
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if max_samples and count >= max_samples:
                break

            item = json.loads(line)

            # conversations 길이 확인 (2 미만이면 스킵)
            if "conversations" not in item or len(item["conversations"]) < 2:
                continue

            src = item["conversations"][0].get("value", "").strip()
            tgt = item["conversations"][1].get("value", "").strip()

            # 텍스트가 너무 짧거나 같은 경우 스킵
            if len(src) < 2 or len(tgt) < 2 or src == tgt:
                continue

            def encode_text(text):
                ids = sp.encode(text, out_type=int)
                ids = [start_id] + ids + [end_id]
                if len(ids) < max_len:
                    ids += [pad_id] * (max_len - len(ids))
                else:
                    ids = ids[:max_len]
                    if ids[-1] != end_id:
                        ids[-1] = end_id
                return np.array(ids, dtype=np.int32)

            yield (encode_text(src), encode_text(tgt))
            count += 1


stream_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator("VeTrans.jsonl", max_len=max_len, pad_id=pad_id, start_id=start_id, end_id=end_id, max_samples=3804190),
    output_signature=(
        tf.TensorSpec(shape=(max_len,), dtype=tf.int32),
        tf.TensorSpec(shape=(max_len,), dtype=tf.int32)
    )
)
stream_dataset = stream_dataset.shuffle(5000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Map dataset to fit Keras (dec_input, dec_target)
def map_fn(src, tgt):
    dec_input = tgt[:, :-1]
    dec_target = tgt[:, 1:]
    return ({"enc_inputs": src, "dec_inputs": dec_input}, dec_target)

train_ds = stream_dataset.map(map_fn)

class SwiGLU(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.proj = tf.keras.layers.Dense(d_ff * 2)
        self.out = tf.keras.layers.Dense(d_model)

    def call(self, x):
        x_proj = self.proj(x)
        x_val, x_gate = tf.split(x_proj, 2, axis=-1)
        return self.out(x_val * tf.nn.silu(x_gate))
        
# =======================
# 3) Transformer 모델 정의
# =======================
class EncoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = SwiGLU(d_model, dff)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, x, mask=None, training=False):
        attn_out = self.mha(x, x, x, attention_mask=mask)
        attn_out = self.dropout1(attn_out, training=training)
        out1 = self.norm1(x + attn_out)
        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out, training=training)
        out2 = self.norm2(out1 + ffn_out)
        return out2

class DecoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.self_mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.cross_mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = SwiGLU(d_model, dff)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        self.dropout3 = layers.Dropout(dropout)

    def call(self, x, enc_out, training=False):
        attn1 = self.self_mha(x, x, x, use_causal_mask=True)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.norm1(x + attn1)
        attn2 = self.cross_mha(out1, enc_out, enc_out)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.norm2(out1 + attn2)
        ffn_out = self.ffn(out2)
        ffn_out = self.dropout3(ffn_out, training=training)
        out3 = self.norm3(out2 + ffn_out)
        return out3

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size,
                 max_len=100, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        
        # 🔹 입력 임베딩 + 위치 임베딩
        self.enc_embedding = layers.Embedding(input_vocab_size, d_model)
        self.enc_pos_embedding = layers.Embedding(max_len, d_model)
        
        self.dec_embedding = layers.Embedding(target_vocab_size, d_model)
        self.dec_pos_embedding = layers.Embedding(max_len, d_model)
        
        self.enc_layers = [EncoderBlock(d_model, num_heads, dff, dropout) for _ in range(num_layers)]
        self.dec_layers = [DecoderBlock(d_model, num_heads, dff, dropout) for _ in range(num_layers)]
        self.final_layer = layers.Dense(target_vocab_size)

    def call(self, inputs, training=False):
        enc_inputs = inputs["enc_inputs"]
        dec_inputs = inputs["dec_inputs"]

        # 🔹 위치 인덱스 생성
        enc_pos = tf.range(tf.shape(enc_inputs)[1])[tf.newaxis, :]
        dec_pos = tf.range(tf.shape(dec_inputs)[1])[tf.newaxis, :]

        # 🔹 임베딩 + 위치 임베딩
        x = self.enc_embedding(enc_inputs) + self.enc_pos_embedding(enc_pos)
        for layer in self.enc_layers:
            x = layer(x, training=training)
        enc_out = x

        y = self.dec_embedding(dec_inputs) + self.dec_pos_embedding(dec_pos)
        for layer in self.dec_layers:
            y = layer(y, enc_out, training=training)

        logits = self.final_layer(y)
        return logits


model = Transformer(num_layers=4, d_model=160, num_heads=8, dff=640,
                    input_vocab_size=vocab_size, target_vocab_size=vocab_size) 

# 4) 옵티마이저 + 손실
# =======================
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)

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


tf.config.optimizer.set_jit(True)  # 전체 모델에 대해 XLA 활성화
model.compile(
    optimizer=optimizer,
    loss=smoothed_loss_keras,
    metrics=[masked_accuracy],
    run_eagerly=False
)

# =======================
# 5) 학습
epochs = 1
model.fit(train_ds, epochs=epochs)

model.save_weights("smolwrite_tuning.weights.h5")
print("모델 가중치 저장 완료!")

# =======================
# 6) Top-p / greedy 샘플링 추론
# =======================
def top_p_sample(logits, p=0.9):
    sorted_ids = tf.argsort(logits, direction='DESCENDING')
    sorted_logits = tf.gather(logits, sorted_ids)
    probs = tf.nn.softmax(sorted_logits)
    cumulative_probs = tf.cumsum(probs)
    cutoff = tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.float32))
    top_ids = sorted_ids[:tf.cast(cutoff+1, tf.int32)]
    top_logits = tf.gather(logits, top_ids)
    top_probs = tf.nn.softmax(top_logits)
    return np.random.choice(top_ids.numpy(), p=top_probs.numpy())

def generate_text(src_text, max_len=100, top_p=0.9):
    def encode_text(text):
        ids = sp.encode(text, out_type=int)
        ids = [start_id] + ids + [end_id]
        if len(ids) < max_len:
            ids += [pad_id] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
            if ids[-1] != end_id: ids[-1] = end_id
        return ids

    src_ids = np.array([encode_text(src_text)])
    enc_emb = model.enc_embedding(src_ids)
    for layer in model.enc_layers:
        enc_emb = layer(enc_emb, training=False)

    out_seq = [start_id]
    for _ in range(max_len - 1):
        dec_ids = np.array([out_seq])
        dec_emb = model.dec_embedding(dec_ids)
        for layer in model.dec_layers:
            dec_emb = layer(dec_emb, enc_emb, training=False)
        logits = model.final_layer(dec_emb)[0, -1]
        next_id = top_p_sample(logits, p=top_p)
        if next_id == end_id:
            break
        out_seq.append(next_id)

    # ⚡ 여기서 타입을 명시적으로 int로 변환
    toks = [int(t) for t in out_seq if t not in (pad_id, start_id, end_id)]
    return sp.decode(toks)


# =======================
# 7) 테스트
print(generate_text("안녕하세요!"))
print(generate_text("오늘은 뭐 할 거야?"))
print(generate_text("오늘 기분은 어때?"))  
