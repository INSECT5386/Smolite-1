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


download_file(
    "https://huggingface.co/datasets/Yuchan5386/Smolwrite-dataset/resolve/main/unigram.model?download=true",
    "ko_unigram.model"
)

# =======================
# 2) 토크나이저 로드
# =======================
sp = spm.SentencePieceProcessor()
sp.load("ko_unigram.model")

pad_id = sp.piece_to_id("<pad>") if sp.piece_to_id("<pad>") != -1 else 0
start_id = sp.piece_to_id("<start>")
end_id = sp.piece_to_id("<end>")
vocab_size = sp.get_piece_size()

max_len = 100
batch_size = 64

def text_to_ids(text):
    return sp.encode(text, out_type=int)

def ids_to_text(ids):
    return sp.decode(ids)

class SwiGLU(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.proj = tf.keras.layers.Dense(d_ff * 2)
        self.out = tf.keras.layers.Dense(d_model)
    def call(self, x):
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

class SSL(tf.keras.layers.Layer):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate_proj = tf.keras.layers.Dense(hidden_dim)
        self.input_proj = tf.keras.layers.Dense(hidden_dim)

        self.A = self.add_weight(shape=(hidden_dim,),initializer=tf.keras.initializers.RandomNormal(mean=-0.5, stddev=0.1),trainable=True, name="A")
        self.B = self.add_weight(shape=(hidden_dim,),initializer='random_normal',trainable=True, name="B")
        self.C = self.add_weight(shape=(hidden_dim,),initializer='random_normal',trainable=True, name="C")
        self.D = self.add_weight(shape=(hidden_dim,),initializer='zeros',trainable=True, name="D")
        self.norm = tf.keras.layers.LayerNormalization()
        self.output_proj = tf.keras.layers.Dense(hidden_dim)

    def fft_convolve(self, u_t, kernel_t, T):
        pad_len = T - 1
        seq_len = T + pad_len
        fft_len_float = tf.math.ceil(tf.math.log(tf.cast(seq_len, tf.float32)) / tf.math.log(2.0))
        fft_len = tf.cast(2 ** fft_len_float, tf.int32)
        u_padded = tf.pad(u_t, [[0, 0], [0, 0], [pad_len, fft_len - seq_len]])
        K_padded = tf.pad(kernel_t, [[0, 0], [0, fft_len - T]])
        U_f = tf.signal.fft(tf.cast(tf.complex(u_padded, 0.0), tf.complex64))
        K_f = tf.signal.fft(tf.cast(tf.complex(K_padded, 0.0), tf.complex64))
        Y_f = U_f * tf.expand_dims(K_f, 0)
        y_full = tf.signal.ifft(Y_f)
        y_real = tf.math.real(y_full)[..., pad_len:pad_len + T]
        return y_real

    def call(self, x):
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        D = self.hidden_dim
        gate = tf.nn.silu(self.gate_proj(x))
        x_proj = self.input_proj(x)
        u = gate * x_proj
        time_idx = tf.cast(tf.range(T), dtype=self.A.dtype)[:, None]
        A_pow = tf.pow(tf.expand_dims(self.A, 0), time_idx)
        kernel = self.B * A_pow
        u_t = tf.transpose(u, [0, 2, 1])
        kernel_t = tf.transpose(kernel, [1, 0])
        y_real = self.fft_convolve(u_t, kernel_t, T)
        y = tf.transpose(y_real, [0, 2, 1])
        y = self.C * y + self.D * u
        y = self.norm(y)
        y = self.output_proj(y)
        return y

class Block(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, num_heads=8, dropout_rate=0.1):
        super().__init__()
        self.attn1 = LightweightAttn(d_model, num_heads=num_heads, dropout_rate=dropout_rate)
        self.attn2 = LightweightAttn(d_model, num_heads=num_heads, dropout_rate=dropout_rate)
        self.ssl1 = SSL(hidden_dim=d_model)
        self.ssl2 = SSL(hidden_dim=d_model)
        self.ffn = SwiGLU(d_model, d_ff)
    def call(self, x):
        x = self.attn1(x) 
        x = self.ssl1(x)
        x = self.ssl2(x)
        x = self.attn2(x)
        x = self.ffn(x)
        return x
    
class Model(tf.keras.Model):  
    def __init__(self, vocab_size, seq_len, d_model,n_layers, d_ff, num_heads=8, dropout_rate=0.1):  
        super().__init__()  
        self.token_embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.out = tf.keras.layers.Dense(vocab_size)
        self.blocks = [Block(d_model, d_ff, num_heads=8, dropout_rate=0.1) for _ in range(n_layers)]
        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=1e-5)  
    def call(self, x, training=False):  
        x = self.token_embedding(x)
        for blk in self.blocks:
            x = blk(x, training=training)
        logits = self.out(x) 
        return logits  
# ✅ 먼저 더미 입력으로 모델 빌드
dummy_input = np.zeros((1, max_len), dtype=np.int32)
model(dummy_input)  # <- 여기서 모델 빌드됨

# 이제 가중치 로드 가능
model.load_weights(r'C:\Users\yuchan\Serve_Project\smolwrite\gpt\model.weights.h5')
print("모델 가중치 로드 완료!")

import re

def ids_to_text(ids):
    # int로 변환 후 decode
    text = sp.decode([int(i) for i in ids])
    # SentencePiece의 ▁를 단어 앞 공백으로 변환
    # 맨 앞의 공백은 제거
    text = re.sub(r'▁+', ' ', text).strip()
    return text

def generate_text_streaming_fixed(model, prompt, max_len=100, max_gen=100, p=0.9, temperature=0.7, min_len=10, repetition_penalty=1.1):
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
