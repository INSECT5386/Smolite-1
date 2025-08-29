# Smolite

Smolite는 효율적이고 경량화된 트랜스포머 기반 텍스트 생성 모델입니다.  
주요 특징으로는 SwiGLU 활성화 기반 FFN, Lightweight Attention, RoPE(Rotary Positional Embedding) 적용 등이 있으며, 자연스러운 일상 대화 생성에 최적화되어 있습니다.

---

## 모델 구조

Smolite는 다음과 같은 구조로 구성됩니다:

1. **Token Embedding**  
   - 입력 토큰을 고정 차원(`d_model`)의 임베딩 벡터로 변환합니다.

2. **Block (Transformer Block)**  
   - **Lightweight Attention**: RoPE 적용, causal masking, multi-head self-attention  
   - **SwiGLU FFN**: GELU와 SiLU 활성화 조합  
   - Layer Normalization + Residual 연결  

3. **Output Projection**  
   - 입력 임베딩과 공유된 가중치를 사용하여 로짓(logits)을 생성합니다.

---

## 핵심 구성 요소

### SwiGLU
```python
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
```
- FFN 구조에서 Gated Linear Unit과 GELU/SiLU 활성화를 결합
- 모델의 파라미터 효율성과 학습 안정성 향상

### LightweightAttn
```python
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
        x_norm = self.ln(x)
        B, L, _ = tf.shape(x_norm)[0], tf.shape(x_norm)[1], tf.shape(x_norm)[2]
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)
        q = tf.reshape(q, [B, L, self.num_heads, self.d_head])
        k = tf.reshape(k, [B, L, self.num_heads, self.d_head])
        v = tf.reshape(v, [B, L, self.num_heads, self.d_head])
        q = tf.transpose(q, [0,2,1,3])
        k = tf.transpose(k, [0,2,1,3])
        v = tf.transpose(v, [0,2,1,3])
        q = self.apply_rope(q)
        k = self.apply_rope(k)
        attn_scores = tf.einsum('bhid,bhjd->bhij', q, k) / tf.math.sqrt(tf.cast(self.d_head, tf.float32))
        mask = tf.linalg.band_part(tf.ones((L, L)), -1, 0) 
        mask = tf.reshape(mask, [1, 1, L, L])
        attn_scores = attn_scores * mask + (1.0 - mask) * (-1e9)
        attn_probs = tf.nn.softmax(attn_scores, axis=-1)
        attn_probs = self.dropout(attn_probs, training=training)
        out = tf.einsum('bhij,bhjd->bhid', attn_probs, v)
        out = tf.transpose(out, [0,2,1,3])
        out = tf.reshape(out, [B, L, self.d_model])
        out = self.o_proj(out)
        return x + out
```
- Rotary Positional Embedding(RoPE) 적용
- causal mask 기반 self-attention
- multi-head attention, dropout 포함

### Block
- LayerNorm → Attention → LayerNorm → SwiGLU → LayerNorm → Residual 연결
- 다중 블록을 쌓아 심층 트랜스포머 모델 구성

---

### 손실 및 평가 지표
- Smoothed Cross-Entropy Loss
```python
def smoothed_loss_keras(y_true, y_pred, eps=0.1):
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
    vocab = tf.shape(y_pred)[-1]
    y_true_oh = tf.one_hot(y_true, depth=vocab, dtype=tf.float32)
    y_true_ls = (1.0 - eps) * y_true_oh + eps / tf.cast(vocab, tf.float32)
    log_probs = tf.nn.log_softmax(y_pred, axis=-1)
    per_tok = -tf.reduce_sum(y_true_ls * log_probs, axis=-1)
    per_tok = per_tok * mask
    return tf.reduce_sum(per_tok) / (tf.reduce_sum(mask)+1e-8)
```
- Masked Accuracy
```python
def masked_accuracy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
    pred_id = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    acc = tf.cast(tf.equal(y_true, pred_id), tf.float32) * mask
    return tf.reduce_sum(acc) / (tf.reduce_sum(mask) + 1e-8)
```
- Masked Perplexity
```python
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
```
- pad_id 토큰을 제외하고 계산하여 패딩 영향을 제거

---

## 모델 하이퍼파라미터

| 파라미터          | 값              |
| ------------- | -------------- |
| vocab\_size   | 32768 |
| seq\_len      | 100       |
| d\_model      | 256            |
| n\_layers     | 12             |
| d\_ff         | 1024           |
| num\_heads    | 8              |
| dropout\_rate | 0.1            |

---

## 데이터셋

Smolite는 다음 데이터셋으로 되었습니다:

* [Smolwrite Dataset](https://huggingface.co/datasets/Yuchan5386/Smolwrite-dataset)

--- 
## 모델 저장소
* [저장소 바로가기](https://huggingface.co/Yuchan5386/Smolite-1/settings)
