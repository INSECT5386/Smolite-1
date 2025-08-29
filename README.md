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
        # x: [B, H, L, D_head]
        seq_len = tf.shape(x)[2]
        inv_freq = 1.0 / (10000 ** (np.arange(0, self.d_head, 2) / self.d_head))
        freqs = tf.einsum('i,j->ij', tf.range(seq_len, dtype=tf.float32), inv_freq)
        sin = tf.r습니다:

* [Smolwrite Dataset](https://huggingface.co/datasets/Yuchan5386/Smolwrite-dataset)

