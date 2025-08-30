# Smolite

Smolite는 효율적이고 경량화된 트랜스포머 기반 텍스트 생성 모델이며, 자연스러운 일상 대화 생성에 최적화되어 있습니다. 

---
### 학습 및 추론 코드 보기
- [바로가기](https://github.com/INSECT5386/Smolite-1/tree/main/smolite)

## 데이터셋

Smolite는 다음 데이터셋으로 되었습니다:
* [Smollite Dataset](https://huggingface.co/datasets/Yuchan5386/Smolwrite-dataset)

## 모델 저장소
* [저장소 바로가기](https://huggingface.co/Yuchan5386/Smolite-1)
---
## 모델 하이퍼파라미터

| 파라미터          | 값              |
| ------------- | -------------- |
| vocab\_size   | 72000 |
| seq\_len      | 100       |
| d\_model      | 256            |
| n\_layers     | 6            |
| d\_ff         | 1024           |
| num\_heads    | 8              |
| dropout\_rate | 0.1            |
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
