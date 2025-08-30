import tensorflow as tf
import numpy as np

def generate_max(model, prompt, max_len=100, max_gen=100, p=0.9, temperature=0.5, min_len=10, repetition_penalty=1.1):
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
