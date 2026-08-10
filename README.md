# Attention Is All You Need: from scratch reimplementation

This is a Pytorch reimplementation of the original encoder-decoder Transformer from [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762), trained on a ~4M-pair subset of WMT14 English -> French. Everything (model, data pipeline, tokenizer, training, beam search) is written from scratch on top of plain PyTorch.

## Model

Base configuration from the paper: 6 encoder + 6 decoder layers, `d_model=512`, 8 heads, FFN 2048, sinusoidal positional encodings, weight tying between the embedding and the output projection. ~60.5M parameters.

Trained with Adam (β=(0.9, 0.98), ε=1e-9), the paper's inverse-sqrt warmup schedule (1500 warmup steps (~10% of total steps)), label smoothing 0.1, bf16 autocast and `torch.compile`. Batches are bucketed by length with a custom `LengthBatchSampler` (shuffle → sort within mega-chunks of 50 batches → shuffle batches) to cut padding waste. 1 epoch take ~20 min total on a single A100 GPU.
