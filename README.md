# Attention Is All You Need: from scratch reimplementation

This is a Pytorch reimplementation of the original encoder-decoder Transformer from [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762), trained on a ~4M-pair subset of WMT14 English -> French. Everything (model, data pipeline, tokenizer, training, beam search) is written from scratch on top of plain PyTorch.

## Model

Base configuration from the paper: 6 encoder + 6 decoder layers, `d_model=512`, 8 heads, FFN 2048, sinusoidal positional encodings, weight tying between the embedding and the output projection. ~60.5M parameters.

Trained with Adam (β=(0.9, 0.98), ε=1e-9), the paper's inverse-sqrt warmup schedule (1500 warmup steps (~10% of total steps)), label smoothing 0.1, bf16 autocast and `torch.compile`. Batches are bucketed by length with a custom `LengthBatchSampler` (shuffle → sort within mega-chunks of 50 batches → shuffle batches) to cut padding waste. 1 epoch take ~20 min total on a single A100 GPU.

## Deviations from the paper

- **Pre-LN instead of post-LN.** I first implemented the paper's original post-norm residual layout and could not get it to train reliably: runs would learn for 1–3 epochs, then the gradient norm would explode (`gn` jumping from ~1 to >3000) and the model collapsed into emitting a single token ("de de de de..."). Two full divergent runs are in
  [`logs/postln_divergence.txt`](logs/postln_divergence.txt). This is the known warmup sensitivity of post-norm Transformers (see e.g. [Xiong et al., 2020](https://arxiv.org/abs/2002.04745)). Switching to pre-LN (norm inside the residual branch, plus final LNs after the encoder and decoder stacks) trained smoothly on the first try with the same hyperparameters, that run is [`logs/preln_training.txt`](logs/preln_training.txt).
- **Dropout**: Since I trained for only 1 epoch, it wasn't useful to use dropout since I can't overfit. That's why set the dropout to 0.0 instead of the papers 0.1.
- Single GPU for ~20min, only ~4M pairs for 1 epoch, way less than the paper's compute, so scores are not comparable to the paper's BLEU 41+.

## Results

The model achieved a Greedy-decode BLEU of 27.3 on a held-out subset, measured with `torchmetrics.BLEUScore` on ~3000 sentences.

Beam search (beam 4, GNMT length penalty α=0.6) is used at inference time.
Sample output from the final checkpoint:

> **EN (input):** I'm Abdou, a student at UofT with a strong interest in computer science, especially AI and its use in science. I'm fascinated by the idea that we can build
> intelligent systems that help scientists analyze data, test ideas, and discover things that are revolutionanry. I want to better understand how these systems work and how > we can make them more useful and reliable. My goal is to develop the skills needed to build AI systems that can help accelerate research and contribute to solving
> important scientific problems.
>
> **FR (model):** Je suis Abdou, étudiant à UofT avec un vif intérêt pour la science informatique, particulièrement AI et son utilisation en sciences. Je suis fasciné par
> l'idée que nous pouvons bâtir des systèmes intelligents qui aident les scientifiques à analyser les données, à tester les idées et à découvrir les choses qui sont
> révolutionnaires, et je veux mieux comprendre comment ces systèmes fonctionnent et nous pouvons les rendre plus utiles et fiables. Mon objectif est de développer les
> compétences nécessaires pour construire des systèmes d'AI qui peuvent aider à accélérer la recherche et contribuer à résoudre d'importants problèmes scientifiques.


## Usage

```bash
pip install -r requirements.txt

# train (downloads + tokenizes 4M WMT14 pairs on first run, CUDA required)
python train.py

# translate with the latest checkpoint (beam search)
python translate.py "The mirror of the telescope is segmented to reduce costs."
```

All hyperparameters live in `config.py`. Checkpoints (model + optimizer + scheduler state) are saved every epoch to `wmt_wmt14_weights/`.

## Files

- `model.py`: attention, encoder/decoder blocks, positional encodings, greedy + beam decoding
- `dataset.py`: dataset with junk/length filtering, padding collate, length-bucketed batch sampler
- `train.py`: data loading, tokenizer training, training loop, greedy-BLEU validation
- `translate.py`: checkpoint loading + beam-search translation CLI
- `config.py`: config dict, path helpers, tokenizer builder, detokenization cleanup
- `logs/`:  raw training logs for the pre-LN run and the two failed post-LN runs
- `wmt14_transformer_colab.ipynb`: jupyter notebook used to train the model





