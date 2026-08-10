# model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
  def __init__(self, d_model: int, h: int, dropout: float):
    super().__init__()
    self.w_q = nn.Linear(d_model, d_model, bias = False)
    self.w_k = nn.Linear(d_model, d_model, bias = False)
    self.w_v = nn.Linear(d_model, d_model, bias = False)
    self.w_o = nn.Linear(d_model, d_model, bias = False)
    self.d_model = d_model
    self.h = h
    assert d_model % h == 0, "d_model should be divisble by the number of heads"
    self.d_k = d_model // h
    self.dropout = dropout
    self.scale = self.d_k ** -0.5

  def split_heads(self, x):
    return x.reshape(x.shape[0], x.shape[1], self.h, self.d_k).swapaxes(1,2)

  def forward(self, q, k, v, mask = None, is_causal: bool = False, is_fast: bool = True):
    # mask is a boolean Tensor where True is a real token
    # (B, seq_len, d_model)
    q, k, v = self.split_heads(self.w_q(q)), self.split_heads(self.w_k(k)), self.split_heads(self.w_v(v)) # (B, h, seq_len, d_k)

    if is_fast:
      x = F.scaled_dot_product_attention(q, k, v, mask, self.dropout if self.training else 0.0, is_causal)
    else:
      scores = (q @ k.swapaxes(-1, -2)) * self.scale
      if is_causal:
        causal = torch.ones((scores.shape[-2], scores.shape[-1]), device=scores.device, dtype=torch.bool).tril()
        mask = causal if mask is None else (mask & causal)
      if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
      probs = F.dropout(scores.softmax(-1), self.dropout, self.training)
      x = probs @ v

    # x.shape == (B, h, seq_len, d_k)
    x = x.swapaxes(1,2).reshape(x.shape[0], x.shape[2], self.d_model)
    return self.w_o(x) # (B, seq_len, d_model)


class EncoderBlock(nn.Module):
  def __init__(self, d_model, h, dropout):
    super().__init__()
    self.self_mha = MultiHeadAttention(d_model, h, dropout)
    self.ln1 = nn.LayerNorm(d_model)
    self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Linear(d_model * 4, d_model))
    self.ln2 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, mask):
    # mask shape should be (B, 1, 1, S)
    h = self.ln1(x)
    x = self.dropout(self.self_mha(h, h, h, mask)) + x
    return self.dropout(self.ffn(self.ln2(x))) + x


class DecoderBlock(nn.Module):
  def __init__(self, d_model, h, dropout):
    super().__init__()
    self.self_causal_mha = MultiHeadAttention(d_model, h, dropout)
    self.ln1 = nn.LayerNorm(d_model)
    self.cross_mha = MultiHeadAttention(d_model, h, dropout)
    self.ln2 = nn.LayerNorm(d_model)
    self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Linear(d_model * 4, d_model))
    self.ln3 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, tgt, enc_out, src_mask):
    h = self.ln1(tgt)
    x = self.dropout(self.self_causal_mha(h, h, h, None, True)) + tgt
    x = self.dropout(self.cross_mha(self.ln2(x), enc_out, enc_out, src_mask)) + x
    return self.dropout(self.ffn(self.ln3(x))) + x


def pe(d_model, max_seq):
  pos = torch.arange(0, max_seq).unsqueeze(1)
  i2 = torch.arange(0, d_model, 2)
  ang = pos * torch.exp(-i2 / d_model * math.log(10000.0))
  out = torch.zeros(max_seq, d_model)
  out[:, 0::2] = torch.sin(ang)
  out[:, 1::2] = torch.cos(ang)

  return out


class Transformer(nn.Module):
  def __init__(self, d_model, h, N, vocab_sz, max_seq, dropout):
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.max_seq = max_seq
    self.d_model = d_model
    self.embed = nn.Embedding(vocab_sz, d_model)
    self.encoder = nn.ModuleList([EncoderBlock(d_model, h, dropout) for _ in range(N)])
    self.decoder = nn.ModuleList([DecoderBlock(d_model, h, dropout) for _ in range(N)])
    self.enc_final_ln = nn.LayerNorm(d_model)
    self.dec_final_ln = nn.LayerNorm(d_model)
    self.lm_head = nn.Linear(d_model, vocab_sz, False)
    self.lm_head.weight = self.embed.weight
    self._init_weights()
    self.register_buffer('pe', pe(d_model, max_seq))

  def _init_weights(self):
    for p in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    nn.init.normal_(self.embed.weight, mean=0.0, std=self.d_model ** -0.5)

  def _embed(self, ids):
    x = self.embed(ids) * math.sqrt(self.d_model)
    x = x + self.pe[:ids.shape[-1], :]
    return self.dropout(x)

  def encode(self, src, src_mask):
    x = self._embed(src)
    for encoder_block in self.encoder:
      x = encoder_block(x, src_mask)
    return self.enc_final_ln(x)

  def decode(self, memory, src_mask, tgt):
    x = self._embed(tgt)
    for decoder_block in self.decoder:
      x = decoder_block(x, memory, src_mask)
    return self.dec_final_ln(x)

  def proj(self, x):
    return self.lm_head(x)

  def forward(self, src, src_mask, tgt):
    assert tgt.shape[-1] <= self.max_seq and src.shape[-1] <= self.max_seq
    return self.proj(self.decode(self.encode(src, src_mask), src_mask, tgt))


@torch.no_grad()
def greedy_decode(model, src, src_mask, sos, eos, pad, max_len):
  B, device = src.size(0), src.device
  memory = model.encode(src, src_mask)
  ys = torch.full((B, 1), sos, dtype=torch.long, device=device)
  finished = torch.zeros(B, dtype=torch.bool, device=device)
  for _ in range(max_len - 1):
    logits = model.proj(model.decode(memory, src_mask, ys)[:, -1])
    nxt = logits.argmax(-1).masked_fill(finished, pad)
    ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
    finished |= nxt == eos
    if bool(finished.all()):
      break
  return ys


@torch.no_grad()
def beam_search_decode(model, src, src_mask, sos, eos, max_len, beam_size=4, alpha=0.6):
  device = src.device
  memory = model.encode(src, src_mask)
  seqs = torch.full((1, 1), sos, dtype=torch.long, device=device)
  scores = torch.zeros(1, device=device)
  finished = []

  for _ in range(max_len - 1):
    n, vocab = seqs.size(0), model.lm_head.out_features
    logits = model.proj(model.decode(memory.expand(n, -1, -1),
                                      src_mask.expand(n, -1, -1, -1), seqs)[:, -1])
    logp = torch.log_softmax(logits, dim=-1)

    cand = (scores.unsqueeze(1) + logp).reshape(-1)
    scores, flat = cand.topk(min(beam_size, cand.numel()))
    seqs = torch.cat([seqs[flat // vocab], (flat % vocab).unsqueeze(1)], dim=1)

    done = seqs[:, -1] == eos
    for j in done.nonzero().flatten().tolist():
      finished.append((seqs[j], scores[j].item()))
    seqs, scores = seqs[~done], scores[~done]
    if seqs.size(0) == 0 or len(finished) >= beam_size:
      break

  finished += [(seqs[j], scores[j].item()) for j in range(seqs.size(0))]
  lp = lambda L: ((5 + L) / 6) ** alpha
  return max(finished, key=lambda t: t[1] / lp(t[0].size(0)))[0]
