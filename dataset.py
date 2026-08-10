# dataset.py

import random
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler


class TranslationDataset(Dataset):
  def __init__(self, tokenizer, raw_ds, src_lng, tgt_lng, max_seq):
    super().__init__()
    self.pad_id, self.sos_id, self.eos_id = tokenizer.token_to_id('[PAD]'), tokenizer.token_to_id('[SOS]'), tokenizer.token_to_id('[EOS]')
    self.samples, self.lengths = [], []
    self.dropped_long = self.dropped_junk = 0
    for item in raw_ds:
      src_text = item['translation'][src_lng].strip()
      tgt_text = item['translation'][tgt_lng].strip()

      if not src_text or not tgt_text:
        self.dropped_junk += 1; continue

      if src_text == tgt_text:
        self.dropped_junk += 1; continue

      src_ids = tokenizer.encode(src_text).ids
      tgt_ids = tokenizer.encode(tgt_text).ids
      max_length = max(len(src_ids) + 2, len(tgt_ids) + 1)
      if max_length >= max_seq :
        self.dropped_long += 1; continue

      if len(src_ids) < 1 or len(tgt_ids) < 1:
        self.dropped_junk += 1; continue

      r = max(len(src_ids), len(tgt_ids)) / min(len(src_ids), len(tgt_ids))
      if r > 2.5:
        self.dropped_junk += 1; continue

      self.samples.append((src_text, tgt_text, src_ids, tgt_ids))
      self.lengths.append(max(len(src_ids) + 2, len(tgt_ids) + 1))
    print(f"junk_items_dropped: {self.dropped_junk} ; long_items_dropped: {self.dropped_long}")

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    src_text, tgt_text, src_ids, tgt_ids = self.samples[idx]
    return {
        "enc_input": torch.tensor([self.sos_id, *src_ids, self.eos_id], dtype = torch.long),
        "dec_input": torch.tensor([self.sos_id, *tgt_ids], dtype = torch.long),
        "label": torch.tensor([*tgt_ids, self.eos_id], dtype = torch.long),
        "src_txt": src_text, "tgt_txt": tgt_text
    }


def make_collate_fn(pad_id):
  def collate_fn(batch):
    enc = pad_sequence([b["enc_input"] for b in batch], batch_first = True, padding_value = pad_id)
    dec = pad_sequence([b["dec_input"] for b in batch], batch_first = True, padding_value = pad_id)
    lbl = pad_sequence([b["label"] for b in batch], batch_first = True, padding_value = pad_id)
    enc_mask = (enc != pad_id).unsqueeze(1).unsqueeze(1)

    return {"enc_input": enc, "dec_input": dec, "label": lbl,
            "enc_mask": enc_mask,
            "src_txt": [b["src_txt"] for b in batch],
            "tgt_txt": [b["tgt_txt"] for b in batch]
            }

  return collate_fn


class LengthBatchSampler(Sampler):
  def __init__(self, lengths, batch_size, shuffle = True, mega_factor = 50):
    self.lengths = lengths
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.mega = batch_size * mega_factor

  def __len__(self):
    return (len(self.lengths) + self.batch_size - 1) // self.batch_size

  def __iter__(self):
    idx = list(range(len(self.lengths)))
    if self.shuffle:
      random.shuffle(idx)
    batches = []
    for i in range(0, len(idx), self.mega):
      chunk = sorted(idx[i:i + self.mega], key=lambda j: self.lengths[j])
      batches += [chunk[k:k + self.batch_size] for k in range(0, len(chunk), self.batch_size)]
    if self.shuffle:
      random.shuffle(batches)
    yield from batches


def _ds_from_blob(tok, samples, lengths):
  ds = TranslationDataset.__new__(TranslationDataset)
  ds.pad_id = tok.token_to_id('[PAD]')
  ds.sos_id = tok.token_to_id('[SOS]')
  ds.eos_id = tok.token_to_id('[EOS]')
  ds.samples, ds.lengths = samples, lengths
  ds.dropped_long = ds.dropped_junk = 0
  return ds
