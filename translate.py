# translate.py

import sys
import textwrap

import torch
from tokenizers import Tokenizer

from config import (get_config, tokenizer_path, get_weights_file_path,
                    latest_weights_file_path, clean_output)
from model import Transformer, beam_search_decode


def load(cfg, epoch=None):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  tok = Tokenizer.from_file(str(tokenizer_path(cfg, "shared")))
  model = Transformer(cfg["d_model"], cfg["h"], cfg["N"],
                      tok.get_vocab_size(), cfg["max_seq"], cfg["dropout"]).to(device)
  ckpt = get_weights_file_path(cfg, epoch) if epoch else latest_weights_file_path(cfg)
  assert ckpt, "no checkpoint found -- train first"
  print(f"Loading {ckpt}")
  model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
  model.eval()
  return model, tok, device


@torch.no_grad()
def translate(sentence, model, tok, cfg, device):
  sos, eos = tok.token_to_id("[SOS]"), tok.token_to_id("[EOS]")
  ids = tok.encode(sentence).ids[: cfg["max_seq"] - 2]
  src = torch.tensor([[sos, *ids, eos]], device=device)
  src_mask = torch.ones(1, 1, 1, src.size(1), dtype=torch.bool, device=device)
  out = beam_search_decode(model, src, src_mask, sos, eos,
                           cfg["max_seq"], cfg["beam_size"], cfg["length_penalty"])
  out = out.tolist()[1:]
  if eos in out:
    out = out[: out.index(eos)]
  return clean_output(tok.decode(out))


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print('usage: python translate.py "sentence to translate"')
    sys.exit(1)
  cfg = get_config()
  model, tok, device = load(cfg)
  print(textwrap.fill(translate(" ".join(sys.argv[1:]), model, tok, cfg, device), 140))
