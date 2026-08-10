# config.py

from pathlib import Path
import re as _re


def get_config():
  return {
      "datasource": "wmt/wmt14",
      "dataset_config": "fr-en",
      "lang_src": "en",
      "lang_tgt": "fr",
      "model_folder": "weights",
      "model_basename": "tmodel_",
      "tokenizer_file": "tokenizer_{0}.json",
      "shuffle_seed": 0,
      "vocab_size": 32000,
      "max_seq": 300,
      "num_workers": 8,
      "prefetch_factor": 4,
      "batch_size": 256,
      "beam_size": 4,
      "length_penalty": 0.6,
      "num_validation_examples": 3,
      "validation_size": 3000,
      "d_model": 512,
      "h": 8,
      "N": 6,
      "dropout": 0.0,
      "betas": (0.9, 0.98),
      "eps": 1e-9,
      "warmup_steps": 1500,
      "lr_scale": 1.0,
      "preload": "latest",
      "use_compile": True,
      "label_smoothing": 0.1,
      "num_epochs": 1,
      "decode_strategy": "beam",
      "num_pairs": 4_040_000,
      "val_loss_every_pairs": 100_000,
      "bleu_every_pairs": 5_000_000,
      "cache_dir": "./mt_cache",
      "seed": 0,
  }


def weights_folder(config):
  return f"{config['datasource'].replace('/', '_')}_{config['model_folder']}"


def get_weights_file_path(config, epoch: str):
  return str(Path(".") / weights_folder(config) / f"{config['model_basename']}{epoch}.pt")


def latest_weights_file_path(config):
  files = sorted(Path(weights_folder(config)).glob(f"{config['model_basename']}*"))
  return str(files[-1]) if files else None


def tokenizer_path(config, name: str) -> Path:
  slug = config["datasource"].split("/")[-1]
  cfg_slug = config["dataset_config"]
  d = Path(config.get("cache_dir", "."))
  d.mkdir(parents=True, exist_ok=True)
  return d / config["tokenizer_file"].format(f"{slug}_{cfg_slug}_s{config['shuffle_seed']}_{name}")


def build_tokenizer(vocab_size, special_tokens):
    from tokenizers import Tokenizer, decoders, pre_tokenizers
    from tokenizers.models import WordPiece
    from tokenizers.trainers import WordPieceTrainer

    tok = Tokenizer(WordPiece(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit(),
        pre_tokenizers.Split(pattern="'", behavior="merged_with_previous"),
    ])
    tok.decoder = decoders.WordPiece(prefix="##", cleanup=True)
    trainer = WordPieceTrainer(vocab_size=vocab_size, min_frequency=2,
                               special_tokens=special_tokens, show_progress=False)
    return tok, trainer


_APOS = _re.compile(r"\s*'\s*")
_HYPH = _re.compile(r"\s+-\s+")
_PUNC = _re.compile(r"\s*([.,;:!?])")
_MULTI = _re.compile(r"\s+")


def clean_output(text: str) -> str:
  text = _APOS.sub("'", text)
  text = _HYPH.sub("-", text)
  text = _PUNC.sub(r"\1", text)
  return _MULTI.sub(" ", text).strip()
