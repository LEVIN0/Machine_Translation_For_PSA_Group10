#!/usr/bin/env python3
"""Tests for training.lang_tokens — NLLB unseen-language extension.

NLLB-200 ships without guz_Latn; ensure_lang_token must add it, resize the
model embeddings, and donor-initialise the new row from swh_Latn — while
being a strict no-op for tokens that already exist. Uses stub
tokenizer/model objects so it runs offline in milliseconds.

Run from the project root:  python tests/test_week3_langtokens.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class StubTokenizer:
    """Minimal NLLB-like tokenizer: vocab dict + unk fallback."""

    def __init__(self, vocab: dict[str, int]):
        self._vocab = dict(vocab)
        self.unk_token_id = 0

    def get_vocab(self):
        return dict(self._vocab)

    def convert_tokens_to_ids(self, tok):
        return self._vocab.get(tok, self.unk_token_id)

    def add_special_tokens(self, mapping):
        for tok in mapping["additional_special_tokens"]:
            if tok not in self._vocab:
                self._vocab[tok] = len(self._vocab)
        return len(mapping["additional_special_tokens"])

    def __len__(self):
        return len(self._vocab)


class StubModel:
    """Tied-embedding seq2seq stub with NLLB-style resize semantics."""

    def __init__(self, n: int, dim: int = 4):
        import torch

        self._torch = torch
        self.emb = torch.nn.Embedding(n, dim)
        with torch.no_grad():
            self.emb.weight.copy_(torch.arange(n * dim, dtype=torch.float)
                                  .reshape(n, dim))

    def resize_token_embeddings(self, n: int):
        torch = self._torch
        old = self.emb.weight.data.clone()
        new = torch.nn.Embedding(n, old.shape[1])
        with torch.no_grad():
            new.weight[: old.shape[0]] = old
        self.emb = new

    def get_input_embeddings(self):
        return self.emb

    def get_output_embeddings(self):  # tied
        return self.emb


VOCAB = {"<unk>": 0, "eng_Latn": 1, "swh_Latn": 2}  # no guz_Latn, like NLLB


def test_existing_token_is_noop():
    from training.lang_tokens import ensure_lang_token
    tok, model = StubTokenizer(VOCAB), StubModel(len(VOCAB))
    tid = ensure_lang_token(model, tok, "swh_Latn")
    assert tid == 2, tid
    assert len(tok) == 3, "existing token must not grow the vocab"
    assert model.emb.weight.shape[0] == 3, "no resize for existing token"
    print("ok  test_existing_token_is_noop")


def test_missing_token_added_with_donor_init():
    from training.lang_tokens import ensure_lang_token
    tok, model = StubTokenizer(VOCAB), StubModel(len(VOCAB))
    donor_row = model.emb.weight[VOCAB["swh_Latn"]].clone()

    new_id = ensure_lang_token(model, tok, "guz_Latn")
    assert new_id == 3, new_id
    assert len(tok) == 4 and "guz_Latn" in tok.get_vocab()
    assert model.emb.weight.shape[0] == 4, "embeddings must be resized"
    assert model.emb.weight[new_id].equal(donor_row), \
        "new row must be donor-initialised from swh_Latn"
    # old rows untouched
    assert model.emb.weight[VOCAB["eng_Latn"]].equal(
        __import__("torch").tensor([4.0, 5.0, 6.0, 7.0]))
    print("ok  test_missing_token_added_with_donor_init")


def test_idempotent_second_call():
    from training.lang_tokens import ensure_lang_token
    tok, model = StubTokenizer(VOCAB), StubModel(len(VOCAB))
    first = ensure_lang_token(model, tok, "guz_Latn")
    second = ensure_lang_token(model, tok, "guz_Latn")
    assert first == second
    assert len(tok) == 4, "second call must be a no-op"
    print("ok  test_idempotent_second_call")


def test_missing_donor_still_adds_token():
    from training.lang_tokens import ensure_lang_token
    tok, model = StubTokenizer({"<unk>": 0}), StubModel(1)
    new_id = ensure_lang_token(model, tok, "guz_Latn")
    assert isinstance(new_id, int) and new_id > 0
    assert "guz_Latn" in tok.get_vocab()
    print("ok  test_missing_donor_still_adds_token")


def run() -> int:
    test_existing_token_is_noop()
    test_missing_token_added_with_donor_init()
    test_idempotent_second_call()
    test_missing_donor_still_adds_token()
    print("ok  test_week3_langtokens")
    return 0


if __name__ == "__main__":
    sys.exit(run())
