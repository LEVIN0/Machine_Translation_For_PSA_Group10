"""NLLB unseen-language extension (SPEC_WEEK3.md §3 addendum).

NLLB-200 ships without a ``guz_Latn`` language token (verified against the
published tokenizer), so out of the box an Ekegusii target/source silently
maps to <unk>. This module adds the missing language token to the tokenizer,
resizes the model embeddings, and initialises the new embedding row from a
donor language (Swahili — a close Bantu relative) so fine-tuning starts
from a sensible point instead of random noise.

No heavy imports at module level; torch is imported lazily inside the
donor-initialisation block.
"""

from __future__ import annotations


def ensure_lang_token(model, tokenizer, lang_code: str,
                      donor: str = "swh_Latn") -> int:
    """Guarantee ``lang_code`` maps to a dedicated token id; return that id.

    No-op when the token already exists (e.g. Swahili/English codes, or when
    loading a fine-tuned checkpoint whose tokenizer was saved with the added
    token). When missing: adds it as an additional special token, resizes the
    model's token embeddings, and copies the donor language's embedding row
    into the new row (input embeddings, and output embeddings too when they
    are untied).
    """
    tid = tokenizer.convert_tokens_to_ids(lang_code)
    if (isinstance(tid, int) and tid >= 0
            and tid != getattr(tokenizer, "unk_token_id", None)
            and lang_code in tokenizer.get_vocab()):
        return tid

    old_len = len(tokenizer)
    tokenizer.add_special_tokens({"additional_special_tokens": [lang_code]})
    model.resize_token_embeddings(len(tokenizer))
    new_id = tokenizer.convert_tokens_to_ids(lang_code)

    try:
        donor_id = (tokenizer.convert_tokens_to_ids(donor)
                    if donor in tokenizer.get_vocab() else None)
        if isinstance(donor_id, int) and 0 <= donor_id < old_len:
            import torch

            with torch.no_grad():
                in_emb = model.get_input_embeddings().weight
                in_emb[new_id] = in_emb[donor_id]
                out_emb = model.get_output_embeddings()
                if (out_emb is not None
                        and out_emb.weight.data_ptr() != in_emb.data_ptr()):
                    out_emb.weight[new_id] = out_emb.weight[donor_id]
    except Exception as exc:  # donor init is best-effort; random init is fine
        print(f"[lang_tokens] donor initialisation skipped ({exc})")

    print(f"[lang_tokens] added missing NLLB language token '{lang_code}' "
          f"(id {new_id}; embedding initialised from '{donor}')")
    return new_id
