"""
ReactionT5v2 integration (sagawatatsuya/ReactionT5v2)
HuggingFace models: sagawa/ReactionT5v2-{retrosynthesis,yield,forward}

Three capabilities used in Flejabosaur:

1. Retrosynthesis  — product SMILES → top-N precursor SMILES (beam search)
   Replaces SMARTS template matching as the primary fallback after AiZynthFinder.

2. Yield prediction — (reactants, product) → numeric yield [0–100]
   Used in scoring.py to replace k-NN yield averaging.

3. Forward validation — (reactants, reagents) → predicted product SMILES
   Used to verify that candidate precursors actually yield the target molecule.

Input formats (from ReactionT5v2 source/README):
  Retrosynthesis: "PRODUCT:{product_smiles}"  → decoded reactant SMILES
  Yield:          "REACTANT:{r1.r2}PRODUCT:{product}"  → numeric string e.g. "85.3"
  Forward:        "REACTANT:{r1.r2}REAGENT:{reagent}PRODUCT:"  → completed product SMILES

All models are loaded lazily (first call) to avoid startup overhead.
Falls back to HuggingFace Inference API if local loading fails (set HF_TOKEN env var).
"""
from __future__ import annotations

import logging
import os
import re
from functools import lru_cache

from rdkit import Chem  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)

MODEL_IDS = {
    "retrosynthesis": "sagawa/ReactionT5v2-retrosynthesis",
    "yield": "sagawa/ReactionT5v2-yield",
    "forward": "sagawa/ReactionT5v2-forward",
}

# Module-level model/tokenizer singletons
_models: dict[str, object] = {}
_tokenizers: dict[str, object] = {}


def _canonicalize(smiles: str) -> str:
    """Canonicalize SMILES; return original if RDKit fails."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol)
    except Exception:
        pass
    return smiles


def _load_model(task: str) -> tuple:
    """Lazy-load model and tokenizer for a given task. Returns (model, tokenizer)."""
    if task in _models:
        return _models[task], _tokenizers[task]

    model_id = MODEL_IDS[task]
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # type: ignore[import-not-found]
        logger.info("Loading ReactionT5v2-%s from HuggingFace...", task)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        model.eval()
        _models[task] = model
        _tokenizers[task] = tokenizer
        logger.info("ReactionT5v2-%s loaded.", task)
        return model, tokenizer
    except Exception as exc:
        logger.warning("Could not load ReactionT5v2-%s locally: %s", task, exc)
        _models[task] = None
        _tokenizers[task] = None
        return None, None


def _hf_inference_api(task: str, input_text: str) -> str | None:
    """
    Fallback: call HuggingFace Inference API.
    Requires HF_TOKEN environment variable.
    """
    import httpx  # type: ignore[import-not-found]
    token = os.environ.get("HF_TOKEN")
    if not token:
        return None
    model_id = MODEL_IDS[task]
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    try:
        resp = httpx.post(
            url,
            headers={"Authorization": f"Bearer {token}"},
            json={"inputs": input_text},
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and data:
                return data[0].get("generated_text", "")
    except Exception as exc:
        logger.warning("HF Inference API call failed for %s: %s", task, exc)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrosynthesis(product_smiles: str, num_beams: int = 5, num_return: int = 5) -> list[str]:
    """
    Predict precursor SMILES for a given product using ReactionT5v2-retrosynthesis.

    Returns a list of candidate reactant SMILES strings (may contain "." separating
    multiple reactants in a single candidate). Returns empty list on failure.

    Input format to the model: "PRODUCT:{product_smiles}"
    Output: decoded SMILES of reactants (beam search, highest score first)
    """
    product_smiles = _canonicalize(product_smiles)
    input_text = f"PRODUCT:{product_smiles}"

    model, tokenizer = _load_model("retrosynthesis")

    if model is not None and tokenizer is not None:
        return _seq2seq_generate(model, tokenizer, input_text, num_beams, num_return)

    # Fallback: HF Inference API (returns single best prediction)
    result = _hf_inference_api("retrosynthesis", input_text)
    if result:
        return [result.strip()]
    return []


def predict_yield(reactant_smiles: list[str], product_smiles: str) -> float | None:
    """
    Predict reaction yield using ReactionT5v2-yield.

    Input format: "REACTANT:{r1.r2}PRODUCT:{product}"
    Output: float 0.0–1.0 (model returns 0–100, we normalize)

    Returns None if the model is unavailable.
    """
    reactants_str = ".".join(_canonicalize(r) for r in reactant_smiles if r)
    product_str = _canonicalize(product_smiles)
    input_text = f"REACTANT:{reactants_str}PRODUCT:{product_str}"

    model, tokenizer = _load_model("yield")

    raw: str | None = None
    if model is not None and tokenizer is not None:
        results = _seq2seq_generate(model, tokenizer, input_text, num_beams=1, num_return=1)
        raw = results[0] if results else None
    else:
        raw = _hf_inference_api("yield", input_text)

    if raw is None:
        return None

    return _parse_yield(raw)


def forward_predict(
    reactant_smiles: list[str],
    reagent_smiles: list[str] | None = None,
    num_beams: int = 5,
    num_return: int = 5,
) -> list[str]:
    """
    Predict products from reactants using ReactionT5v2-forward.

    Input format: "REACTANT:{r1.r2}REAGENT:{reagent}PRODUCT:"
    Output: list of predicted product SMILES (beam search)

    Used to validate retrosynthetic candidates by checking if forward prediction
    recovers the target molecule.
    """
    reactants_str = ".".join(_canonicalize(r) for r in reactant_smiles if r)
    reagents_str = ".".join(_canonicalize(r) for r in (reagent_smiles or []) if r)
    input_text = f"REACTANT:{reactants_str}REAGENT:{reagents_str}PRODUCT:"

    model, tokenizer = _load_model("forward")
    if model is not None and tokenizer is not None:
        return _seq2seq_generate(model, tokenizer, input_text, num_beams, num_return)

    result = _hf_inference_api("forward", input_text)
    if result:
        return [result.strip()]
    return []


def forward_validates_target(
    reactant_smiles: list[str],
    target_smiles: str,
    reagent_smiles: list[str] | None = None,
) -> bool | None:
    """
    Returns True if forward prediction of reactants recovers the target molecule.
    Returns None if the model is unavailable (so callers can treat as neutral).
    Returns False if the model ran but the target was not among predicted products.

    Checks the first (major) product of each beam candidate, since minor
    byproducts are less relevant for validation.
    """
    target_canonical = _canonicalize(target_smiles)
    predicted_products = forward_predict(reactant_smiles, reagent_smiles, num_beams=5, num_return=5)

    if not predicted_products:
        return None  # model unavailable or failed

    for pred in predicted_products:
        # Check the first (major) component — byproducts are after "."
        major_product = pred.split(".")[0].strip()
        if _canonicalize(major_product) == target_canonical:
            return True
    return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _seq2seq_generate(model, tokenizer, input_text: str, num_beams: int, num_return: int) -> list[str]:
    """Run T5 seq2seq generation and decode output tokens."""
    import torch  # type: ignore[import-not-found]
    try:
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                num_beams=num_beams,
                num_return_sequences=num_return,
                max_new_tokens=256,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # Filter empty and deduplicate while preserving order
        seen: set[str] = set()
        results = []
        for s in decoded:
            s = s.strip()
            if s and s not in seen:
                seen.add(s)
                results.append(s)
        return results
    except Exception as exc:
        logger.warning("ReactionT5v2 generation failed: %s", exc)
        return []


def _parse_yield(raw: str) -> float | None:
    """Parse a yield prediction string to a 0.0–1.0 float."""
    raw = raw.strip()

    def _normalize(val: float) -> float:
        if val > 100.0:
            logger.warning("T5 yield prediction > 100%% (%s); clamping. Raw: %r", val, raw)
        if val < 0:
            logger.warning("T5 yield prediction < 0 (%s); clamping. Raw: %r", val, raw)
        if val > 1.0:
            val /= 100.0
        return max(0.0, min(1.0, val))

    # Try direct float parse
    try:
        return _normalize(float(raw))
    except ValueError:
        pass
    # Extract first number from string
    match = re.search(r"[\d.]+", raw)
    if match:
        try:
            return _normalize(float(match.group()))
        except ValueError:
            pass
    logger.warning("Could not parse yield from T5 output: %r", raw)
    return None
