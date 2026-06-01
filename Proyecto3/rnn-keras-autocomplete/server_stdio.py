#!/usr/bin/env python3
"""Autocompletado Keras; JSON por línea en stdin/stdout."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
_model = None
_stoi: dict[str, int] = {}
_itos: dict[int, str] = {}
_BLOCK_SIZE = 64

def _load() -> None:
    global _model, _stoi, _itos, _BLOCK_SIZE
    meta = json.loads((ROOT / "meta.json").read_text(encoding="utf-8"))
    _BLOCK_SIZE = int(meta["block_size"])
    chars = meta["chars"]
    _stoi = {c: i for i, c in enumerate(chars)}
    _itos = {i: c for c, i in _stoi.items()}
    _model = tf.keras.models.load_model(ROOT / "model.keras")

def _encode(s: str) -> list[int]:
    return [_stoi[c] for c in s]

def _decode(ids: list[int]) -> str:
    return "".join(_itos[i] for i in ids)

def _complete(prefix: str, max_new: int = 80, temperature: float = 0.2) -> str:
    ids = _encode(prefix)
    rng = np.random.default_rng(42)
    for _ in range(max_new):
        x = np.array(ids[-_BLOCK_SIZE:], dtype=np.int64)
        if x.shape[0] < _BLOCK_SIZE:
            pad = np.full(_BLOCK_SIZE - x.shape[0], ids[0], dtype=np.int64)
            x = np.concatenate([pad, x])
        x = x.reshape(1, _BLOCK_SIZE)
        logits = _model(x.reshape(1, _BLOCK_SIZE), training=False).numpy()[0, -1, :]
        logits = logits / max(temperature, 1e-6)
        logits = logits - logits.max()
        probs = np.exp(logits)
        probs = probs / probs.sum()
        ids.append(int(rng.choice(len(probs), p=probs)))
    return _decode(ids)

def _suggest(prefix: str, n: int = 5) -> list[str]:
    seen, out = set(), []
    for i in range(n * 3):
        text = _complete(prefix, max_new=50, temperature=0.1 + 0.05 * i)
        line = (prefix + text[len(prefix) :].split("\n")[0])[:80]
        if line not in seen and len(line) > len(prefix):
            seen.add(line)
            out.append(line)
        if len(out) >= n:
            break
    return out

def _handle(msg: dict) -> dict:
    rid = msg.get("_id")
    base = {"_id": rid} if rid is not None else {}
    try:
        if msg.get("method") == "complete":
            return {
                **base,
                "ok": True,
                "text": _complete(
                    msg.get("prefix", ""),
                    int(msg.get("max_new", 80)),
                    float(msg.get("temperature", 0.75)),
                ),
            }
        if msg.get("method") == "suggest":
            return {
                **base,
                "ok": True,
                "items": _suggest(msg.get("prefix", ""), int(msg.get("n", 5))),
            }
        return {**base, "ok": False, "error": "metodo desconocido"}
    except Exception as exc:
        return {**base, "ok": False, "error": str(exc)}

def main() -> None:
    if not (ROOT / "model.keras").is_file():
        sys.stderr.write("Falta model.keras — entrena y guarda el modelo primero.\n")
        sys.exit(1)
    _load()
    sys.stderr.write("servidor Keras listo\n")
    sys.stderr.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            out = _handle(json.loads(line))
        except json.JSONDecodeError as exc:
            out = {"ok": False, "error": f"JSON invalido: {exc}"}
        sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
