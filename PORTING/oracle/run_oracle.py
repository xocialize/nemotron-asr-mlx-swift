"""Phase-0 oracle spike: transcript quality + per-chunk latency on this machine.

Runs mlx-audio's Nemotron 3.5 ASR streaming path on the fixed test wavs at two
latency modes, prints transcripts and per-chunk wall-clock, and dumps the weight
key inventory for the Swift key-contract gate.
"""

import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_audio.stt.utils import load

HERE = Path(__file__).resolve().parent
AUDIO = HERE.parent / "audio"
GOLDENS = HERE.parent / "goldens"
REPO = "mlx-community/nemotron-3.5-asr-streaming-0.6b"


def tree_flatten_keys(model):
    from mlx.utils import tree_flatten

    return {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in tree_flatten(model.parameters())}


def run(model, wav: Path, att_context_size, language=None):
    print(f"\n=== {wav.name}  acs={att_context_size}  language={language or 'auto'} ===")
    chunk_times = []
    t_prev = time.perf_counter()
    result = None
    for result in model.stream_generate(
        str(wav), language=language, att_context_size=att_context_size
    ):
        now = time.perf_counter()
        chunk_times.append(now - t_prev)
        t_prev = now
    if result is None:
        print("  (no output)")
        return
    total = sum(chunk_times)
    print(f"  transcript: {result.text}")
    print(f"  chunks: {len(chunk_times)}  total: {total:.3f}s  "
          f"mean/chunk: {total / max(len(chunk_times), 1) * 1000:.1f}ms  "
          f"max/chunk: {max(chunk_times) * 1000:.1f}ms")
    return result


def main():
    print(f"loading {REPO} ...")
    t0 = time.perf_counter()
    model = load(REPO)
    mx.eval(model.parameters())
    print(f"loaded in {time.perf_counter() - t0:.1f}s")
    print(f"default_att_context_size: {model.default_att_context_size}")
    print(f"default_language: {model.default_language}")
    print(f"vocab: {len(model.vocabulary)}  blank_id: {model.blank_id}  "
          f"max_symbols: {model.max_symbols}")
    print(f"prompt keys ({len(model.prompt_dictionary)}): "
          f"{sorted(model.prompt_dictionary)[:8]} ...")

    GOLDENS.mkdir(exist_ok=True)
    keys = tree_flatten_keys(model)
    (GOLDENS / "weight_keys.json").write_text(json.dumps(keys, indent=1, sort_keys=True))
    print(f"weight key inventory: {len(keys)} tensors -> goldens/weight_keys.json")

    for acs in ([56, 13], [56, 3]):
        run(model, AUDIO / "en_16k.wav", acs)
    r = run(model, AUDIO / "de_16k.wav", [56, 13])
    if r is not None and r.sentences:
        first = r.sentences[0].tokens[0]
        print(f"  first token: {first.text!r} @ {first.start:.2f}s")


if __name__ == "__main__":
    main()
