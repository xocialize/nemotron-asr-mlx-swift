"""Capture parity goldens for the Swift port by instrumenting the REAL mlx-audio
code paths (monkeypatch, no reimplementation drift). CPU-pinned, fp32 activations,
default latency mode [56, 13] on the fixed English wav.

Outputs (PORTING/goldens/):
  mel_full.safetensors        offline log-mel of the whole wav (mel gate)
  pre_encode.safetensors      subsampling in/out for inner chunks 0..2
  stream_blocks.safetensors   per-layer _stream_block io for inner chunks 0 and 2
  prompt.safetensors          apply_prompt in/out for inner chunks 0..2
  decoder_joint.safetensors   first N decoder/joint step ios + LSTM states
  pos_emb.safetensors         pos_emb_for(70) PE-table slice
  transcript.json             final tokens/timestamps/text + config snapshot
  meta.json                   versions, shapes summary, chunking parameters
"""

import dataclasses
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_audio.stt.utils import load
from mlx_audio.stt.models.nemotron_asr import nemotron_asr as na_mod
from mlx_audio.stt.models.nemotron_asr import streaming as st_mod
from mlx_audio.stt.models.nemotron_asr.audio import log_mel_spectrogram

HERE = Path(__file__).resolve().parent
AUDIO = HERE.parent / "audio" / "en_16k.wav"
GOLDENS = HERE.parent / "goldens"
REPO = "mlx-community/nemotron-3.5-asr-streaming-0.6b"
ACS = [56, 13]
CAPTURE_ENC_CHUNKS = (0, 1, 2)   # pre_encode/prompt captured for these
CAPTURE_BLOCK_CHUNKS = (0, 2)    # per-layer block io captured for these (cold + warm)
MAX_DECODE_STEPS = 60


def f32(a):
    return a.astype(mx.float32)


def save(path: Path, tensors: dict):
    mx.eval(*tensors.values())  # materialize BEFORE saving (lazy-zeros trap)
    mx.save_safetensors(str(path), tensors)
    print(f"  wrote {path.name}: {len(tensors)} tensors")


def main():
    mx.set_default_device(mx.cpu)  # parity doctrine: CPU stream for goldens
    GOLDENS.mkdir(exist_ok=True)

    print(f"loading {REPO} (cpu) ...")
    model = load(REPO)
    mx.eval(model.parameters())
    n_layers = len(model.encoder.layers)

    # ---- offline full mel (mel gate) --------------------------------------
    from mlx_audio.stt.utils import load_audio
    wav = load_audio(str(AUDIO), model.preprocessor_config.sample_rate, dtype=mx.float32)
    mel_full = log_mel_spectrogram(wav, model.preprocessor_config)
    save(GOLDENS / "mel_full.safetensors", {"wav": wav, "mel_full": f32(mel_full)})

    # ---- PE table slice ----------------------------------------------------
    pe70 = model.encoder.pos_enc.pos_emb_for(70, mx.float32)
    save(GOLDENS / "pos_emb.safetensors", {"pos_emb_70": f32(pe70)})

    # ---- instrument the real streaming path -------------------------------
    pre_encode_cap, block_cap, prompt_cap, dec_cap, joint_cap = {}, {}, {}, {}, {}

    # NOTE: dunder methods resolve on the CLASS, so all __call__ patches below are
    # class-level (instance-attribute patching silently captures nothing).
    from mlx_audio.stt.models.nemotron_asr.conformer import CausalDwStridingSubsampling
    from mlx_audio.stt.models.nemotron_asr.rnnt import JointNetwork, PredictNetwork

    real_pre_encode = CausalDwStridingSubsampling.__call__
    pe_calls = {"n": 0}

    def wrapped_pre_encode(self, x, lengths):
        i = pe_calls["n"]
        pe_calls["n"] += 1
        out, out_len = real_pre_encode(self, x, lengths)
        if i in CAPTURE_ENC_CHUNKS:
            pre_encode_cap[f"c{i}_win"] = f32(x)
            pre_encode_cap[f"c{i}_sub"] = f32(out)
        return out, out_len

    real_block = st_mod._stream_block
    blk_calls = {"n": 0}

    def wrapped_block(block, x, pos_enc, attn_cache, conv_cache, left_cache, conv_left):
        i = blk_calls["n"]
        blk_calls["n"] += 1
        chunk, layer = divmod(i, n_layers)
        h, attn_next, conv_next = real_block(
            block, x, pos_enc, attn_cache, conv_cache, left_cache, conv_left
        )
        if chunk in CAPTURE_BLOCK_CHUNKS:
            p = f"c{chunk}_l{layer:02d}"
            block_cap[f"{p}_x"] = f32(x)
            if attn_cache is not None:
                block_cap[f"{p}_attn_cache_in"] = f32(attn_cache)
            if conv_cache is not None:
                block_cap[f"{p}_conv_cache_in"] = f32(conv_cache)
            block_cap[f"{p}_h"] = f32(h)
            block_cap[f"{p}_attn_next"] = f32(attn_next)
            block_cap[f"{p}_conv_next"] = f32(conv_next)
        return h, attn_next, conv_next

    real_prompt = model.apply_prompt
    pr_calls = {"n": 0}

    def wrapped_prompt(encoded, language):
        i = pr_calls["n"]
        pr_calls["n"] += 1
        out = real_prompt(encoded, language)
        if i in CAPTURE_ENC_CHUNKS:
            prompt_cap[f"c{i}_in"] = f32(encoded)
            prompt_cap[f"c{i}_out"] = f32(out)
        return out

    real_decoder = PredictNetwork.__call__
    dec_calls = {"n": 0}

    def wrapped_decoder(self, y, h_c=None):
        i = dec_calls["n"]
        dec_calls["n"] += 1
        out, (h, c) = real_decoder(self, y, h_c)
        if i < MAX_DECODE_STEPS:
            dec_cap[f"s{i:03d}_token"] = (
                mx.array([-1], dtype=mx.int32) if y is None else y.reshape(-1).astype(mx.int32)
            )
            dec_cap[f"s{i:03d}_out"] = f32(out)
            if i < 10:
                dec_cap[f"s{i:03d}_h"] = f32(h)
                dec_cap[f"s{i:03d}_c"] = f32(c)
        return out, (h, c)

    real_joint = JointNetwork.__call__
    jt_calls = {"n": 0}

    def wrapped_joint(self, enc, pred):
        i = jt_calls["n"]
        jt_calls["n"] += 1
        out = real_joint(self, enc, pred)
        if i < MAX_DECODE_STEPS:
            joint_cap[f"s{i:03d}_enc"] = f32(enc)
            joint_cap[f"s{i:03d}_pred"] = f32(pred)
            joint_cap[f"s{i:03d}_logits"] = f32(out)
        return out

    CausalDwStridingSubsampling.__call__ = wrapped_pre_encode
    st_mod._stream_block = wrapped_block
    model.apply_prompt = wrapped_prompt
    PredictNetwork.__call__ = wrapped_decoder
    JointNetwork.__call__ = wrapped_joint

    # ---- run the REAL streaming transcription ------------------------------
    print(f"streaming {AUDIO.name} acs={ACS} (cpu, fp32) ...")
    t0 = time.perf_counter()
    result = None
    n_chunks = 0
    for result in model.stream_generate(str(AUDIO), att_context_size=ACS):
        n_chunks += 1
    dt = time.perf_counter() - t0
    assert result is not None
    print(f"  {n_chunks} decode chunks in {dt:.1f}s")
    print(f"  transcript: {result.text}")

    save(GOLDENS / "pre_encode.safetensors", pre_encode_cap)
    save(GOLDENS / "stream_blocks.safetensors", block_cap)
    save(GOLDENS / "prompt.safetensors", prompt_cap)
    save(GOLDENS / "decoder_joint.safetensors", {**dec_cap, **joint_cap})

    # ---- transcript + config snapshot --------------------------------------
    tokens = [
        {"id": t.id, "text": t.text, "start": round(t.start, 6), "duration": round(t.duration, 6)}
        for s in result.sentences
        for t in s.tokens
    ]
    cfg = dataclasses.asdict(model.config)
    (GOLDENS / "transcript.json").write_text(json.dumps({
        "text": result.text,
        "tokens": tokens,
        "att_context_size": ACS,
        "language": "auto",
    }, ensure_ascii=False, indent=1))
    (GOLDENS / "config_snapshot.json").write_text(json.dumps(cfg, ensure_ascii=False))

    import importlib.metadata as md
    (GOLDENS / "meta.json").write_text(json.dumps({
        "weights_repo": REPO,
        "wav": AUDIO.name,
        "acs": ACS,
        "device": "cpu",
        "activation_dtype": "float32",
        "mlx": mx.__version__,
        "mlx_audio": md.version("mlx-audio"),
        "n_inner_encode_chunks": pe_calls["n"],
        "n_decode_steps_total": dec_calls["n"],
        "n_layers": n_layers,
        "captured_block_chunks": list(CAPTURE_BLOCK_CHUNKS),
        "captured_enc_chunks": list(CAPTURE_ENC_CHUNKS),
    }, indent=1))
    print("goldens complete.")


if __name__ == "__main__":
    main()
