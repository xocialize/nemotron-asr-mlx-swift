# nemotron-asr-mlx-swift

![platform](https://img.shields.io/badge/platform-macOS%2014%2B%20%C2%B7%20Apple%20Silicon-black)
![swift](https://img.shields.io/badge/Swift-5.9-orange)
![port code](https://img.shields.io/badge/port%20code-MIT-blue)
![weights](https://img.shields.io/badge/weights-OpenMDW--1.1-green)

MLX-Swift port of **NVIDIA Nemotron 3.5 ASR streaming** (`nvidia/nemotron-3.5-asr-streaming-0.6b`):
a 600M cache-aware FastConformer-RNNT with language-ID prompt conditioning, transcribing
40 language-locales with native punctuation/capitalization.

Parity-locked against the Python `Blaizzy/mlx-audio` reference — every encoder sub-op
(subsampling, all 24 cache-aware conformer blocks with cold and warm caches, prompt
conditioning, decoder + joint) matches **bit-for-bit** on the CPU stream, and the end-to-end
transcript (token ids + timestamps + auto-detected locale) is exact. ~27× realtime on the GPU.

- **Reference:** Blaizzy/mlx-audio `mlx_audio/stt/models/nemotron_asr/` (MIT). File and
  module names mirror the reference 1:1 for parity diffing.
- **Weights:** [`mlx-community/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/mlx-community/nemotron-3.5-asr-streaming-0.6b)
  (bf16, ~1.2 GB; an `-8bit` variant exists). Weight license: OpenMDW-1.1 (permissive).
- **Engine kit:** `mlx-nemotron-stt-swift` wraps this core as the MLXEngine `stt` capability
  package (`NemotronSTTPackage`). This core stays engine-free (no MLXToolKit import).

## Install

Add the package to your `Package.swift`:

```swift
.package(url: "https://github.com/xocialize/nemotron-asr-mlx-swift.git", from: "0.2.0"),
```

and the product to your target:

```swift
.product(name: "NemotronASR", package: "nemotron-asr-mlx-swift"),
```

> Building an app on the engine? Depend on the
> [`mlx-nemotron-stt-swift`](https://github.com/xocialize/mlx-nemotron-stt-swift) kit instead —
> it wraps this core as the MLXEngine `stt` capability and handles weight download + lifecycle.
> Use this core directly only when you want the model without the engine.

## Quick start

```swift
import NemotronASR

let model = try NemotronLoad.load(directory: modelDir)  // config.json + model.safetensors
let audio: MLXArray = ...                               // 16 kHz mono Float32 waveform
let t = model.streamGenerate(audio)                     // language: nil = auto-detect
print(t.result.text, t.detectedLanguage ?? "?")
```

Latency modes ride `attContextSize`: `[56, 13]` (default, 1.12 s chunks, best accuracy)
down to `[56, 0]` (80 ms chunks, lowest latency).

### Live audio

`streamGenerate` needs the whole waveform up front. When audio is still arriving — a
microphone, a socket — drive a `NemotronStreamingSession` instead: push samples as they
land, read cumulative partials, then `finish`.

```swift
let session = NemotronStreamingSession(model: model, attContextSize: [56, 3])
for buffer in micBuffers {                     // any size; 16 kHz mono Float32
    session.push(buffer) { partial in
        print(partial.text)                    // cumulative, absolute timestamps
    }
}
let final = session.finish()
```

The session buffers to the model's native chunk internally, so push sizes are free. It is
the same code path `streamGenerate` runs on, and `stream-gate` holds the two to *exact*
equality — ragged live pushes produce byte-identical tokens and timestamps to the
whole-buffer pass. Not thread-safe: serialize access (the engine kit drives it on its
inference actor, and MLX work must never run on an audio callback thread).

### Scoring transcripts

`ASRMetrics` is a separate, dependency-free product (no MLX) for grading a hypothesis
against a reference — ASR gates, or TTS round-trips that synthesize, transcribe, and score
against the input text.

```swift
import ASRMetrics

let r = ASRMetrics.wordErrorRate(reference: expected, hypothesis: transcript)
print(r.werPercent, r.substitutions, r.deletions, r.insertions)
for op in r.ops { ... }   // word-level alignment, for color-coded display
```

## CLI

```bash
swift run nemotron-gate transcribe path/to/16k-mono.wav
swift run nemotron-gate key-gate            # weight key contract
swift run nemotron-gate mel-gate            # audio front-end vs golden
swift run nemotron-gate component-gate      # per-layer stream blocks, prompt, decoder/joint
swift run nemotron-gate e2e-gate            # full transcript parity vs Python (CPU, fp32)
swift run nemotron-gate gpu-smoke           # GPU-stream run + RTF
swift run nemotron-gate stream-gate         # live session (ragged pushes) == batch, exactly
swift run nemotron-gate wer "reference" "hypothesis"
swift run -c release nemotron-gate creep --model-dir $M --wav long.wav   # memory vs session length
```

Gates compare against Python-oracle goldens in `PORTING/goldens/`
(regenerate with `PORTING/oracle/capture_goldens.py`).

`creep` is a diagnostic, not a gate. It streams a wav through one live session in 100 ms pushes
and prints, every minute of audio and between pushes, phys_footprint beside MLX's live arrays,
its pool and the malloc heap. On a 45-minute English file at Dictate's `[56, 3]` (bf16, M5 Max,
2026-09-23), live arrays held at 1.325 GB, the pool under 0.06 GB, and the heap at 0.182–0.183 GB
(the growing transcript is the only thing in it). phys settled at ~2.2 GB after the warm-up. So
the session needs no per-step `autoreleasepool`, unlike R2T2's loop (qwen3-asr-mlx-swift 0.1.3).
MLX drains its own Metal objects inside each eval, and no chunk autoreleases Foundation objects in
volume: the language-tag regex leaves about 0.2 KB per minute.

## Architecture notes (port-critical)

- Streaming caches per conformer layer: attention cache = last 56 **normalized
  pre-attention** frames (`(B, 56, 1024)`); conv cache = last 8 **post-GLU** frames
  (`(B, 8, 1024)`, zeros on first chunk). Subsampling threads a 16-frame mel cache.
- The chunked mel front-end is full-audio-equivalent (same frame centers/preemph state);
  never materialize the full STFT or a full-length attention mask (upstream OOM #816/#817).
- RNN-T greedy decode: `max_symbols=10` per frame; decoder hidden advances only on
  non-blank; language tags are special tokens captured for `detectedLanguage`, never
  emitted into the transcript.
- The slaney 128-mel filterbank is a baked resource (generated by the oracle), per the
  mlx-audio-dsp "bake fixed transforms" rule.
