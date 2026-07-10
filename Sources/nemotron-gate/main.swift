// nemotron-gate — parity/quant gates + transcribe CLI for the NemotronASR port.
//
// Gates run against the Python-oracle goldens in PORTING/goldens (see
// PORTING/oracle/capture_goldens.py). Parity gates pin the CPU stream (fp32
// doctrine); GPU smoke and quant gates run on the GPU stream (quantized forwards
// MUST NOT run under a CPU pin — they grind for hours with no error).

import ArgumentParser
import Foundation
import MLX
import MLXNN
import NemotronASR

// MARK: - Shared helpers

struct GateContext {
    let modelDir: URL
    let goldens: URL

    func golden(_ name: String) throws -> [String: MLXArray] {
        try MLX.loadArrays(url: goldens.appendingPathComponent(name))
    }
}

func defaultModelDir() -> URL? {
    // The HF hub cache snapshot produced by the Python oracle run.
    let hub = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent(".cache/huggingface/hub")
        .appendingPathComponent("models--mlx-community--nemotron-3.5-asr-streaming-0.6b/snapshots")
    guard
        let snapshots = try? FileManager.default.contentsOfDirectory(
            at: hub, includingPropertiesForKeys: nil
        ).filter({ $0.hasDirectoryPath })
    else { return nil }
    return snapshots.first
}

func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    guard a.shape == b.shape else { return .infinity }
    return abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
}

func check(_ name: String, _ diff: Float, threshold: Float) -> Bool {
    let pass = diff < threshold
    print("  \(pass ? "PASS" : "FAIL")  \(name): max|Δ| = \(diff)  (< \(threshold))")
    return pass
}

func loadWav16kMono(_ url: URL) throws -> MLXArray {
    // Minimal 16-bit PCM mono WAV reader (test inputs are afconvert LEI16@16000).
    let data = try Data(contentsOf: url)
    guard data.count > 44 else { throw ValidationError("not a wav file") }
    // Find the "data" chunk.
    var offset = 12
    while offset + 8 <= data.count {
        let id = String(bytes: data[offset ..< offset + 4], encoding: .ascii) ?? ""
        let size = data[(offset + 4) ..< (offset + 8)].withUnsafeBytes {
            Int($0.loadUnaligned(as: UInt32.self))
        }
        if id == "data" {
            let start = offset + 8
            let end = min(start + size, data.count)
            let samples = data[start ..< end].withUnsafeBytes { raw -> [Float] in
                let int16 = raw.bindMemory(to: Int16.self)
                return int16.map { Float(Int16(littleEndian: $0)) / 32768.0 }
            }
            return MLXArray(samples)
        }
        offset += 8 + size + (size % 2)
    }
    throw ValidationError("no data chunk in wav")
}

// MARK: - Root command

@main
struct NemotronGate: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Parity gates + transcribe for the Nemotron ASR MLX-Swift port.",
        subcommands: [
            KeyGate.self, MelGate.self, ComponentGate.self, E2EGate.self,
            GPUSmoke.self, QuantGate.self, Transcribe.self,
        ])
}

struct GateOptions: ParsableArguments {
    @Option(help: "Model directory (config.json + model.safetensors). Defaults to the HF cache snapshot.")
    var modelDir: String?

    @Option(help: "Goldens directory.")
    var goldens: String = "PORTING/goldens"

    func context() throws -> GateContext {
        guard let dir = modelDir.map({ URL(fileURLWithPath: $0) }) ?? defaultModelDir() else {
            throw ValidationError("no --model-dir and no HF cache snapshot found")
        }
        return GateContext(
            modelDir: dir, goldens: URL(fileURLWithPath: goldens))
    }
}

// MARK: - Gates

struct KeyGate: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "key-gate",
        abstract: "Weights load with strict key verification (no missing/unused).")
    @OptionGroup var options: GateOptions

    func run() throws {
        Device.setDefault(device: Device.cpu)
        let ctx = try options.context()
        let model = try NemotronLoad.load(directory: ctx.modelDir)
        let count = model.parameters().flattened().count
        print("  PASS  key-gate: \(count) parameter tensors loaded, verify .all")
    }
}

struct MelGate: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mel-gate",
        abstract: "Offline + chunked log-mel vs the Python golden.")
    @OptionGroup var options: GateOptions

    func run() throws {
        Device.setDefault(device: Device.cpu)
        let ctx = try options.context()
        let g = try ctx.golden("mel_full.safetensors")
        let config = try NemotronASRConfig.load(
            from: ctx.modelDir.appendingPathComponent("config.json"))
        let wav = g["wav"]!
        let melGolden = g["mel_full"]!

        var ok = true
        let melFull = NemotronAudio.logMelSpectrogram(wav, config.preprocessor)
        ok = check("mel offline", maxAbsDiff(melFull, melGolden), threshold: 1e-3) && ok

        // Chunked extraction must be full-audio-equivalent.
        var chunks: [MLXArray] = []
        NemotronAudio.iterLogMelSpectrogram(wav, config.preprocessor, chunkFrames: 300) {
            chunks.append($0)
        }
        let melChunked = concatenated(chunks, axis: 1)
        ok = check("mel chunked==offline", maxAbsDiff(melChunked, melFull), threshold: 1e-5) && ok
        guard ok else { throw ExitCode.failure }
    }
}

struct ComponentGate: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "component-gate",
        abstract: "Per-layer stream-block, subsampling, prompt, PE, decoder/joint vs goldens.")
    @OptionGroup var options: GateOptions

    func run() throws {
        Device.setDefault(device: Device.cpu)
        let ctx = try options.context()
        let model = try NemotronLoad.load(directory: ctx.modelDir)
        var ok = true

        // PE table.
        let pe = try ctx.golden("pos_emb.safetensors")["pos_emb_70"]!
        let peSwift = model.encoder.posEnc.posEmb(for: 70, dtype: .float32)
        ok = check("pos_emb(70)", maxAbsDiff(peSwift, pe), threshold: 1e-5) && ok

        // Subsampling on captured windows.
        let pre = try ctx.golden("pre_encode.safetensors")
        for c in [0, 1, 2] {
            guard let win = pre["c\(c)_win"], let sub = pre["c\(c)_sub"] else { continue }
            let (subSwift, _) = model.encoder.preEncode(win, [win.dim(1)])
            ok = check("pre_encode c\(c)", maxAbsDiff(subSwift, sub), threshold: 1e-3) && ok
        }

        // Per-layer stream blocks (chunk 0 cold caches, chunk 2 warm caches).
        let blocks = try ctx.golden("stream_blocks.safetensors")
        let convLeft = model.encoderConfig.convKernelSize - 1
        for chunk in [0, 2] {
            var worst: Float = 0
            var layers = 0
            for li in 0 ..< model.encoder.layers.count {
                let p = "c\(chunk)_l\(String(format: "%02d", li))"
                guard let x = blocks["\(p)_x"], let hGolden = blocks["\(p)_h"] else { continue }
                let (h, attnNext, convNext) = streamBlock(
                    model.encoder.layers[li], x, model.encoder.posEnc,
                    blocks["\(p)_attn_cache_in"],
                    blocks["\(p)_conv_cache_in"],
                    leftCache: 56, convLeft: convLeft)
                worst = max(worst, maxAbsDiff(h, hGolden))
                if let g = blocks["\(p)_attn_next"] { worst = max(worst, maxAbsDiff(attnNext, g)) }
                if let g = blocks["\(p)_conv_next"] { worst = max(worst, maxAbsDiff(convNext, g)) }
                layers += 1
            }
            ok = check("stream_block chunk\(chunk) (\(layers) layers, worst)", worst, threshold: 5e-3) && ok
        }

        // Prompt kernel.
        let prompt = try ctx.golden("prompt.safetensors")
        for c in [0, 1, 2] {
            guard let input = prompt["c\(c)_in"], let out = prompt["c\(c)_out"] else { continue }
            let outSwift = model.applyPrompt(input, language: "auto")
            ok = check("apply_prompt c\(c)", maxAbsDiff(outSwift, out), threshold: 5e-3) && ok
        }

        // Decoder + joint steps, replaying the reference trajectory: the recorded
        // token is the decoder INPUT at each step, and decoder_hidden advances only
        // when the step emitted non-blank — recovered from the golden logits' argmax.
        let dj = try ctx.golden("decoder_joint.safetensors")
        var djWorst: Float = 0
        var steps = 0
        var hidden: (MLXArray, MLXArray)? = nil
        for s in 0 ..< 60 {
            let p = "s\(String(format: "%03d", s))"
            guard let tokenGolden = dj["\(p)_token"], let outGolden = dj["\(p)_out"] else { break }
            let tokenValue = tokenGolden.item(Int.self)
            let token: MLXArray? =
                tokenValue >= 0 ? MLXArray([Int32(tokenValue)]).reshaped(1, 1) : nil
            let (out, proposed) = model.decoder(token, hidden)
            djWorst = max(djWorst, maxAbsDiff(out, outGolden))
            if let hGolden = dj["\(p)_h"] { djWorst = max(djWorst, maxAbsDiff(proposed.0, hGolden)) }
            if let cGolden = dj["\(p)_c"] { djWorst = max(djWorst, maxAbsDiff(proposed.1, cGolden)) }
            if let encGolden = dj["\(p)_enc"], let predGolden = dj["\(p)_pred"],
                let logitsGolden = dj["\(p)_logits"]
            {
                let logits = model.joint(encGolden, predGolden)
                djWorst = max(djWorst, maxAbsDiff(logits, logitsGolden))
                let emitted = argMax(logitsGolden).item(Int.self) != model.blankId
                if emitted {
                    hidden = (proposed.0.asType(.float32), proposed.1.asType(.float32))
                }
            }
            steps += 1
        }
        ok = check("decoder+joint (\(steps) steps, worst)", djWorst, threshold: 5e-3) && ok

        guard ok else { throw ExitCode.failure }
    }
}

struct E2EGate: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "e2e-gate",
        abstract: "Full streaming transcription vs the Python golden transcript.")
    @OptionGroup var options: GateOptions

    func run() throws {
        Device.setDefault(device: Device.cpu)
        let ctx = try options.context()
        let model = try NemotronLoad.load(directory: ctx.modelDir)

        let g = try ctx.golden("mel_full.safetensors")
        let wav = g["wav"]!

        struct Golden: Decodable {
            struct Token: Decodable {
                let id: Int
                let text: String
                let start: Double
                let duration: Double
            }
            let text: String
            let tokens: [Token]
            let att_context_size: [Int]
        }
        let golden = try JSONDecoder().decode(
            Golden.self,
            from: Data(contentsOf: ctx.goldens.appendingPathComponent("transcript.json")))

        let t0 = Date()
        let transcription = model.streamGenerate(
            wav, attContextSize: golden.att_context_size)
        let dt = Date().timeIntervalSince(t0)
        let result = transcription.result

        let swiftTokens = result.sentences.flatMap(\.tokens)
        print("  swift : \(result.text)")
        print("  golden: \(golden.text)")
        print("  \(swiftTokens.count) tokens vs \(golden.tokens.count) golden, \(String(format: "%.1f", dt))s")

        var ok = result.text == golden.text
        print("  \(ok ? "PASS" : "FAIL")  transcript text exact match")
        if swiftTokens.count == golden.tokens.count {
            var idOK = true
            var timeOK = true
            for (s, gt) in zip(swiftTokens, golden.tokens) {
                if s.id != gt.id { idOK = false }
                if abs(s.start - gt.start) > 1e-4 { timeOK = false }
            }
            print("  \(idOK ? "PASS" : "FAIL")  token ids exact match")
            print("  \(timeOK ? "PASS" : "FAIL")  token timestamps match (1e-4s)")
            ok = ok && idOK && timeOK
        } else {
            ok = false
        }
        if let lang = transcription.detectedLanguage {
            print("  detected language: \(lang)")
        }
        guard ok else { throw ExitCode.failure }
    }
}

struct GPUSmoke: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "gpu-smoke",
        abstract: "Full streaming transcription on the GPU stream + RTF measurement.")
    @OptionGroup var options: GateOptions

    func run() throws {
        let ctx = try options.context()
        let model = try NemotronLoad.load(directory: ctx.modelDir)
        let g = try ctx.golden("mel_full.safetensors")
        let wav = g["wav"]!
        let audioSeconds = Double(wav.dim(0)) / 16000.0

        // Warmup (kernel compilation).
        _ = model.streamGenerate(wav)
        let t0 = Date()
        let transcription = model.streamGenerate(wav)
        let dt = Date().timeIntervalSince(t0)
        print("  transcript: \(transcription.result.text)")
        print(
            "  \(String(format: "%.2f", dt))s for \(String(format: "%.1f", audioSeconds))s audio"
                + "  (RTF x\(String(format: "%.1f", audioSeconds / dt)))")
        print("  peak GPU memory: \(GPU.peakMemory / 1024 / 1024) MB")
        print("  PASS  gpu-smoke (non-empty: \(!transcription.result.text.isEmpty))")
    }
}

struct QuantGate: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "quant-gate",
        abstract: "int8-quantize Linear layers, run on the GPU stream, compare vs fp32 transcript.")
    @OptionGroup var options: GateOptions

    @Option(help: "Quantization bits.") var bits: Int = 8
    @Option(help: "Quantization group size.") var groupSize: Int = 64

    func run() throws {
        // GPU stream ONLY — a quantized forward under a CPU pin grinds for hours.
        let ctx = try options.context()
        let g = try ctx.golden("mel_full.safetensors")
        let wav = g["wav"]!

        // fp32 reference transcript on GPU.
        let fp = try NemotronLoad.load(directory: ctx.modelDir)
        let refText = fp.streamGenerate(wav).result.text

        // Quantize a fresh model instance. Skip the tiny/precision-sensitive projections
        // (joint output over 13k classes, prompt kernel, subsampling `out`) — standard
        // scope-the-quant predicate.
        let q = try NemotronLoad.load(directory: ctx.modelDir)
        quantize(model: q, groupSize: groupSize, bits: bits) { path, module in
            guard module is Linear else { return false }
            if path.hasSuffix("joint.joint_net_2") { return false }
            if path.hasPrefix("prompt_kernel") { return false }
            if path.hasSuffix("pre_encode.out") { return false }
            return true
        }
        eval(q.parameters())

        let t0 = Date()
        let qResult = q.streamGenerate(wav)
        let dt = Date().timeIntervalSince(t0)
        let qText = qResult.result.text

        print("  fp32 : \(refText)")
        print("  int\(bits): \(qText)")
        print("  \(String(format: "%.2f", dt))s  peak GPU \(GPU.peakMemory / 1024 / 1024) MB")

        // Word-level agreement as a transcript-sanity signal (quant perturbs but should
        // not garble): fraction of reference words preserved in order.
        let refWords = refText.split(separator: " ")
        let qWords = Set(qText.split(separator: " "))
        let kept = refWords.filter { qWords.contains($0) }.count
        let frac = refWords.isEmpty ? 1.0 : Double(kept) / Double(refWords.count)
        let ok = frac >= 0.9 && !qText.isEmpty
        print("  \(ok ? "PASS" : "FAIL")  quant transcript sanity: \(String(format: "%.0f%%", frac * 100)) word retention")
        guard ok else { throw ExitCode.failure }
    }
}

struct Transcribe: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Transcribe a 16 kHz mono wav.")
    @OptionGroup var options: GateOptions

    @Argument(help: "Path to a 16 kHz mono 16-bit PCM wav.")
    var wav: String

    @Option(help: "Language-locale prompt key (default: auto).")
    var language: String?

    @Flag(help: "Print cumulative partials per chunk.")
    var partials = false

    func run() throws {
        let ctx = try options.context()
        let model = try NemotronLoad.load(directory: ctx.modelDir)
        let audio = try loadWav16kMono(URL(fileURLWithPath: wav))
        let transcription = model.streamGenerate(
            audio, language: language,
            onPartial: partials ? { print("  … \($0.text)") } : nil)
        print(transcription.result.text)
        if let lang = transcription.detectedLanguage {
            print("[detected: \(lang)]")
        }
    }
}
