// Push-style live transcription for the cache-aware streaming encoder.
//
// `NemotronASRModel.streamGenerate` transcribes a waveform that is already complete; this
// session is the same machinery with the audio arriving over time. It owns the streaming
// mel front-end (a trailing sample window plus absolute frame bookkeeping, so every frame
// is identical to the whole-buffer pass), the `StreamEncoder` caches, and the greedy RNN-T
// decode state. `streamGenerate` is itself a driver of this session, so the batch and live
// paths cannot drift.

import Foundation
import MLX

/// A live transcription session: push 16 kHz mono samples as they arrive, read cumulative
/// partial transcripts back, then `finish` for the final result.
///
/// Not thread-safe and not `Sendable` — callers serialize access. (The MLXNemotronSTT kit
/// drives it on the engine's inference actor.)
public final class NemotronStreamingSession {

    /// Cumulative transcription state after one or more decoded chunks.
    public struct Update: Sendable {
        /// Every token decoded since the session started; timestamps are absolute.
        public let result: AlignedResult
        /// The first `<xx-YY>` tag the model emitted, when running in auto mode.
        public let detectedLanguage: String?
        /// Seconds of audio decoded so far — trails the audio pushed by up to one chunk.
        public let processedSeconds: Double
    }

    private let model: NemotronASRModel
    private let args: PreprocessArgs
    private let encoder: StreamEncoder
    /// Upper bound on the frames turned into mel at once, so one huge push cannot
    /// materialize an unbounded STFT.
    private let maxFramesPerEmit: Int

    private var state: NemotronASRModel.DecodeState
    /// The trailing slice of the stream that frames not yet emitted still reach into.
    private var tail = MLXArray.zeros([0], dtype: .float32)
    /// Global index of `tail[0]`.
    private var tailStart = 0
    private var totalSamples = 0
    private var melEmitted = 0
    private var finished = false

    /// True once `cancel()` was called; pushes become no-ops and `finish` stops flushing.
    public private(set) var isCancelled = false

    /// - Parameters:
    ///   - language: prompt key (e.g. `"en-US"`, `"auto"`); `nil` uses the model default.
    ///   - chunkFrames: encoder frames per chunk; `nil` uses the native `right + 1`.
    ///   - attContextSize: `[left, right]` look-ahead — the latency lever. `[56, 13]`
    ///     (default) decodes 1.12 s chunks, `[56, 0]` decodes 80 ms chunks.
    ///   - melChunkDuration: caps the audio converted to mel per step, bounding STFT memory.
    public init(
        model: NemotronASRModel,
        language: String? = nil,
        chunkFrames: Int? = nil,
        attContextSize: [Int]? = nil,
        melChunkDuration: Double = 30.0
    ) {
        precondition(
            model.preprocessorConfig.padTo == 0,
            "streaming needs padTo == 0; utterance padding has no streaming equivalent")
        self.model = model
        self.args = model.preprocessorConfig
        self.encoder = StreamEncoder(
            model: model, language: language ?? model.defaultLanguage,
            chunkFrames: chunkFrames, attContextSize: attContextSize)
        self.maxFramesPerEmit = model.melChunkFrames(chunkDuration: melChunkDuration)
        self.state = NemotronASRModel.DecodeState(lastToken: model.blankId, decoderHidden: nil)
    }

    // ------------------------------------------------------------------- interface

    /// Samples of 16 kHz audio per native encode chunk — 17 920 (1.12 s) at the default
    /// `[56, 13]` context. Informational: `push` accepts any buffer size.
    public var samplesPerChunk: Int { encoder.chunkMel * args.hopLength }

    /// Seconds of audio decoded so far.
    public var processedSeconds: Double { Double(state.globalTime) * model.frameSeconds }

    /// Feed 16 kHz mono samples. Any buffer size works — the session accumulates until a
    /// native chunk is available, then encodes and decodes it, so `onPartial` fires zero or
    /// more times per push. Returns the cumulative update if at least one chunk decoded.
    @discardableResult
    public func push(_ samples: [Float], onPartial: ((AlignedResult) -> Void)? = nil) -> Update? {
        push(MLXArray(samples), onPartial: onPartial)
    }

    @discardableResult
    public func push(_ samples: MLXArray, onPartial: ((AlignedResult) -> Void)? = nil) -> Update? {
        guard !isCancelled, !finished else { return nil }
        append(samples)

        var decoded = false
        let handle = chunkHandler(onPartial) { decoded = true }
        while !isCancelled, let mel = nextReadyMel() {
            encoder.push(mel, handle)
        }
        guard decoded else { return nil }
        return Update(
            result: sentencesToResult(tokensToSentences(state.hypothesis)),
            detectedLanguage: state.detectedLanguage,
            processedSeconds: processedSeconds)
    }

    /// Flush the tail — right zero-padded exactly as a whole-buffer pass would — and return
    /// the final transcription. Idempotent; later pushes are ignored.
    @discardableResult
    public func finish(
        onPartial: ((AlignedResult) -> Void)? = nil
    ) -> NemotronASRModel.Transcription {
        defer { MLX.GPU.clearCache() }
        guard !finished, !isCancelled else { return transcription() }
        finished = true

        let handle = chunkHandler(onPartial) {}
        let totalFrames = totalSamples / args.hopLength + 1
        while !isCancelled, melEmitted < totalFrames {
            let frameEnd = min(totalFrames, melEmitted + maxFramesPerEmit)
            encoder.push(emit(through: frameEnd, totalSamples: totalSamples), handle)
        }
        if !isCancelled { encoder.finish(handle) }
        return transcription()
    }

    /// Stop decoding. `finish` then returns whatever was decoded before the stop.
    public func cancel() { isCancelled = true }

    // -------------------------------------------------------------------- internals

    private func chunkHandler(
        _ onPartial: ((AlignedResult) -> Void)?, _ didDecode: @escaping () -> Void
    ) -> (MLXArray) -> Void {
        { prompted in
            guard !self.isCancelled else { return }
            self.model.decodePromptedChunk(prompted, state: &self.state)
            onPartial?(sentencesToResult(tokensToSentences(self.state.hypothesis)))
            MLX.GPU.clearCache()
            didDecode()
        }
    }

    private func append(_ samples: MLXArray) {
        let flat = samples.ndim == 1 ? samples : samples.reshaped(-1)
        guard flat.dim(0) > 0 else { return }
        tail = totalSamples == 0 ? flat : concatenated([tail, flat], axis: 0)
        totalSamples += flat.dim(0)
        eval(tail)  // collapse the graph so the window holds samples, not lazy nodes
    }

    /// The next batch of frames whose windows have fully arrived, or nil if none have.
    private func nextReadyMel() -> MLXArray? {
        let half = args.nFFT / 2
        // Frame k spans [k·hop − nFFT/2, k·hop + nFFT/2), so it is complete once
        // k·hop + nFFT/2 samples exist.
        let ready = totalSamples >= half ? (totalSamples - half) / args.hopLength + 1 : 0
        guard ready > melEmitted else { return nil }
        return emit(through: min(ready, melEmitted + maxFramesPerEmit), totalSamples: nil)
    }

    /// Convert frames `[melEmitted, frameEnd)` to mel and release the samples behind them.
    private func emit(through frameEnd: Int, totalSamples known: Int?) -> MLXArray {
        let mel = NemotronAudio.logMelFrames(
            buffer: tail, bufferStart: tailStart, totalSamples: known, args,
            frameStart: melEmitted, frameEnd: frameEnd)
        melEmitted = frameEnd

        // Keep only what later frames still read, plus the pre-emphasis look-back sample.
        let keepFrom = max(0, melEmitted * args.hopLength - args.nFFT / 2 - 1)
        let drop = min(keepFrom - tailStart, tail.dim(0))
        if drop > 0 {
            tail = tail[drop...]
            tailStart += drop
            eval(tail)
        }
        return mel
    }

    private func transcription() -> NemotronASRModel.Transcription {
        NemotronASRModel.Transcription(
            result: sentencesToResult(tokensToSentences(state.hypothesis)),
            detectedLanguage: state.detectedLanguage)
    }
}
