// Nemotron 3.5 ASR (streaming, 0.6B) for MLX-Swift.
// Mirrors mlx_audio/stt/models/nemotron_asr/nemotron_asr.py: a cache-aware streaming
// FastConformer-RNNT with language-ID prompt conditioning
// (NeMo `EncDecRNNTBPEModelWithPrompt`).

import Foundation
import MLX
import MLXNN

public final class NemotronASRModel: Module {
    public let config: NemotronASRConfig
    public let preprocessorConfig: PreprocessArgs
    public let encoderConfig: ConformerArgs
    public let vocabulary: [String]
    public let promptDictionary: [String: Int]
    public let numPrompts: Int
    public let blankId: Int
    public let maxSymbols: Int?
    public let defaultLanguage: String
    public let defaultAttContextSize: [Int]

    @ModuleInfo(key: "encoder") public var encoder: Conformer
    // prompt_kernel: reference list [Linear, ReLU, Linear] → checkpoint keys
    // prompt_kernel.0 / prompt_kernel.2; remapped to _0/_2 (dots illegal in keys).
    @ModuleInfo(key: "prompt_kernel_0") var promptKernel0: Linear
    @ModuleInfo(key: "prompt_kernel_2") var promptKernel2: Linear
    @ModuleInfo(key: "decoder") public var decoder: PredictNetwork
    @ModuleInfo(key: "joint") public var joint: JointNetwork

    public init(config: NemotronASRConfig) {
        self.config = config
        self.preprocessorConfig = config.preprocessor
        self.encoderConfig = config.encoder
        self.vocabulary = config.vocabulary
        self.promptDictionary = config.prompt.promptDictionary
        self.numPrompts = config.prompt.numPrompts
        self.blankId = config.decoder.vocabSize  // == num_classes
        self.maxSymbols = config.maxSymbols
        self.defaultLanguage = config.defaultLanguage
        self.defaultAttContextSize = config.defaultAttContextSize

        self._encoder.wrappedValue = Conformer(config.encoder)
        self._promptKernel0.wrappedValue = Linear(
            config.encoder.dModel + config.prompt.numPrompts, config.prompt.promptHidden)
        self._promptKernel2.wrappedValue = Linear(
            config.prompt.promptHidden, config.encoder.dModel)
        self._decoder.wrappedValue = PredictNetwork(config.decoder)
        self._joint.wrappedValue = JointNetwork(config.joint)
        super.init()
    }

    // ------------------------------------------------------------------ prompt

    func resolvePromptIndex(_ language: String?) -> Int {
        let lang = language ?? defaultLanguage
        if let idx = promptDictionary[lang] { return idx }
        if let idx = promptDictionary[defaultLanguage] { return idx }
        return 0
    }

    /// Concatenate the one-hot language prompt and project back to d_model.
    public func applyPrompt(_ encoded: MLXArray, language: String?) -> MLXArray {
        let idx = resolvePromptIndex(language)
        let (b, t) = (encoded.dim(0), encoded.dim(1))
        var oneHot = MLXArray.zeros([b, t, numPrompts], dtype: encoded.dtype)
        oneHot[0..., 0..., idx] = MLXArray(Float(1)).asType(encoded.dtype)
        var x = concatenated([encoded, oneHot], axis: -1)
        x = promptKernel0(x)
        x = relu(x)
        x = promptKernel2(x)
        return x
    }

    // ------------------------------------------------------------------ decode

    /// Seconds of audio per encoder output frame.
    public var frameSeconds: Double {
        Double(encoderConfig.subsamplingFactor)
            * Double(preprocessorConfig.hopLength)
            / Double(preprocessorConfig.sampleRate)
    }

    /// Greedy RNN-T decode state threaded across prompted chunks.
    struct DecodeState {
        var lastToken: Int
        var decoderHidden: (MLXArray, MLXArray)?
        var hypothesis: [AlignedToken] = []
        var globalTime = 0
        /// First emitted `<xx-YY>` tag (auto mode). Lang tags are special tokens and
        /// never enter `hypothesis`, so detection is captured here at emission time.
        var detectedLanguage: String? = nil
    }

    /// Decode one prompted chunk, mutating `state`; mirrors the inner loop of
    /// `_decode_prompted_chunks`.
    func decodePromptedChunk(_ prompted: MLXArray, state: inout DecodeState) {
        let chunkLen = prompted.dim(1)
        let frameSec = frameSeconds
        var time = 0
        var newSymbols = 0
        while time < chunkLen {
            let feature = prompted[0..., time ..< (time + 1)]
            let currentToken: MLXArray? =
                state.lastToken != blankId
                ? MLXArray([Int32(state.lastToken)]).reshaped(1, 1) : nil
            var (decoderOutput, hc) = decoder(currentToken, state.decoderHidden)
            decoderOutput = decoderOutput.asType(feature.dtype)
            let proposedHidden = (hc.0.asType(feature.dtype), hc.1.asType(feature.dtype))
            let jointOutput = joint(feature, decoderOutput)
            let predToken = argMax(jointOutput).item(Int.self)

            if predToken != blankId {
                state.lastToken = predToken
                state.decoderHidden = proposedHidden
                if !Tokenizer.isSpecialToken(predToken, vocabulary: vocabulary) {
                    state.hypothesis.append(
                        AlignedToken(
                            id: predToken,
                            text: Tokenizer.decode([predToken], vocabulary: vocabulary),
                            start: Double(state.globalTime + time) * frameSec,
                            duration: frameSec))
                } else if state.detectedLanguage == nil, predToken < vocabulary.count,
                    Tokenizer.isLangTag(vocabulary[predToken])
                {
                    state.detectedLanguage = String(
                        vocabulary[predToken].dropFirst().dropLast())
                }
                newSymbols += 1
                if let maxSymbols, newSymbols >= maxSymbols {
                    time += 1
                    newSymbols = 0
                }
            } else {
                time += 1
                newSymbols = 0
            }
        }
        state.globalTime += chunkLen
    }

    // ---------------------------------------------------------- stream generate

    func melChunkFrames(chunkDuration: Double) -> Int {
        precondition(chunkDuration > 0, "chunkDuration must be positive")
        return max(
            Int(chunkDuration * Double(preprocessorConfig.sampleRate)
                / Double(preprocessorConfig.hopLength)),
            1)
    }

    /// A transcription result plus the language the model detected (auto mode).
    public struct Transcription: Sendable {
        public let result: AlignedResult
        public let detectedLanguage: String?
    }

    /// Cache-aware streaming transcription of a 16 kHz mono waveform.
    ///
    /// Calls `onPartial` with the cumulative `AlignedResult` after each decoded chunk
    /// (mirrors the reference generator's per-chunk yield) and returns the final
    /// result. `shouldContinue` is polled per chunk — returning `false` stops early
    /// (the package layer maps this to cooperative cancellation).
    ///
    /// This drives a `NemotronStreamingSession` over already-captured audio; feed a session
    /// directly to transcribe audio that is still arriving.
    @discardableResult
    public func streamGenerate(
        _ audio: MLXArray,
        language: String? = nil,
        chunkFrames: Int? = nil,
        chunkDuration: Double = 30.0,
        attContextSize: [Int]? = nil,
        onPartial: ((AlignedResult) -> Void)? = nil,
        shouldContinue: @escaping () -> Bool = { true }
    ) -> Transcription {
        let session = NemotronStreamingSession(
            model: self, language: language, chunkFrames: chunkFrames,
            attContextSize: attContextSize, melChunkDuration: chunkDuration)

        let relay: (AlignedResult) -> Void = { partial in
            onPartial?(partial)
            if !shouldContinue() { session.cancel() }
        }

        // Push in mel-chunk strides so STFT memory stays bounded on long audio.
        let step = melChunkFrames(chunkDuration: chunkDuration) * preprocessorConfig.hopLength
        let total = audio.dim(0)
        var start = 0
        while start < total, !session.isCancelled {
            let end = min(start + step, total)
            session.push(audio[start ..< end], onPartial: relay)
            start = end
        }
        return session.finish(onPartial: relay)
    }

    /// Transcribe a complete utterance (16 kHz mono waveform). `language` is a prompt
    /// key (e.g. "en-US", "auto"); defaults to the model's default.
    public func generate(
        _ audio: MLXArray,
        language: String? = nil,
        attContextSize: [Int]? = nil,
        chunkDuration: Double = 30.0
    ) -> Transcription {
        streamGenerate(
            audio, language: language, chunkDuration: chunkDuration,
            attContextSize: attContextSize)
    }
}
