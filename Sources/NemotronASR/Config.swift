// Configuration for the Nemotron 3.5 ASR (streaming) model.
// Mirrors mlx_audio/stt/models/nemotron_asr/config.py — the schema of the
// config.json produced by the reference convert.py.

import Foundation

/// Mel-spectrogram featurizer settings (NeMo AudioToMelSpectrogramPreprocessor).
public struct PreprocessArgs: Codable, Sendable {
    public var sampleRate: Int = 16000
    public var features: Int = 128
    public var nFFT: Int = 512
    public var windowSize: Double = 0.025
    public var windowStride: Double = 0.01
    public var window: String = "hann"
    public var preemph: Float = 0.97
    public var dither: Float = 1.0e-05
    /// NeMo `normalize: NA` => no normalization.
    public var normalize: String = "NA"
    public var logZeroGuardValue: Float = 5.960464477539063e-08  // 2^-24
    public var padTo: Int = 0
    public var padValue: Float = 0.0

    public var winLength: Int { Int(windowSize * Double(sampleRate)) }
    public var hopLength: Int { Int(windowStride * Double(sampleRate)) }

    enum CodingKeys: String, CodingKey {
        case sampleRate = "sample_rate"
        case features
        case nFFT = "n_fft"
        case windowSize = "window_size"
        case windowStride = "window_stride"
        case window
        case preemph
        case dither
        case normalize
        case logZeroGuardValue = "log_zero_guard_value"
        case padTo = "pad_to"
        case padValue = "pad_value"
    }
}

/// Cache-aware FastConformer encoder settings.
public struct ConformerArgs: Codable, Sendable {
    public var featIn: Int = 128
    public var nLayers: Int = 24
    public var dModel: Int = 1024
    public var nHeads: Int = 8
    public var ffExpansionFactor: Int = 4
    public var subsamplingFactor: Int = 8
    public var subsamplingConvChannels: Int = 256
    public var convKernelSize: Int = 9
    public var causalDownsampling: Bool = true
    /// "causal" or [left, right]; resolved against convKernelSize at build time.
    public var convContextSize: ConvContextSize = .causal
    public var convNormType: String = "layer_norm"
    public var selfAttentionModel: String = "rel_pos"
    public var attContextStyle: String = "chunked_limited"
    /// All look-ahead contexts the model was trained with (list of [left, right]).
    public var attContextSize: [[Int]] = [[56, 13]]
    public var posEmbMaxLen: Int = 5000
    public var useBias: Bool = false
    public var xscaling: Bool = false

    public enum ConvContextSize: Codable, Sendable {
        case causal
        case explicit(left: Int, right: Int)

        public init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let s = try? c.decode(String.self), s == "causal" {
                self = .causal
            } else {
                let pair = try c.decode([Int].self)
                self = .explicit(left: pair[0], right: pair[1])
            }
        }

        public func encode(to encoder: Encoder) throws {
            var c = encoder.singleValueContainer()
            switch self {
            case .causal: try c.encode("causal")
            case .explicit(let l, let r): try c.encode([l, r])
            }
        }
    }

    enum CodingKeys: String, CodingKey {
        case featIn = "feat_in"
        case nLayers = "n_layers"
        case dModel = "d_model"
        case nHeads = "n_heads"
        case ffExpansionFactor = "ff_expansion_factor"
        case subsamplingFactor = "subsampling_factor"
        case subsamplingConvChannels = "subsampling_conv_channels"
        case convKernelSize = "conv_kernel_size"
        case causalDownsampling = "causal_downsampling"
        case convContextSize = "conv_context_size"
        case convNormType = "conv_norm_type"
        case selfAttentionModel = "self_attention_model"
        case attContextStyle = "att_context_style"
        case attContextSize = "att_context_size"
        case posEmbMaxLen = "pos_emb_max_len"
        case useBias = "use_bias"
        case xscaling
    }
}

/// Language-ID prompt conditioning (PromptStreamingMixin).
public struct PromptArgs: Codable, Sendable {
    public var numPrompts: Int = 128
    public var promptHidden: Int = 2048
    public var promptDictionary: [String: Int] = [:]

    enum CodingKeys: String, CodingKey {
        case numPrompts = "num_prompts"
        case promptHidden = "prompt_hidden"
        case promptDictionary = "prompt_dictionary"
    }
}

/// RNN-T prediction (decoder) network.
public struct PredictArgs: Codable, Sendable {
    public var predHidden: Int = 640
    public var predRnnLayers: Int = 2
    public var vocabSize: Int = 13087
    public var blankAsPad: Bool = true

    enum CodingKeys: String, CodingKey {
        case predHidden = "pred_hidden"
        case predRnnLayers = "pred_rnn_layers"
        case vocabSize = "vocab_size"
        case blankAsPad = "blank_as_pad"
    }
}

/// RNN-T joint network.
public struct JointArgs: Codable, Sendable {
    public var jointHidden: Int = 640
    public var activation: String = "relu"
    public var encoderHidden: Int = 1024
    public var predHidden: Int = 640
    public var numClasses: Int = 13087

    enum CodingKeys: String, CodingKey {
        case jointHidden = "joint_hidden"
        case activation
        case encoderHidden = "encoder_hidden"
        case predHidden = "pred_hidden"
        case numClasses = "num_classes"
    }
}

public struct NemotronASRConfig: Codable, Sendable {
    public var preprocessor: PreprocessArgs
    public var encoder: ConformerArgs
    public var prompt: PromptArgs
    public var decoder: PredictArgs
    public var joint: JointArgs
    public var vocabulary: [String]
    public var modelType: String = "nemotron_asr"
    /// Default language prompt key (e.g. "auto", "en-US").
    public var defaultLanguage: String = "auto"
    /// Look-ahead [left, right] used at inference. [56, 13] = best offline accuracy.
    public var defaultAttContextSize: [Int] = [56, 13]
    public var maxSymbols: Int = 10

    enum CodingKeys: String, CodingKey {
        case preprocessor, encoder, prompt, decoder, joint, vocabulary
        case modelType = "model_type"
        case defaultLanguage = "default_language"
        case defaultAttContextSize = "default_att_context_size"
        case maxSymbols = "max_symbols"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        preprocessor = try c.decodeIfPresent(PreprocessArgs.self, forKey: .preprocessor) ?? PreprocessArgs()
        encoder = try c.decodeIfPresent(ConformerArgs.self, forKey: .encoder) ?? ConformerArgs()
        prompt = try c.decodeIfPresent(PromptArgs.self, forKey: .prompt) ?? PromptArgs()
        self.decoder = try c.decodeIfPresent(PredictArgs.self, forKey: .decoder) ?? PredictArgs()
        joint = try c.decodeIfPresent(JointArgs.self, forKey: .joint) ?? JointArgs()
        vocabulary = try c.decodeIfPresent([String].self, forKey: .vocabulary) ?? []
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "nemotron_asr"
        defaultLanguage = try c.decodeIfPresent(String.self, forKey: .defaultLanguage) ?? "auto"
        defaultAttContextSize = try c.decodeIfPresent([Int].self, forKey: .defaultAttContextSize) ?? [56, 13]
        maxSymbols = try c.decodeIfPresent(Int.self, forKey: .maxSymbols) ?? 10
    }

    public static func load(from url: URL) throws -> NemotronASRConfig {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(NemotronASRConfig.self, from: data)
    }
}
