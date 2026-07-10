// Cache-aware FastConformer encoder for Nemotron 3.5 ASR.
// Mirrors mlx_audio/stt/models/nemotron_asr/conformer.py:
//   * causal `dw_striding` subsampling (CausalConv2D, asymmetric pad),
//   * convolution module with `layer_norm` and a *causal* depthwise conv,
//   * `rel_pos` attention with the `chunked_limited` look-ahead mask,
//   * no bias in the conformer-block projections (`use_bias=false`).

import Foundation
import MLX
import MLXNN

let negInf: Float = -1e30

/// Additive (1, 1, T, T) attention mask for NeMo's `chunked_limited` style.
/// Frames group into non-overlapping chunks of `rightContext + 1`; a frame may
/// attend to its own chunk and the previous `leftContext / chunk` chunks.
public func createChunkedLimitedMask(
    seqLen: Int, leftContext: Int, rightContext: Int
) -> MLXArray {
    let chunkSize = rightContext + 1
    let leftChunks = leftContext >= 0 ? leftContext / chunkSize : 100_000_000

    let chunkIdx = MLXArray(0 ..< Int32(seqLen)) / Int32(chunkSize)
    let diff = expandedDimensions(chunkIdx, axis: 1) - expandedDimensions(chunkIdx, axis: 0)
    let visible = (diff .>= 0) .&& (diff .<= Int32(leftChunks))
    let mask = MLX.where(visible, MLXArray(Float(0)), MLXArray(negInf)).asType(.float32)
    return mask[.newAxis, .newAxis, 0..., 0...]
}

public final class FeedForward: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear
    let activation = SiLU()

    public init(dModel: Int, dFF: Int, useBias: Bool) {
        self._linear1.wrappedValue = Linear(dModel, dFF, bias: useBias)
        self._linear2.wrappedValue = Linear(dFF, dModel, bias: useBias)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(activation(linear1(x)))
    }
}

/// Convolution module with causal depthwise conv and LayerNorm.
/// NeMo names the normalization layer `batch_norm` even when it is a LayerNorm;
/// the key is kept so checkpoint keys line up.
public final class ConformerConvolution: Module {
    let kernelSize: Int
    let padLeft: Int
    let padRight: Int

    @ModuleInfo(key: "pointwise_conv1") var pointwiseConv1: Conv1d
    @ModuleInfo(key: "depthwise_conv") var depthwiseConv: Conv1d
    @ModuleInfo(key: "batch_norm") var batchNorm: LayerNorm
    @ModuleInfo(key: "pointwise_conv2") var pointwiseConv2: Conv1d
    let activation = SiLU()

    public init(_ args: ConformerArgs) {
        let dModel = args.dModel
        self.kernelSize = args.convKernelSize

        switch args.convContextSize {
        case .causal:
            self.padLeft = kernelSize - 1
            self.padRight = 0
        case .explicit(let l, let r):
            self.padLeft = l
            self.padRight = r
        }

        self._pointwiseConv1.wrappedValue = Conv1d(
            inputChannels: dModel, outputChannels: dModel * 2, kernelSize: 1, bias: args.useBias)
        self._depthwiseConv.wrappedValue = Conv1d(
            inputChannels: dModel, outputChannels: dModel, kernelSize: kernelSize,
            groups: dModel, bias: args.useBias)
        precondition(
            args.convNormType == "layer_norm",
            "conv_norm_type=\(args.convNormType) not supported (expected layer_norm)")
        self._batchNorm.wrappedValue = LayerNorm(dimensions: dModel)
        self._pointwiseConv2.wrappedValue = Conv1d(
            inputChannels: dModel, outputChannels: dModel, kernelSize: 1, bias: args.useBias)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, T, C)
        var x = pointwiseConv1(x)
        x = glu(x, axis: -1)
        // causal depthwise conv: left-pad the time axis only.
        x = padded(x, widths: [IntOrPair((0, 0)), IntOrPair((padLeft, padRight)), IntOrPair((0, 0))])
        x = depthwiseConv(x)
        x = batchNorm(x)
        x = activation(x)
        x = pointwiseConv2(x)
        return x
    }
}

public final class ConformerBlock: Module {
    @ModuleInfo(key: "norm_feed_forward1") var normFeedForward1: LayerNorm
    @ModuleInfo(key: "feed_forward1") var feedForward1: FeedForward
    @ModuleInfo(key: "norm_self_att") var normSelfAtt: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttn: RelPositionMultiHeadAttention
    @ModuleInfo(key: "norm_conv") var normConv: LayerNorm
    @ModuleInfo(key: "conv") var conv: ConformerConvolution
    @ModuleInfo(key: "norm_feed_forward2") var normFeedForward2: LayerNorm
    @ModuleInfo(key: "feed_forward2") var feedForward2: FeedForward
    @ModuleInfo(key: "norm_out") var normOut: LayerNorm

    public init(_ args: ConformerArgs) {
        let dFF = args.dModel * args.ffExpansionFactor
        self._normFeedForward1.wrappedValue = LayerNorm(dimensions: args.dModel)
        self._feedForward1.wrappedValue = FeedForward(
            dModel: args.dModel, dFF: dFF, useBias: args.useBias)
        self._normSelfAtt.wrappedValue = LayerNorm(dimensions: args.dModel)
        self._selfAttn.wrappedValue = RelPositionMultiHeadAttention(
            nHead: args.nHeads, nFeat: args.dModel, bias: args.useBias)
        self._normConv.wrappedValue = LayerNorm(dimensions: args.dModel)
        self._conv.wrappedValue = ConformerConvolution(args)
        self._normFeedForward2.wrappedValue = LayerNorm(dimensions: args.dModel)
        self._feedForward2.wrappedValue = FeedForward(
            dModel: args.dModel, dFF: dFF, useBias: args.useBias)
        self._normOut.wrappedValue = LayerNorm(dimensions: args.dModel)
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray, posEmb: MLXArray, mask: MLXArray? = nil
    ) -> MLXArray {
        var x = x + 0.5 * feedForward1(normFeedForward1(x))
        x = x + selfAttn(normSelfAtt(x), posEmb: posEmb, mask: mask)
        x = x + conv(normConv(x))
        x = x + 0.5 * feedForward2(normFeedForward2(x))
        return normOut(x)
    }
}

/// Depthwise-striding conv subsampling with causal (asymmetric) padding.
///
/// The reference stores convs and ReLUs in one list (checkpoint keys conv.0/2/3/5/6,
/// ReLUs at 1/4/7 carry no params); Swift keeps only the convs in `convs` and the
/// loader remaps the sparse indices (see NemotronLoad.sanitize).
public final class CausalDwStridingSubsampling: Module {
    let samplingNum: Int
    let kernelSize = 3
    let stride = 2
    let padLeft: Int
    let padRight: Int

    @ModuleInfo(key: "convs") var convs: [Conv2d]
    @ModuleInfo(key: "out") var out: Linear

    /// Compacted `convs` indices of the stride-2 3×3 convs that need causal padding
    /// (original list indices 0, 2, 5, … → compacted 0, 1, 3, …).
    let stridedIdx: Set<Int>
    /// Compacted indices followed by a ReLU (original 0, 3, 6, … → 0, 2, 4, …).
    let reluAfterIdx: Set<Int>

    public init(_ args: ConformerArgs) {
        let ch = args.subsamplingConvChannels
        self.samplingNum = Int(log2(Double(args.subsamplingFactor)).rounded())
        self.padLeft = kernelSize - 1  // CausalConv2D: left = k-1
        self.padRight = stride - 1  //               right = stride-1

        var freq = args.featIn
        for _ in 0 ..< samplingNum {
            freq = (freq + padLeft + padRight - kernelSize) / stride + 1
        }

        var convs: [Conv2d] = [
            Conv2d(
                inputChannels: 1, outputChannels: ch, kernelSize: 3,
                stride: 2, padding: 0)
        ]
        var stridedIdx: Set<Int> = [0]
        var reluAfterIdx: Set<Int> = [0]
        for i in 0 ..< (samplingNum - 1) {
            stridedIdx.insert(convs.count)
            convs.append(
                Conv2d(
                    inputChannels: ch, outputChannels: ch, kernelSize: 3,
                    stride: 2, padding: 0, groups: ch))  // depthwise
            convs.append(
                Conv2d(
                    inputChannels: ch, outputChannels: ch, kernelSize: 1,
                    stride: 1, padding: 0))  // pointwise
            reluAfterIdx.insert(convs.count - 1)
            _ = i
        }
        self.stridedIdx = stridedIdx
        self.reluAfterIdx = reluAfterIdx
        self._convs.wrappedValue = convs
        self._out.wrappedValue = Linear(ch * freq, args.dModel)
        super.init()
    }

    public func calcLength(_ length: Int) -> Int {
        var length = length
        for _ in 0 ..< samplingNum {
            length = (length + padLeft + padRight - kernelSize) / stride + 1
        }
        return length
    }

    public func callAsFunction(_ x: MLXArray, _ lengths: [Int]) -> (MLXArray, [Int]) {
        let outLengths = lengths.map(calcLength)
        // (B, T, F) -> (B, T, F, 1) channels-last for MLX conv2d.
        var x = expandedDimensions(x, axis: -1)
        for (i, layer) in convs.enumerated() {
            if stridedIdx.contains(i) {
                x = padded(
                    x,
                    widths: [
                        IntOrPair((0, 0)),
                        IntOrPair((padLeft, padRight)),  // time
                        IntOrPair((padLeft, padRight)),  // freq
                        IntOrPair((0, 0)),
                    ])
            }
            x = layer(x)
            if reluAfterIdx.contains(i) {
                x = relu(x)
            }
        }
        // (B, T', F', C) -> (B, T', C*F') with channel-major ordering (matches NeMo).
        let (b, t, f, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        x = x.transposed(0, 1, 3, 2).reshaped(b, t, c * f)
        return (out(x), outLengths)
    }
}

public final class Conformer: Module {
    public let args: ConformerArgs
    public let posEnc: RelPositionalEncoding

    @ModuleInfo(key: "pre_encode") public var preEncode: CausalDwStridingSubsampling
    @ModuleInfo(key: "layers") public var layers: [ConformerBlock]

    public init(_ args: ConformerArgs) {
        self.args = args
        self.posEnc = RelPositionalEncoding(
            dModel: args.dModel, maxLen: args.posEmbMaxLen, scaleInput: args.xscaling)
        self._preEncode.wrappedValue = CausalDwStridingSubsampling(args)
        self._layers.wrappedValue = (0 ..< args.nLayers).map { _ in ConformerBlock(args) }
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray, lengths: [Int]? = nil, attContextSize: [Int]? = nil
    ) -> (MLXArray, [Int]) {
        let inLengths = lengths ?? Array(repeating: x.dim(-2), count: x.dim(0))
        var (h, outLengths) = preEncode(x, inLengths)
        let (scaled, posEmb) = posEnc(h)
        h = scaled

        let acs = attContextSize ?? args.attContextSize[0]
        let (leftContext, rightContext) = (acs[0], acs[1])

        var mask: MLXArray? = nil
        if args.attContextStyle == "chunked_limited" {
            mask = createChunkedLimitedMask(
                seqLen: h.dim(1), leftContext: leftContext, rightContext: rightContext
            ).asType(h.dtype)
        }

        for layer in layers {
            h = layer(h, posEmb: posEmb, mask: mask)
        }
        return (h, outLengths)
    }
}
