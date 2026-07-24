// Cache-aware streaming for the Nemotron FastConformer encoder.
// Mirrors mlx_audio/stt/models/nemotron_asr/streaming.py.
//
// Each conformer layer keeps an attention cache (last `leftContext` attention-input
// frames, i.e. the NORMALIZED pre-attention window) and a causal-conv cache (last
// `convKernel-1` GLU-output frames); subsampling is incremental with a small mel
// cache. With the window sized to the allowed left context, no attention mask is
// needed — the streamed encoder output is frame-identical to the offline
// `chunked_limited` encoder at the native chunk size (`rightContext + 1`).

import Foundation
import MLX
import MLXNN

/// >= causal receptive field of the 8x dw-striding stack.
let preEncodeMelCache = 16

/// One cache-aware conformer-block step. Returns (h, attnNext, convNext).
/// Public for the parity-gate CLI; production callers go through StreamEncoder.
public func streamBlock(
    _ block: ConformerBlock,
    _ x: MLXArray,
    _ posEnc: RelPositionalEncoding,
    _ attnCache: MLXArray?,
    _ convCache: MLXArray?,
    leftCache: Int,
    convLeft: Int
) -> (MLXArray, MLXArray, MLXArray) {
    // half-step FFN 1
    var residual = x + 0.5 * block.feedForward1(block.normFeedForward1(x))

    // cache-aware self-attention: Q = chunk, K/V = [cache ++ chunk]
    let xn = block.normSelfAtt(residual)
    let kv = attnCache.map { concatenated([$0, xn], axis: 1) } ?? xn
    let posEmb = posEnc.posEmb(for: kv.dim(1), dtype: x.dtype)
    residual = residual + block.selfAttn.stream(xn, kv, posEmb: posEmb)
    let attnNext = leftCache > 0 ? kv[0..., (kv.dim(1) - min(leftCache, kv.dim(1)))...] : kv[0..., 0 ..< 0]

    // cache-aware causal conv: prepend conv cache instead of zero-padding
    let xc = block.normConv(residual)
    let g = glu(block.conv.pointwiseConv1(xc), axis: -1)  // (B, c, d)
    let convCacheResolved =
        convCache ?? MLXArray.zeros([g.dim(0), convLeft, g.dim(2)], dtype: g.dtype)
    let din = concatenated([convCacheResolved, g], axis: 1)
    let dw = block.conv.depthwiseConv(din)  // valid conv -> (B, c, d)
    let convNext = din[0..., (din.dim(1) - convLeft)...]
    var y = block.conv.batchNorm(dw)
    y = block.conv.activation(y)
    residual = residual + block.conv.pointwiseConv2(y)

    // half-step FFN 2 + final norm
    residual = residual + 0.5 * block.feedForward2(block.normFeedForward2(residual))
    return (block.normOut(residual), attnNext, convNext)
}

/// Streams mel chunks through the cache-aware encoder, yielding post-prompt encoder
/// frames per native chunk. The encoder/conv/subsampling caches persist across input
/// mel chunks, so callers can keep STFT memory bounded without resetting model
/// context at chunk boundaries.
///
/// Swift shape of the reference's generator: feed mel chunks via `push`, then call
/// `finish`; prompted chunks are delivered to `onPromptedChunk` as they complete.
final class StreamEncoder {
    private let model: NemotronASRModel
    private let language: String?
    private let leftCache: Int
    /// Mel frames per native encode chunk (`chunkFrames × subsamplingFactor`).
    let chunkMel: Int
    private let convLeft: Int
    private let subsamplingFactor: Int

    private var attnCache: [MLXArray?]
    private var convCache: [MLXArray?]
    private var melCache: MLXArray?
    private var emitted = 0
    private var consumed = 0
    private var pending: MLXArray?

    init(
        model: NemotronASRModel,
        language: String?,
        chunkFrames: Int? = nil,
        attContextSize: [Int]? = nil
    ) {
        self.model = model
        self.language = language
        let acs = attContextSize ?? model.defaultAttContextSize
        self.leftCache = acs[0]
        let right = acs[1]
        let cf = chunkFrames ?? (right + 1)
        self.subsamplingFactor = model.encoder.args.subsamplingFactor
        self.chunkMel = cf * subsamplingFactor
        self.convLeft = model.encoder.args.convKernelSize - 1
        let n = model.encoder.layers.count
        self.attnCache = Array(repeating: nil, count: n)
        self.convCache = Array(repeating: nil, count: n)
    }

    private func appendPending(_ chunk: MLXArray) {
        var chunk = chunk
        if chunk.ndim == 2 {
            chunk = expandedDimensions(chunk, axis: 0)
        }
        if chunk.dim(1) == 0 { return }
        pending = pending.map { concatenated([$0, chunk], axis: 1) } ?? chunk
    }

    private func encodeMelChunk(
        _ m: MLXArray, isFinal: Bool, _ onPromptedChunk: (MLXArray) throws -> Void
    ) rethrows {
        let enc = model.encoder
        let cacheLen = melCache?.dim(1) ?? 0
        let win = melCache.map { concatenated([$0, m], axis: 1) } ?? m
        let winLen = win.dim(1)
        let (sub, _) = enc.preEncode(win, [winLen])  // (1, k, d)

        let sf = subsamplingFactor
        let end = consumed + m.dim(1)
        let base = (consumed - cacheLen) / sf
        let lo = emitted - base
        let hi = isFinal ? sub.dim(1) : (end / sf - base)
        consumed = end
        melCache = win[0..., max(0, win.dim(1) - preEncodeMelCache)...]

        if hi <= lo {
            emitted = base + max(lo, hi)
            return
        }
        emitted = base + hi
        var h = sub[0..., lo ..< hi]
        for li in 0 ..< enc.layers.count {
            let (hNext, attnNext, convNext) = streamBlock(
                enc.layers[li], h, enc.posEnc, attnCache[li], convCache[li],
                leftCache: leftCache, convLeft: convLeft)
            h = hNext
            attnCache[li] = attnNext
            convCache[li] = convNext
        }
        // Materialize the caches so a long stream carries state, not a growing graph.
        eval(attnCache.compactMap { $0 } + convCache.compactMap { $0 } + [melCache].compactMap { $0 })
        try onPromptedChunk(model.applyPrompt(h, language: language))
    }

    private func encodeReady(
        isFinal: Bool, _ onPromptedChunk: (MLXArray) throws -> Void
    ) rethrows {
        while let p = pending, p.dim(1) > 0 {
            if p.dim(1) < chunkMel && !isFinal { break }

            var take = min(chunkMel, p.dim(1))
            if isFinal && p.dim(1) <= chunkMel {
                take = p.dim(1)
            }

            let m = p[0..., 0 ..< take]
            pending = p[0..., take...]
            let isFinalChunk = isFinal && pending?.dim(1) == 0
            try encodeMelChunk(m, isFinal: isFinalChunk, onPromptedChunk)
        }
    }

    /// Feed one mel chunk (1, T, F) or (T, F).
    func push(_ melChunk: MLXArray, _ onPromptedChunk: (MLXArray) throws -> Void) rethrows {
        appendPending(melChunk)
        try encodeReady(isFinal: false, onPromptedChunk)
    }

    /// Flush the tail once all mel chunks have been pushed.
    func finish(_ onPromptedChunk: (MLXArray) throws -> Void) rethrows {
        try encodeReady(isFinal: true, onPromptedChunk)
    }
}
