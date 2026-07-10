// Relative-position multi-head attention for the FastConformer encoder.
// Mirrors mlx_audio/stt/models/nemotron_asr/attention.py (NeMo's
// RelPositionMultiHeadAttention, Transformer-XL style) — additive attention mask,
// `use_bias=false` projections, plus the cache-aware `stream` step.

import Foundation
import MLX
import MLXFast
import MLXNN

public final class RelPositionMultiHeadAttention: Module {
    let nHead: Int
    let nFeat: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "linear_q") var linearQ: Linear
    @ModuleInfo(key: "linear_k") var linearK: Linear
    @ModuleInfo(key: "linear_v") var linearV: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear
    @ModuleInfo(key: "linear_pos") var linearPos: Linear

    // Per-layer (untied) position biases; populated from the checkpoint.
    @ParameterInfo(key: "pos_bias_u") var posBiasU: MLXArray
    @ParameterInfo(key: "pos_bias_v") var posBiasV: MLXArray

    public init(nHead: Int, nFeat: Int, bias: Bool = false) {
        precondition(nFeat % nHead == 0)
        self.nHead = nHead
        self.nFeat = nFeat
        self.headDim = nFeat / nHead
        self.scale = pow(Float(headDim), -0.5)

        self._linearQ.wrappedValue = Linear(nFeat, nFeat, bias: bias)
        self._linearK.wrappedValue = Linear(nFeat, nFeat, bias: bias)
        self._linearV.wrappedValue = Linear(nFeat, nFeat, bias: bias)
        self._linearOut.wrappedValue = Linear(nFeat, nFeat, bias: bias)
        self._linearPos.wrappedValue = Linear(nFeat, nFeat, bias: false)

        self._posBiasU.wrappedValue = MLXArray.zeros([nHead, headDim])
        self._posBiasV.wrappedValue = MLXArray.zeros([nHead, headDim])
        super.init()
    }

    func relShift(_ x: MLXArray) -> MLXArray {
        let (b, h, tq, posLen) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        var x = padded(
            x,
            widths: [IntOrPair((0, 0)), IntOrPair((0, 0)), IntOrPair((0, 0)), IntOrPair((1, 0))])
        x = x.reshaped(b, h, posLen + 1, tq)
        x = x[0..., 0..., 1..., 0...]
        return x.reshaped(b, h, tq, posLen)
    }

    public func callAsFunction(
        _ x: MLXArray, posEmb: MLXArray, mask: MLXArray? = nil
    ) -> MLXArray {
        var q = linearQ(x)
        let kIn = linearK(x)
        let vIn = linearV(x)
        let p = linearPos(posEmb)

        let (batch, seq) = (q.dim(0), q.dim(1))
        let posLen = p.dim(1)

        q = q.reshaped(batch, seq, nHead, headDim)
        let qU = (q + posBiasU).transposed(0, 2, 1, 3)
        let qV = (q + posBiasV).transposed(0, 2, 1, 3)

        let k = kIn.reshaped(batch, seq, nHead, headDim).transposed(0, 2, 1, 3)
        let v = vIn.reshaped(batch, seq, nHead, headDim).transposed(0, 2, 1, 3)
        let pHeads = p.reshaped(batch, posLen, nHead, headDim).transposed(0, 2, 1, 3)

        // matrix_bd: position-dependent term, scaled and rel-shifted to (B, H, Tq, Tk).
        var matrixBD = matmul(qV, pHeads.swappedAxes(-2, -1))
        matrixBD = relShift(matrixBD)
        matrixBD = matrixBD[0..., 0..., 0..., 0 ..< k.dim(-2)] * scale

        if let mask {
            // Additive (0 visible / −inf blocked), applied at the same post-scale
            // point SDPA adds its mask.
            matrixBD = matrixBD + mask
        }

        let o = MLXFast.scaledDotProductAttention(
            queries: qU, keys: k, values: v, scale: scale, mask: matrixBD)
        return linearOut(o.transposed(0, 2, 1, 3).reshaped(batch, seq, nFeat))
    }

    /// Cache-aware step: qIn (B,c,d) attends to kvIn (B,L,d), no mask (the L-window
    /// IS the allowed context). posEmb is for length L (2L−1).
    public func stream(_ qIn: MLXArray, _ kvIn: MLXArray, posEmb: MLXArray) -> MLXArray {
        var q = linearQ(qIn)
        let kIn = linearK(kvIn)
        let vIn = linearV(kvIn)
        let p = linearPos(posEmb)

        let (batch, c) = (q.dim(0), q.dim(1))
        let ksz = kvIn.dim(1)
        let posLen = p.dim(1)

        q = q.reshaped(batch, c, nHead, headDim)
        let qU = (q + posBiasU).transposed(0, 2, 1, 3)
        let qV = (q + posBiasV).transposed(0, 2, 1, 3)
        let k = kIn.reshaped(batch, ksz, nHead, headDim).transposed(0, 2, 1, 3)
        let v = vIn.reshaped(batch, ksz, nHead, headDim).transposed(0, 2, 1, 3)
        let pHeads = p.reshaped(1, posLen, nHead, headDim).transposed(0, 2, 1, 3)

        var matrixBD = matmul(qV, pHeads.swappedAxes(-2, -1))
        matrixBD = relShift(matrixBD)[0..., 0..., 0..., 0 ..< ksz] * scale

        let o = MLXFast.scaledDotProductAttention(
            queries: qU, keys: k, values: v, scale: scale, mask: matrixBD)
        return linearOut(o.transposed(0, 2, 1, 3).reshaped(batch, c, -1))
    }
}

/// Transformer-XL relative positional encoding table. `_pe` is NOT a parameter
/// (computed, never loaded), mirroring the reference's underscore-prefixed buffer.
public final class RelPositionalEncoding {
    let dModel: Int
    var maxLen: Int
    let scale: Float
    private var pe: MLXArray

    public init(dModel: Int, maxLen: Int = 5000, scaleInput: Bool = false) {
        precondition(dModel % 2 == 0 && maxLen > 0)
        self.dModel = dModel
        self.maxLen = maxLen
        self.scale = scaleInput ? sqrt(Float(dModel)) : 1.0
        self.pe = Self.calculatePE(dModel: dModel, maxLen: maxLen)
    }

    static func calculatePE(dModel: Int, maxLen: Int) -> MLXArray {
        // positions: maxLen-1, ..., 0, ..., -(maxLen-1)  → (2*maxLen-1, 1)
        let positions = expandedDimensions(
            MLXArray(stride(from: Int32(maxLen - 1), through: Int32(-(maxLen - 1)), by: -1).map { $0 }),
            axis: 1
        ).asType(.float32)
        let divTerm = exp(
            MLXArray(stride(from: 0, to: Int32(dModel), by: 2).map { Float($0) })
                * (-logf(10000.0) / Float(dModel)))
        // Interleave sin/cos: pe[:, 0::2] = sin, pe[:, 1::2] = cos.
        let angles = positions * divTerm  // (2*maxLen-1, dModel/2)
        let interleaved = stacked([sin(angles), cos(angles)], axis: -1)
            .reshaped(2 * maxLen - 1, dModel)
        let pe = expandedDimensions(interleaved, axis: 0).asType(.float32)
        eval(pe)
        return pe
    }

    private func ensure(maxLen newLen: Int) {
        if newLen > maxLen {
            maxLen = newLen + 1
            pe = Self.calculatePE(dModel: dModel, maxLen: maxLen)
        }
    }

    public func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let inputLen = x.dim(1)
        ensure(maxLen: inputLen)
        let x = x * scale
        let bufferLen = pe.dim(1)
        let startIdx = bufferLen / 2 - (inputLen - 1)
        let endIdx = bufferLen / 2 + (inputLen - 1) + 1
        let posEmb = pe[0..., startIdx ..< endIdx].asType(x.dtype)
        return (x, posEmb)
    }

    /// Positional embedding for a window of `length` frames (2*length−1). Used by
    /// cache-aware streaming where queries attend to a longer key window.
    public func posEmb(for length: Int, dtype: DType = .float32) -> MLXArray {
        ensure(maxLen: length)
        let center = pe.dim(1) / 2
        return pe[0..., (center - (length - 1)) ..< (center + length)].asType(dtype)
    }
}
