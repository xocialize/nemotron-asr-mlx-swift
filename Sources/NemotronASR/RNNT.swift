// RNN-T prediction (decoder) and joint networks.
// Mirrors mlx_audio/stt/models/nemotron_asr/rnnt.py; layout mirrors NeMo's
// RNNTDecoder/RNNTJoint so checkpoint keys map directly:
// `decoder.prediction.embed`, `decoder.prediction.dec_rnn.lstm.{i}`,
// `joint.enc`, `joint.pred`, `joint.joint_net.2` (remapped, see NemotronLoad).

import Foundation
import MLX
import MLXNN

/// One LSTM layer with MLX-Python `nn.LSTM` parameter layout and gate order:
/// Wx (4H, in), Wh (4H, H), bias (4H); gates split as i, f, g, o.
/// Hand-rolled (rather than MLXNN.LSTM) so the keys and semantics are pinned to
/// the reference checkpoint exactly.
public final class LSTMLayer: Module {
    @ParameterInfo(key: "Wx") var wx: MLXArray
    @ParameterInfo(key: "Wh") var wh: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray

    let hiddenSize: Int

    public init(inputSize: Int, hiddenSize: Int) {
        self.hiddenSize = hiddenSize
        let scale = 1.0 / sqrt(Float(hiddenSize))
        self._wx.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale, [4 * hiddenSize, inputSize])
        self._wh.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale, [4 * hiddenSize, hiddenSize])
        self._bias.wrappedValue = MLXRandom.uniform(low: -scale, high: scale, [4 * hiddenSize])
        super.init()
    }

    /// x: (..., L, inputSize) scanned over axis -2 (mirrors mlx.nn.LSTM).
    /// Returns (allH, allC) stacked over the scan axis.
    public func callAsFunction(
        _ x: MLXArray, hidden: MLXArray? = nil, cell: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        let z = addMM(bias, x, wx.transposed())
        let steps = z.dim(-2)

        var h = hidden
        var c = cell
        var allH: [MLXArray] = []
        var allC: [MLXArray] = []

        for t in 0 ..< steps {
            var zt = z[.ellipsis, t, 0...]
            if let hPrev = h {
                zt = addMM(zt, hPrev, wh.transposed())
            }
            let parts = split(zt, parts: 4, axis: -1)
            let i = sigmoid(parts[0])
            let f = sigmoid(parts[1])
            let g = tanh(parts[2])
            let o = sigmoid(parts[3])
            let cNext = (c.map { f * $0 } ?? MLXArray.zeros(like: g)) + i * g
            let hNext = o * tanh(cNext)
            h = hNext
            c = cNext
            allH.append(hNext)
            allC.append(cNext)
        }
        return (stacked(allH, axis: -2), stacked(allC, axis: -2))
    }
}

/// Multi-layer LSTM holding one layer per entry under `lstm` (keys lstm.0, lstm.1).
public final class LSTMStack: Module {
    @ModuleInfo(key: "lstm") var lstm: [LSTMLayer]
    let numLayers: Int

    public init(inputSize: Int, hiddenSize: Int, numLayers: Int) {
        self.numLayers = numLayers
        self._lstm.wrappedValue = (0 ..< numLayers).map { i in
            LSTMLayer(inputSize: i == 0 ? inputSize : hiddenSize, hiddenSize: hiddenSize)
        }
        super.init()
    }

    /// x: (B, L, D). Returns (output (B, L, H), (h, c)) with h,c (numLayers, B, H).
    public func callAsFunction(
        _ x: MLXArray, _ hc: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        // Mirrors the reference: transpose to time-major, scan, transpose back.
        var outputs = x.transposed(1, 0, 2)
        var nextH: [MLXArray] = []
        var nextC: [MLXArray] = []
        for i in 0 ..< numLayers {
            let (allH, allC) = lstm[i](
                outputs,
                hidden: hc.map { $0.0[i] },
                cell: hc.map { $0.1[i] })
            outputs = allH
            nextH.append(allH[.ellipsis, -1, 0...])
            nextC.append(allC[.ellipsis, -1, 0...])
        }
        return (
            outputs.transposed(1, 0, 2),
            (stacked(nextH, axis: 0), stacked(nextC, axis: 0))
        )
    }
}

public final class PredictNetwork: Module {
    public final class Prediction: Module {
        @ModuleInfo(key: "embed") var embed: Embedding
        @ModuleInfo(key: "dec_rnn") var decRnn: LSTMStack

        init(_ args: PredictArgs) {
            let vocab = args.blankAsPad ? args.vocabSize + 1 : args.vocabSize
            self._embed.wrappedValue = Embedding(
                embeddingCount: vocab, dimensions: args.predHidden)
            self._decRnn.wrappedValue = LSTMStack(
                inputSize: args.predHidden, hiddenSize: args.predHidden,
                numLayers: args.predRnnLayers)
            super.init()
        }
    }

    let predHidden: Int
    @ModuleInfo(key: "prediction") var prediction: Prediction

    public init(_ args: PredictArgs) {
        self.predHidden = args.predHidden
        self._prediction.wrappedValue = Prediction(args)
        super.init()
    }

    /// y: (B, 1) previous non-blank token, or nil for the blank/pad zero embedding.
    public func callAsFunction(
        _ y: MLXArray?, _ hc: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let embedded: MLXArray
        if let y {
            embedded = prediction.embed(y)
        } else {
            let batch = hc?.0.dim(1) ?? 1
            embedded = MLXArray.zeros([batch, 1, predHidden])
        }
        return prediction.decRnn(embedded, hc)
    }
}

public final class JointNetwork: Module {
    let numClasses: Int

    @ModuleInfo(key: "enc") var enc: Linear
    @ModuleInfo(key: "pred") var pred: Linear
    /// Reference list [activation, Identity, Linear] → key joint_net.2; the loader
    /// remaps to joint_net_2 (dots are illegal in Swift Module keys).
    @ModuleInfo(key: "joint_net_2") var jointNetOut: Linear

    public init(_ args: JointArgs) {
        self.numClasses = args.numClasses + 1  // + blank
        precondition(
            args.activation.lowercased() == "relu",
            "joint activation \(args.activation) not supported (checkpoint uses relu)")
        self._enc.wrappedValue = Linear(args.encoderHidden, args.jointHidden)
        self._pred.wrappedValue = Linear(args.predHidden, args.jointHidden)
        self._jointNetOut.wrappedValue = Linear(args.jointHidden, numClasses)
        super.init()
    }

    public func callAsFunction(_ encIn: MLXArray, _ predIn: MLXArray) -> MLXArray {
        let e = enc(encIn)
        let p = pred(predIn)
        var x = expandedDimensions(e, axis: 2) + expandedDimensions(p, axis: 1)
        x = relu(x)
        return jointNetOut(x)
    }
}
