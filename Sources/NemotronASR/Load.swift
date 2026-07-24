// Weight/config loading for the Nemotron 3.5 ASR MLX-Swift port.
//
// Loads the mlx-community conversion (config.json + model.safetensors, produced by
// the reference convert.py — all NeMo-side remapping/transposes already applied).
// The only remaining remap is checkpoint-list indices that are ILLEGAL as Swift
// Module keys (dotted numeric indices on heterogeneous lists): see `sanitize`.

import Foundation
import MLX
import MLXNN

public enum NemotronLoad {

    public enum LoadError: Error, CustomStringConvertible {
        case missingFile(String)
        public var description: String {
            switch self {
            case .missingFile(let f): return "missing model file: \(f)"
            }
        }
    }

    /// Remap reference checkpoint keys → Swift Module key paths.
    ///
    /// - `prompt_kernel.{0,2}` (Linear at list indices 0/2, ReLU at 1) → `prompt_kernel_{0,2}`.
    /// - `joint.joint_net.2` (Linear at list index 2 after activation/Identity) → `joint.joint_net_2`.
    /// - `encoder.pre_encode.conv.{0,2,3,5,6}` (convs sparse among ReLUs) → dense `convs.{0..4}`.
    /// Everything else passes through unchanged (LSTM Wx/Wh/bias, conv layouts already MLX).
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        // Dense re-indexing for the pre_encode conv list: sorted original indices.
        let convIndices = weights.keys
            .compactMap { key -> Int? in
                guard key.hasPrefix("encoder.pre_encode.conv.") else { return nil }
                return Int(key.split(separator: ".")[3])
            }
        let sortedConv = Set(convIndices).sorted()
        var convRemap: [Int: Int] = [:]
        for (dense, original) in sortedConv.enumerated() { convRemap[original] = dense }

        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)
        for (key, value) in weights {
            var key = key
            if key.hasPrefix("prompt_kernel.") {
                key = key
                    .replacingOccurrences(of: "prompt_kernel.0.", with: "prompt_kernel_0.")
                    .replacingOccurrences(of: "prompt_kernel.2.", with: "prompt_kernel_2.")
            } else if key.hasPrefix("joint.joint_net.") {
                key = key.replacingOccurrences(of: "joint.joint_net.2.", with: "joint.joint_net_2.")
            } else if key.hasPrefix("encoder.pre_encode.conv.") {
                var parts = key.split(separator: ".").map(String.init)
                if let original = Int(parts[3]), let dense = convRemap[original] {
                    parts[2] = "convs"
                    parts[3] = String(dense)
                    key = parts.joined(separator: ".")
                }
            }
            out[key] = value
        }
        return out
    }

    /// Load the model from a directory containing config.json + *.safetensors.
    ///
    /// If the checkpoint carries `quantization` metadata (mlx-community `-8bit` repos) OR
    /// per-Linear `.scales` tensors are present, the matching layers are quantized before the
    /// weights are applied. `verify` relaxes to `.none` for quantized checkpoints (the quantized
    /// Linear swaps the parameter set, so a strict all-keys match no longer holds).
    public static func load(directory: URL) throws -> NemotronASRModel {
        let configURL = directory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw LoadError.missingFile(configURL.path)
        }
        let config = try NemotronASRConfig.load(from: configURL)
        let model = NemotronASRModel(config: config)

        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "safetensors" }
        guard !files.isEmpty else {
            throw LoadError.missingFile("\(directory.path)/*.safetensors")
        }

        var weights: [String: MLXArray] = [:]
        for file in files.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            for (key, value) in try MLX.loadArrays(url: file) {
                weights[key] = value
            }
        }
        weights = sanitize(weights)

        // Quantized checkpoint? Apply the same quantization to the module tree so shapes line up.
        let isQuantized = weights.keys.contains { $0.hasSuffix(".scales") }
        if isQuantized, let q = quantMetadata(configURL: configURL) {
            quantize(model: model, groupSize: q.groupSize, bits: q.bits) { path, module in
                module is MLXNN.Linear && weights["\(path).scales"] != nil
            }
        }

        try model.update(
            parameters: ModuleParameters.unflattened(weights),
            verify: isQuantized ? [] : [.all])
        eval(model.parameters())
        return model
    }

    private static func quantMetadata(configURL: URL) -> (groupSize: Int, bits: Int)? {
        guard let data = try? Data(contentsOf: configURL),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let q = json["quantization"] as? [String: Any]
        else { return nil }
        let groupSize = (q["group_size"] as? Int) ?? 64
        let bits = (q["bits"] as? Int) ?? 8
        return (groupSize, bits)
    }
}
