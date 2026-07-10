// Vocabulary helpers for the Nemotron multilingual BPE tokenizer.
// Mirrors mlx_audio/stt/models/nemotron_asr/tokenizer.py.
//
// The vocabulary is the flat list of SentencePiece pieces from config.json. The
// model can emit a leading language tag such as `<en-US>` (especially in `auto`
// mode); `decode(stripLangTags:)` removes these from the final text.

import Foundation

public enum Tokenizer {
    // ^<[a-z]{2,3}-[A-Za-z]{2,4}>$
    private static let langTagRegex = try! NSRegularExpression(
        pattern: "^<[a-z]{2,3}-[A-Za-z]{2,4}>$"
    )
    private static let otherSpecial: Set<String> = ["<unk>", "<pad>", "<s>", "</s>"]

    public static func isLangTag(_ piece: String) -> Bool {
        let range = NSRange(piece.startIndex..., in: piece)
        return langTagRegex.firstMatch(in: piece, range: range) != nil
    }

    public static func isSpecialPiece(_ piece: String) -> Bool {
        otherSpecial.contains(piece) || isLangTag(piece)
    }

    public static func isSpecialToken(_ tokenID: Int, vocabulary: [String]) -> Bool {
        guard tokenID >= 0, tokenID < vocabulary.count else { return false }
        return isSpecialPiece(vocabulary[tokenID])
    }

    public static func pieceToText(_ piece: String) -> String {
        piece.replacingOccurrences(of: "\u{2581}", with: " ")  // ▁ -> space
    }

    public static func decode(
        _ tokens: [Int], vocabulary: [String], stripLangTags: Bool = true
    ) -> String {
        var parts: [String] = []
        for token in tokens {
            guard token >= 0, token < vocabulary.count else { continue }
            let piece = vocabulary[token]
            if otherSpecial.contains(piece) { continue }
            if stripLangTags && isLangTag(piece) { continue }
            parts.append(pieceToText(piece))
        }
        return parts.joined()
    }

    /// The first emitted language tag (e.g. `en-US`) if any.
    public static func detectedLanguage(_ tokens: [Int], vocabulary: [String]) -> String? {
        for token in tokens where token >= 0 && token < vocabulary.count {
            let piece = vocabulary[token]
            if isLangTag(piece) {
                return String(piece.dropFirst().dropLast())
            }
        }
        return nil
    }
}
