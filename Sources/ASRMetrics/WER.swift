// Word error rate with word-level alignment.
//
// Standalone by design: no MLX, no model, nothing from NemotronASR. It grades any pair of
// transcripts, so it serves ASR gates, TTS round-trip checks (synthesize → transcribe →
// score against the input text), and UI that colors each word by what happened to it.
//
// The normalization is the usual ASR-benchmark convention (lowercase, drop punctuation but
// keep intra-word apostrophes, collapse whitespace), matching what NeMo and the reference
// demos score with.

import Foundation

/// What happened to one word when the hypothesis is aligned against the reference.
public enum EditOp: Sendable, Equatable {
    case match(ref: String, hyp: String)
    case substitution(ref: String, hyp: String)
    /// A reference word the hypothesis missed.
    case deletion(ref: String)
    /// A hypothesis word with no reference counterpart.
    case insertion(hyp: String)
}

/// A scored alignment of one hypothesis against one reference.
public struct WERResult: Sendable, Equatable {
    public let substitutions: Int
    public let deletions: Int
    public let insertions: Int
    /// Words in the normalized reference — the denominator.
    public let referenceCount: Int
    /// The alignment in reference order, for display.
    public let ops: [EditOp]

    public init(
        substitutions: Int, deletions: Int, insertions: Int, referenceCount: Int, ops: [EditOp]
    ) {
        self.substitutions = substitutions
        self.deletions = deletions
        self.insertions = insertions
        self.referenceCount = referenceCount
        self.ops = ops
    }

    public var errors: Int { substitutions + deletions + insertions }

    /// (S + D + I) / N. An empty reference scores 0 with no hypothesis and 1 otherwise,
    /// rather than dividing by zero.
    public var wer: Double {
        guard referenceCount > 0 else { return insertions > 0 ? 1 : 0 }
        return Double(errors) / Double(referenceCount)
    }

    /// `wer` as a percentage.
    public var werPercent: Double { wer * 100 }

    public var matches: Int { referenceCount - substitutions - deletions }
}

public enum ASRMetrics {

    /// Lowercase, drop punctuation except intra-word apostrophes, collapse whitespace.
    public static func normalize(_ text: String) -> [String] {
        text.lowercased()
            .replacingOccurrences(of: "\u{2019}", with: "'")  // ’ → '
            .split(whereSeparator: { $0.isWhitespace })
            .map { word in
                String(word.filter { $0.isLetter || $0.isNumber || $0 == "'" })
            }
            .map { trimApostrophes($0) }
            .filter { !$0.isEmpty }
    }

    /// Drop apostrophes that quote rather than elide: `'stop'` → `stop`, `dogs'` → `dogs`,
    /// while `'em` and `don't` keep theirs. A leading apostrophe only goes when the word
    /// closed with one too, which is what separates a quotation from an elision.
    private static func trimApostrophes(_ word: String) -> String {
        var out = Substring(word)
        let wasQuoted = out.count > 2 && out.hasPrefix("'") && out.hasSuffix("'")
        while out.count > 1, out.hasSuffix("'") { out = out.dropLast() }
        if wasQuoted, out.count > 1, out.hasPrefix("'") { out = out.dropFirst() }
        return String(out)
    }

    /// Levenshtein alignment of two word sequences with unit costs.
    ///
    /// Ties prefer substitution over deletion over insertion, so the alignment reads in
    /// reference order and diagonal moves are favored — the conventional ASR presentation.
    public static func align(reference: [String], hypothesis: [String]) -> WERResult {
        let n = reference.count
        let m = hypothesis.count

        // cost[i][j] = edit distance between reference[0..<i] and hypothesis[0..<j].
        var cost = Array(repeating: Array(repeating: 0, count: m + 1), count: n + 1)
        for i in stride(from: 1, through: n, by: 1) { cost[i][0] = i }
        for j in stride(from: 1, through: m, by: 1) { cost[0][j] = j }
        for i in stride(from: 1, through: n, by: 1) {
            for j in stride(from: 1, through: m, by: 1) {
                let diagonal = cost[i - 1][j - 1] + (reference[i - 1] == hypothesis[j - 1] ? 0 : 1)
                cost[i][j] = min(diagonal, cost[i - 1][j] + 1, cost[i][j - 1] + 1)
            }
        }

        var ops: [EditOp] = []
        var substitutions = 0, deletions = 0, insertions = 0
        var i = n, j = m
        while i > 0 || j > 0 {
            if i > 0, j > 0,
                cost[i][j] == cost[i - 1][j - 1] + (reference[i - 1] == hypothesis[j - 1] ? 0 : 1)
            {
                if reference[i - 1] == hypothesis[j - 1] {
                    ops.append(.match(ref: reference[i - 1], hyp: hypothesis[j - 1]))
                } else {
                    ops.append(.substitution(ref: reference[i - 1], hyp: hypothesis[j - 1]))
                    substitutions += 1
                }
                i -= 1
                j -= 1
            } else if i > 0, cost[i][j] == cost[i - 1][j] + 1 {
                ops.append(.deletion(ref: reference[i - 1]))
                deletions += 1
                i -= 1
            } else {
                ops.append(.insertion(hyp: hypothesis[j - 1]))
                insertions += 1
                j -= 1
            }
        }

        return WERResult(
            substitutions: substitutions, deletions: deletions, insertions: insertions,
            referenceCount: n, ops: ops.reversed())
    }

    /// Normalize both sides, then align and score.
    public static func wordErrorRate(reference: String, hypothesis: String) -> WERResult {
        align(reference: normalize(reference), hypothesis: normalize(hypothesis))
    }
}
