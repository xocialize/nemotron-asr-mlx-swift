// Aligned transcription results.
// Mirrors mlx_audio/stt/models/nemo/alignment.py (the subset Nemotron uses):
// AlignedToken/AlignedSentence/AlignedResult + tokens_to_sentences/sentences_to_result.
// The sentence-split heuristic is ported verbatim (including its quirks) — the
// transcript golden gate compares segment text and timing against the reference.

import Foundation

public struct AlignedToken: Sendable, Equatable {
    public let id: Int
    public let text: String
    public let start: Double
    public let duration: Double
    public var end: Double { start + duration }

    public init(id: Int, text: String, start: Double, duration: Double) {
        self.id = id
        self.text = text
        self.start = start
        self.duration = duration
    }
}

public struct AlignedSentence: Sendable, Equatable {
    public let text: String
    public let tokens: [AlignedToken]
    public var start: Double { tokens.first?.start ?? 0 }
    public var end: Double { tokens.last?.end ?? 0 }
    public var duration: Double { end - start }

    public init(text: String, tokens: [AlignedToken]) {
        // Reference sorts tokens by start time on construction.
        self.tokens = tokens.sorted { $0.start < $1.start }
        self.text = text
    }
}

public struct AlignedResult: Sendable, Equatable {
    public let text: String
    public let sentences: [AlignedSentence]

    public init(text: String, sentences: [AlignedSentence]) {
        self.text = text.trimmingCharacters(in: .whitespaces)
        self.sentences = sentences
    }
}

/// Sentence-split on terminal punctuation ("hacky, will fix" upstream — kept verbatim).
public func tokensToSentences(_ tokens: [AlignedToken]) -> [AlignedSentence] {
    var sentences: [AlignedSentence] = []
    var current: [AlignedToken] = []

    for (idx, token) in tokens.enumerated() {
        current.append(token)
        let isSentenceEnd =
            token.text.contains("!")
            || token.text.contains("?")
            || token.text.contains("。")
            || token.text.contains("？")
            || token.text.contains("！")
            || (token.text.contains(".")
                && (idx == tokens.count - 1 || tokens[idx + 1].text.contains(" ")))
        if isSentenceEnd {
            let text = current.map(\.text).joined()
            sentences.append(AlignedSentence(text: text, tokens: current))
            current = []
        }
    }
    if !current.isEmpty {
        let text = current.map(\.text).joined()
        sentences.append(AlignedSentence(text: text, tokens: current))
    }
    return sentences
}

public func sentencesToResult(_ sentences: [AlignedSentence]) -> AlignedResult {
    AlignedResult(text: sentences.map(\.text).joined(), sentences: sentences)
}
