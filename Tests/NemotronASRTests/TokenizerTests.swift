import XCTest

@testable import NemotronASR

final class TokenizerTests: XCTestCase {
    func testLangTag() {
        XCTAssertTrue(Tokenizer.isLangTag("<en-US>"))
        XCTAssertTrue(Tokenizer.isLangTag("<yue-Hant>"))
        XCTAssertFalse(Tokenizer.isLangTag("<unk>"))
        XCTAssertFalse(Tokenizer.isLangTag("hello"))
    }

    func testDecodeStripsSpecials() {
        let vocab = ["<unk>", "\u{2581}Hello", ",", "\u{2581}world", "<en-US>"]
        XCTAssertEqual(Tokenizer.decode([0, 1, 2, 3, 4], vocabulary: vocab), " Hello, world")
        XCTAssertEqual(Tokenizer.detectedLanguage([0, 4, 1], vocabulary: vocab), "en-US")
    }

    func testSentenceSplit() {
        let tokens = [
            AlignedToken(id: 1, text: " Hi", start: 0.0, duration: 0.08),
            AlignedToken(id: 2, text: ".", start: 0.08, duration: 0.08),
            AlignedToken(id: 3, text: " There", start: 0.16, duration: 0.08),
        ]
        let sentences = tokensToSentences(tokens)
        XCTAssertEqual(sentences.count, 2)
        XCTAssertEqual(sentences[0].text, " Hi.")
        XCTAssertEqual(sentencesToResult(sentences).text, "Hi. There")
    }
}
