import XCTest

@testable import ASRMetrics

final class WERTests: XCTestCase {

    func testNormalizeFoldsCasePunctuationAndWhitespace() {
        XCTAssertEqual(
            ASRMetrics.normalize("  Hello,   WORLD!  "),
            ["hello", "world"])
        XCTAssertEqual(ASRMetrics.normalize("It's a dog's life."), ["it's", "a", "dog's", "life"])
        // Curly apostrophes fold to straight ones, quoting apostrophes are dropped.
        XCTAssertEqual(ASRMetrics.normalize("don\u{2019}t 'stop' 'em"), ["don't", "stop", "'em"])
        XCTAssertEqual(ASRMetrics.normalize("   "), [])
    }

    func testIdenticalTranscriptsScoreZero() {
        let r = ASRMetrics.wordErrorRate(
            reference: "the quick brown fox", hypothesis: "the quick brown fox")
        XCTAssertEqual(r.wer, 0)
        XCTAssertEqual(r.errors, 0)
        XCTAssertEqual(r.referenceCount, 4)
        XCTAssertTrue(r.ops.allSatisfy { if case .match = $0 { return true } else { return false } })
    }

    func testScoringIsCaseAndPunctuationInvariant() {
        let r = ASRMetrics.wordErrorRate(
            reference: "The quick, brown fox!", hypothesis: "the QUICK brown fox")
        XCTAssertEqual(r.wer, 0)
    }

    func testCountsSubstitutionDeletionAndInsertion() {
        // ref: the quick brown fox jumps
        // hyp: the quick red    fox high jumps   → 1 sub (brown→red), 1 ins (high)
        let r = ASRMetrics.wordErrorRate(
            reference: "the quick brown fox jumps",
            hypothesis: "the quick red fox high jumps")
        XCTAssertEqual(r.substitutions, 1)
        XCTAssertEqual(r.insertions, 1)
        XCTAssertEqual(r.deletions, 0)
        XCTAssertEqual(r.referenceCount, 5)
        XCTAssertEqual(r.wer, 2.0 / 5.0, accuracy: 1e-12)

        // A dropped word is a deletion.
        let d = ASRMetrics.wordErrorRate(
            reference: "the quick brown fox", hypothesis: "the brown fox")
        XCTAssertEqual(d.deletions, 1)
        XCTAssertEqual(d.substitutions, 0)
        XCTAssertEqual(d.insertions, 0)
        XCTAssertEqual(d.wer, 0.25, accuracy: 1e-12)
    }

    func testEmptyHypothesisIsAllDeletions() {
        let r = ASRMetrics.wordErrorRate(reference: "one two three", hypothesis: "")
        XCTAssertEqual(r.deletions, 3)
        XCTAssertEqual(r.wer, 1.0, accuracy: 1e-12)
    }

    func testEmptyReferenceDoesNotDivideByZero() {
        let empty = ASRMetrics.wordErrorRate(reference: "", hypothesis: "")
        XCTAssertEqual(empty.referenceCount, 0)
        XCTAssertEqual(empty.wer, 0)

        let spurious = ASRMetrics.wordErrorRate(reference: "", hypothesis: "hello there")
        XCTAssertEqual(spurious.insertions, 2)
        XCTAssertEqual(spurious.wer, 1.0)
    }

    /// The ops are a complete description of the alignment: replaying them against the
    /// reference must rebuild the hypothesis (and dropping insertions rebuilds the reference).
    func testOpsReplayReconstructsBothSides() {
        let reference = "she sells sea shells by the sea shore"
        let hypothesis = "he sells seashells by the shore today"
        let r = ASRMetrics.wordErrorRate(reference: reference, hypothesis: hypothesis)

        var rebuiltHyp: [String] = []
        var rebuiltRef: [String] = []
        for op in r.ops {
            switch op {
            case .match(let ref, let hyp):
                rebuiltRef.append(ref)
                rebuiltHyp.append(hyp)
            case .substitution(let ref, let hyp):
                rebuiltRef.append(ref)
                rebuiltHyp.append(hyp)
            case .deletion(let ref):
                rebuiltRef.append(ref)
            case .insertion(let hyp):
                rebuiltHyp.append(hyp)
            }
        }
        XCTAssertEqual(rebuiltRef, ASRMetrics.normalize(reference))
        XCTAssertEqual(rebuiltHyp, ASRMetrics.normalize(hypothesis))
        XCTAssertEqual(r.substitutions + r.deletions + r.insertions, r.errors)
    }

    func testEditDistanceIsMinimal() {
        // "a b c" → "x b c" costs exactly one substitution, not a delete+insert pair.
        let r = ASRMetrics.wordErrorRate(reference: "a b c", hypothesis: "x b c")
        XCTAssertEqual(r.errors, 1)
        XCTAssertEqual(r.substitutions, 1)
    }
}
