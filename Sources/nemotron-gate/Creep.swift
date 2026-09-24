import ArgumentParser
import Darwin
import Foundation
import MLX
import NemotronASR

/// Diagnostic, not a gate: does a long live session's memory grow with its length, and if so,
/// whose is it? Streams a wav through `NemotronStreamingSession` in 100 ms pushes on the GPU,
/// with MLX's pool capped the way MLXEngine's automatic policy caps it. Every `--every` seconds
/// of audio, BETWEEN pushes (so no chunk's temporaries are alive), it prints phys_footprint beside:
///   active     bytes referenced by live MLX arrays (weights, encoder caches, the mel tail)
///   pool       MLX's recycling cache, bounded by the cap
///   heap       the malloc zones in use (Swift arrays and strings: the hypothesis, its text)
///   other      phys − active − pool − heap: memory no counter owns (Metal heaps, driver state)
/// plus MLX's high-water mark since the previous line. Growth in `active` is a reference held
/// too long, in `heap` a Swift collection that only appends or autoreleased objects no pool
/// drains, in `other` retention below MLX. The same instrument as `qwen3asr-gates creep`.
struct Creep: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Memory over session length: phys vs MLX active / pool vs the malloc heap, sampled between pushes.")
    @OptionGroup var options: GateOptions
    @Option var wav: String
    @Option(help: "Prompt key (e.g. en-US); omitted rides the model default.") var language: String?
    @Option(help: "Right context (left is 56): 3 is Dictate's [56, 3], 13 the model default.")
    var right: Int = 3
    @Option(help: "Seconds of audio between samples.") var every: Double = 60
    @Option(help: "MLX pool cap in bytes; MLXEngine's automatic policy is min(2 GiB, 5 % of the budget).")
    var pool: Int = 2 << 30

    func run() throws {
        let ctx = try options.context()
        let model = try NemotronLoad.load(directory: ctx.modelDir)
        Memory.cacheLimit = pool
        let samples = try loadWav16kMonoSamples(URL(fileURLWithPath: wav))
        let session = NemotronStreamingSession(model: model, language: language, attContextSize: [56, right])
        var textBytes = 0
        let t0 = Date()
        func gb(_ b: Int) -> String { String(format: "%6.3f", Double(b) / 1e9) }
        func line(_ label: String) {
            let phys = physFootprint(), active = Memory.activeMemory, cache = Memory.cacheMemory, heap = mallocInUse()
            print("  \(label.padding(toLength: 16, withPad: " ", startingAt: 0)) phys \(gb(phys))  active \(gb(active))  "
                  + "pool \(gb(cache))  heap \(gb(heap))  other \(gb(phys - active - cache - heap))  "
                  + "mlx-peak \(gb(Memory.peakMemory))  text \(textBytes) B  "
                  + "wall \(String(format: "%.0f", Date().timeIntervalSince(t0))) s")
            fflush(stdout)
            Memory.peakMemory = 0   // the setter resets the high-water mark
        }
        print("# CREEP — \(ctx.modelDir.lastPathComponent) · \(URL(fileURLWithPath: wav).lastPathComponent)"
              + " · \(String(format: "%.1f", Double(samples.count) / 16000 / 60)) min · acs [56, \(right)]"
              + " · pool cap \(pool) B · pid \(getpid())\n")
        line("loaded")
        var pos = 0, next = every
        while pos < samples.count {
            let end = min(pos + 1600, samples.count)
            if let update = session.push(Array(samples[pos ..< end])) { textBytes = update.result.text.utf8.count }
            pos = end
            if Double(pos) / 16000 >= next {
                line(String(format: "%.1f min", Double(pos) / 16000 / 60))
                next += every
            }
        }
        textBytes = session.finish().result.text.utf8.count
        line("finished")
        Memory.clearCache()
        line("+ clearCache")
    }
}

/// The number Activity Monitor and the jetsam limit use — resident, compressed and GPU memory
/// owned by this process.
func physFootprint() -> Int {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? Int(info.phys_footprint) : -1
}

/// Bytes allocated from every malloc zone and not yet freed.
func mallocInUse() -> Int {
    var stats = malloc_statistics_t()
    malloc_zone_statistics(nil, &stats)
    return Int(stats.size_in_use)
}
