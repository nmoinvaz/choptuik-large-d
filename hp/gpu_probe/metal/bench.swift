import Metal
import Foundation

// Usage: bench <directory of kernels.metal and bad.metal> [mutate]
// "mutate" breaks the CPU references (float-float product without FMA, binary64 product one ulp up), so the two
// bitwise checks must report differences and the program must exit 1.
let dir = CommandLine.arguments[1]
let mutate = CommandLine.arguments.count > 2 && CommandLine.arguments[2] == "mutate"
var failed = false
guard let dev = MTLCreateSystemDefaultDevice() else { print("no Metal device"); exit(1) }
print("device: \(dev.name), unified memory \(dev.hasUnifiedMemory), recommended working set \(dev.recommendedMaxWorkingSetSize >> 20) MB")
print("max threads per threadgroup \(dev.maxThreadsPerThreadgroup.width), apple9 \(dev.supportsFamily(.apple9)), apple10 \(dev.supportsFamily(MTLGPUFamily(rawValue: 1010) ?? .apple9)), metal4 \(dev.supportsFamily(MTLGPUFamily(rawValue: 5002) ?? .apple9))")

// 1. float64 in the shading language
let bad = try! String(contentsOfFile: dir + "/bad.metal", encoding: .utf8)
do { _ = try dev.makeLibrary(source: bad, options: nil); print("double: COMPILED (unexpected)") }
catch { print("double: compile error:\n  " + String(describing: error).replacingOccurrences(of: "\n", with: "\n  ").prefix(600)) }

// 2. the kernels, IEEE-safe math
let src = try! String(contentsOfFile: dir + "/kernels.metal", encoding: .utf8)
let opts = MTLCompileOptions()
opts.mathMode = .safe
opts.mathFloatingPointFunctions = .precise
let lib = try! dev.makeLibrary(source: src, options: opts)
let q = dev.makeCommandQueue()!

func run(_ name: String, threads: Int, iters: UInt32, outBytes: Int, reps: Int = 5) -> Double {
    let pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: name)!)
    let out = dev.makeBuffer(length: threads * outBytes, options: .storageModeShared)!
    var it = iters
    var best = 1e9
    for _ in 0..<reps {
        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(out, offset: 0, index: 0)
        enc.setBytes(&it, length: 4, index: 1)
        enc.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        best = min(best, cb.gpuEndTime - cb.gpuStartTime)
    }
    return best
}
let T = 1 << 18
var t = run("fma_peak", threads: T, iters: 4096, outBytes: 4)
let fmaRate = Double(T) * 4096 * 16 / t
print(String(format: "f32 FMA: %.3e FMA/s = %.2f TFLOPS (FMA = 2 flops), %.2f ms", fmaRate, 2 * fmaRate / 1e12, t * 1e3))
t = run("ff_axpy", threads: T, iters: 1024, outBytes: 8)
let ffRate = Double(T) * 1024 * 8 / t
print(String(format: "float-float axpy (27 f32 ops): %.3e axpy/s, %.2e f32 op/s, %.2f ms", ffRate, ffRate * 27, t * 1e3))
t = run("sf_mul_tp", threads: T, iters: 256, outBytes: 8)
let mulRate = Double(T) * 256 * 8 / t
print(String(format: "soft binary64 mul: %.3e /s, %.2f ms", mulRate, t * 1e3))
t = run("sf_add_tp", threads: T, iters: 256, outBytes: 8)
let addRate = Double(T) * 256 * 8 / t
print(String(format: "soft binary64 add: %.3e /s, %.2f ms", addRate, t * 1e3))
t = run("imul_tp", threads: T, iters: 1024, outBytes: 4)
let imRate = Double(T) * 1024 * 16 / t
print(String(format: "uint32 mul + mulhi + add: %.3e /s (each of the three), %.2f ms", imRate, t * 1e3))

// 3. bitwise checks against the CPU
var rng = SystemRandomNumberGenerator()
let N = 1 << 20
func check(_ name: String, inBytes: Int, fill: (UnsafeMutableRawPointer) -> Void, verify: (UnsafeRawPointer, UnsafeRawPointer) -> (Int, Int)) {
    let pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: name)!)
    let inb = dev.makeBuffer(length: N * inBytes, options: .storageModeShared)!
    let outb = dev.makeBuffer(length: N * inBytes, options: .storageModeShared)!
    fill(inb.contents())
    let cb = q.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pso); enc.setBuffer(inb, offset: 0, index: 0); enc.setBuffer(outb, offset: 0, index: 1)
    enc.dispatchThreads(MTLSize(width: N, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    let (bad, total) = verify(inb.contents(), outb.contents())
    print("\(name): \(bad) of \(total) results differ from the CPU bitwise")
    if bad != 0 { failed = true }
}
// float-float on the CPU, the same operation order
func cpuTwoSum(_ a: Float, _ b: Float) -> (Float, Float) { let s = a + b; let bb = s - a; return (s, (a - (s - bb)) + (b - bb)) }
func cpuQTS(_ a: Float, _ b: Float) -> (Float, Float) { let s = a + b; return (s, b - (s - a)) }
func cpuFFAdd(_ a: (Float, Float), _ b: (Float, Float)) -> (Float, Float) {
    let s = cpuTwoSum(a.0, b.0); let t = cpuTwoSum(a.1, b.1)
    var e = s.1 + t.0; let u = cpuQTS(s.0, e); e = u.1 + t.1; return cpuQTS(u.0, e)
}
func cpuFFMul(_ a: (Float, Float), _ b: (Float, Float)) -> (Float, Float) {
    let p = a.0 * b.0; var e = mutate ? a.0 * b.0 - p : (-p).addingProduct(a.0, b.0)
    e = e.addingProduct(a.0, b.1); e = e.addingProduct(a.1, b.0); return cpuQTS(p, e)
}
func randFF() -> (Float, Float) {
    let h = Float(Double.random(in: -1...1, using: &rng)) * Float(pow(2.0, Double(Int.random(in: -20...20, using: &rng))))
    let l = h * Float.ulpOfOne * Float(Double.random(in: -0.5...0.5, using: &rng))
    let (s, e) = cpuQTS(h, l); return (s, e)
}
check("ff_check", inBytes: 16, fill: { p in
    let f = p.bindMemory(to: Float.self, capacity: 4 * N)
    for i in 0..<N { let a = randFF(); var b = randFF(); if i % 4 == 0 { b = (-a.0 * (1 + Float.ulpOfOne * 4), b.1) }
        f[4*i] = a.0; f[4*i+1] = a.1; f[4*i+2] = b.0; f[4*i+3] = b.1 }
}, verify: { ip, op in
    let f = ip.bindMemory(to: Float.self, capacity: 4 * N); let o = op.bindMemory(to: Float.self, capacity: 4 * N)
    var bad = 0
    for i in 0..<N { let a = (f[4*i], f[4*i+1]), b = (f[4*i+2], f[4*i+3])
        let s = cpuFFAdd(a, b), m = cpuFFMul(a, b)
        let g = [o[4*i], o[4*i+1], o[4*i+2], o[4*i+3]], c = [s.0, s.1, m.0, m.1]
        for k in 0..<4 where g[k].bitPattern != c[k].bitPattern { bad += 1 } }
    return (bad, 4 * N)
})
func randD() -> Double { Double.random(in: 1...2, using: &rng) * pow(2.0, Double(Int.random(in: -60...60, using: &rng))) * (Bool.random(using: &rng) ? 1 : -1) }
check("sf_check", inBytes: 16, fill: { p in
    let u = p.bindMemory(to: UInt64.self, capacity: 2 * N)
    for i in 0..<N { let a = randD(); var b = randD()
        if i % 3 == 0 { b = -a * (1 + Double(Int.random(in: 1...1000, using: &rng)) * Double.ulpOfOne) }  // cancellation
        if i % 3 == 1 { b = a * pow(2.0, Double(Int.random(in: -70...0, using: &rng))) }                      // alignment, sticky
        u[2*i] = a.bitPattern; u[2*i+1] = b.bitPattern }
}, verify: { ip, op in
    let u = ip.bindMemory(to: UInt64.self, capacity: 2 * N); let o = op.bindMemory(to: UInt64.self, capacity: 2 * N)
    var bad = 0
    for i in 0..<N { let a = Double(bitPattern: u[2*i]), b = Double(bitPattern: u[2*i+1])
        if (a + b).bitPattern != o[2*i] { bad += 1 }
        if (mutate ? (a * b).nextUp : a * b).bitPattern != o[2*i+1] { bad += 1 } }
    return (bad, 2 * N)
})
exit(failed ? 1 : 0)
