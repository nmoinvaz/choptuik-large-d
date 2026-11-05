// GPU side of notes/GPU_FEASIBILITY.md, run by bench.swift: float32 FMA peak, float-float axpy, binary64 add and
// multiply emulated with 64-bit integers, and 32-bit integer multiply throughput.
#include <metal_stdlib>
using namespace metal;

// ---- f32 FMA peak: 16 independent chains per thread
kernel void fma_peak(device float *out [[buffer(0)]], constant uint &iters [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
    float x[16];
    for (int j = 0; j < 16; j++) x[j] = 1.0f + 1e-3f * float(j) + 1e-7f * float(gid & 1023);
    const float a = 0.999999f, b = 1e-6f;
    for (uint k = 0; k < iters; k++) {
        for (int j = 0; j < 16; j++) x[j] = fma(x[j], a, b);
    }
    float s = 0; for (int j = 0; j < 16; j++) s += x[j];
    out[gid] = s;
}

// ---- float-float (error-free transforms in f32), the DD algorithms of multidouble on float
struct ff { float hi, lo; };
inline ff two_sum(float a, float b) { float s = a + b; float bb = s - a; return {s, (a - (s - bb)) + (b - bb)}; }
inline ff quick_two_sum(float a, float b) { float s = a + b; return {s, b - (s - a)}; }
inline ff ff_add(ff a, ff b) {
    ff s = two_sum(a.hi, b.hi); ff t = two_sum(a.lo, b.lo);
    float e = s.lo + t.hi; ff u = quick_two_sum(s.hi, e); e = u.lo + t.lo; return quick_two_sum(u.hi, e);
}
inline ff ff_mul(ff a, ff b) {
    float p = a.hi * b.hi; float e = fma(a.hi, b.hi, -p);
    e = fma(a.hi, b.lo, e); e = fma(a.lo, b.hi, e); return quick_two_sum(p, e);
}
// y = y*t + l, 8 independent chains per thread (27 f32 ops per axpy, as DD)
kernel void ff_axpy(device float2 *out [[buffer(0)]], constant uint &iters [[buffer(1)]],
                    uint gid [[thread_position_in_grid]]) {
    ff y[8];
    for (int j = 0; j < 8; j++) y[j] = {1.0f + 1e-3f * float(j), 1e-9f * float(gid & 255)};
    const ff t = {0.5f, 1.4901161e-08f}, l = {0.3333333f, 9.934108e-09f};
    for (uint k = 0; k < iters; k++) {
        for (int j = 0; j < 8; j++) y[j] = ff_add(ff_mul(y[j], t), l);
    }
    ff s = y[0]; for (int j = 1; j < 8; j++) s = ff_add(s, y[j]);
    out[gid] = float2(s.hi, s.lo);
}
// one ff op per element, for the bitwise comparison with the CPU
kernel void ff_check(device const float4 *in [[buffer(0)]], device float4 *out [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
    float4 v = in[gid]; ff a = {v.x, v.y}, b = {v.z, v.w};
    ff s = ff_add(a, b), p = ff_mul(a, b);
    out[gid] = float4(s.hi, s.lo, p.hi, p.lo);
}

// ---- soft binary64 from 64-bit integers (round to nearest even; normal inputs and results only)
inline ulong sf_mul(ulong a, ulong b) {
    ulong s = (a ^ b) & 0x8000000000000000ul;
    int ea = int((a >> 52) & 0x7ff), eb = int((b >> 52) & 0x7ff);
    ulong ma = ((a & 0xfffffffffffffull) | 0x10000000000000ul) << 11;
    ulong mb = ((b & 0xfffffffffffffull) | 0x10000000000000ul) << 11;
    ulong hi = mulhi(ma, mb), lo = ma * mb;
    int e = ea + eb - 1023;
    if (hi >> 63) { e += 1; } else { hi = (hi << 1) | (lo >> 63); lo <<= 1; }
    ulong r = hi & 0x7ff; ulong m = hi >> 11; bool sticky = lo != 0;
    if (r > 0x400 || (r == 0x400 && (sticky || (m & 1)))) m += 1;
    if (m >> 53) { m >>= 1; e += 1; }
    return s | (ulong(e) << 52) | (m & 0xfffffffffffffull);
}
inline ulong sf_add(ulong a, ulong b) {
    if ((a & 0x7fffffffffffffffull) < (b & 0x7fffffffffffffffull)) { ulong t = a; a = b; b = t; }
    ulong s = a & 0x8000000000000000ul; bool sub = ((a ^ b) >> 63) != 0;
    int ea = int((a >> 52) & 0x7ff), eb = int((b >> 52) & 0x7ff);
    ulong ma = ((a & 0xfffffffffffffull) | 0x10000000000000ul) << 9;  // leading bit at 61
    ulong mb = ((b & 0xfffffffffffffull) | 0x10000000000000ul) << 9;
    int d = ea - eb;
    if (d > 0) {
        if (d < 63) { ulong st = (mb << (64 - d)) != 0; mb = (mb >> d) | st; } else { mb = (mb != 0); }
    }
    ulong m; int e = ea;
    if (!sub) { m = ma + mb; if (m >> 62) { m = (m >> 1) | (m & 1); e += 1; } }
    else {
        m = ma - mb; if (m == 0) return 0;
        int z = clz(m) - 2; m <<= z; e -= z;  // leading bit back at 61
    }
    ulong r = m & 0x1ff; ulong q = m >> 9;
    if (r > 0x100 || (r == 0x100 && (q & 1))) q += 1;
    if (q >> 53) { q >>= 1; e += 1; }
    return s | (ulong(e) << 52) | (q & 0xfffffffffffffull);
}
// throughput: 8 independent chains, x = x*c (mul) or x = x + c (add)
kernel void sf_mul_tp(device ulong *out [[buffer(0)]], constant uint &iters [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
    ulong x[8];
    for (int j = 0; j < 8; j++) x[j] = 0x3ff0000000000000ul + ulong(gid) * 977ul + ulong(j) * 131ul;
    const ulong c = 0x3ff0000000001234ul;  // 1 + tiny
    for (uint k = 0; k < iters; k++) for (int j = 0; j < 8; j++) x[j] = sf_mul(x[j], c);
    ulong s = 0; for (int j = 0; j < 8; j++) s ^= x[j];
    out[gid] = s;
}
kernel void sf_add_tp(device ulong *out [[buffer(0)]], constant uint &iters [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
    ulong x[8];
    for (int j = 0; j < 8; j++) x[j] = 0x3ff0000000000000ul + ulong(gid) * 977ul + ulong(j) * 131ul;
    const ulong c = 0x3e70000000001234ul;  // about 6e-8
    for (uint k = 0; k < iters; k++) for (int j = 0; j < 8; j++) x[j] = sf_add(x[j], (k & 1) ? c : (c | 0x8000000000000000ul) + 0x10ul);
    ulong s = 0; for (int j = 0; j < 8; j++) s ^= x[j];
    out[gid] = s;
}
kernel void sf_check(device const ulong2 *in [[buffer(0)]], device ulong2 *out [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
    ulong2 v = in[gid];
    out[gid] = ulong2(sf_add(v.x, v.y), sf_mul(v.x, v.y));
}

// ---- 32-bit integer multiply throughput (the limb product of an integer-mantissa format): per
// iteration and chain one mul (low 32 bits), one mulhi (high 32 bits) and one add
kernel void imul_tp(device uint *out [[buffer(0)]], constant uint &iters [[buffer(1)]],
                    uint gid [[thread_position_in_grid]]) {
    uint x[16];
    for (int j = 0; j < 16; j++) x[j] = gid * 2654435761u + uint(j) * 40503u + 1u;
    const uint a = 1664525u, c = 2246822519u;
    for (uint k = 0; k < iters; k++) {
        for (int j = 0; j < 16; j++) x[j] = x[j] * a + mulhi(x[j], c);
    }
    uint s = 0; for (int j = 0; j < 16; j++) s ^= x[j];
    out[gid] = s;
}
