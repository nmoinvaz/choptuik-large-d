// float64 in a Metal kernel, which bench.swift expects the compiler to reject
#include <metal_stdlib>
using namespace metal;
kernel void k(device double *x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i] = x[i] * 2.0; }
