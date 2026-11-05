"""Instruction counts of the gpu_probe kernels (notes/GPU_FEASIBILITY.md), from an aarch64 disassembly.

Usage: objdump -d --no-show-raw-insn <target>/release/libgpu_probe.dylib > all.s
       python3 count.py all.s k_dv_add k_dv_mul k_dv_axpy k_qv_add k_qv_mul k_qv_axpy

Calls are followed, so a kernel's count includes its callees. The lane code is branch-free, so the static count is
the executed one (a kernel with a backward branch is flagged "loops"). On NEON an 8-lane V8 operation is four
2-lane instructions, so the operations per value are the instruction counts divided by 4 (the "per value" line).
"""
import re, sys, collections
funcs = collections.OrderedDict(); cur = None
for line in open(sys.argv[1]):
    m = re.match(r'^([0-9a-f]+) <(.+)>:$', line.strip())
    if m:
        cur = m.group(2); funcs[cur] = []; continue
    m = re.match(r'^\s*([0-9a-f]+):\s+(\S+)\s*(.*)$', line)
    if m and cur:
        funcs[cur].append((int(m.group(1), 16), m.group(2), m.group(3)))
def cls(op):
    base = op.split('.')[0]
    if base in ('fadd','fsub'): return 'add'
    if base in ('fmul',): return 'mul'
    if base in ('fmla','fmls','fmadd','fmsub','fnmadd','fnmsub'): return 'fma'
    if base in ('fneg','fabs'): return 'negabs'
    if base.startswith('fcm') or base in ('cmeq','cmtst','fcmp','fccmp'): return 'cmp'
    if base in ('bsl','bif','bit','and','orr','eor','bic','mvn','not','fcsel'): return 'sel'
    if base.startswith('ld') or base.startswith('st'): return 'mem'
    if base == 'bl': return 'call'
    if base.startswith('b.') or base in ('cbz','cbnz','tbz','tbnz','b'): return 'br'
    return 'other'
memo = {}
def total(name, depth=0):
    if name in memo: return memo[name]
    c = collections.Counter(); loops = False
    ins = funcs.get(name, [])
    for addr, op, args in ins:
        k = cls(op); c[k] += 1
        if k == 'br':
            m = re.search(r'0x([0-9a-f]+)', args)
            if m and int(m.group(1), 16) <= addr: loops = True
        if k == 'call':
            m = re.search(r'<(.+)>', args)
            if m and depth < 20:
                sub, l2 = total(m.group(1), depth + 1); c += sub; loops |= l2
    memo[name] = (c, loops); return memo[name]
for f in sys.argv[2:]:
    c, loops = total('_' + f)
    fp = c['add'] + c['mul'] + c['fma']
    pv = {k: c[k] / 4 for k in ('add', 'mul', 'fma', 'negabs', 'cmp', 'sel')}
    print(f"{f:10s} per value: add/sub {pv['add']:g} mul {pv['mul']:g} fma {pv['fma']:g} neg/abs {pv['negabs']:g} "
          f"cmp {pv['cmp']:g} select {pv['sel']:g}")
    print(f"{f:10s} add/sub {c['add']:5d} mul {c['mul']:4d} fma {c['fma']:4d} neg/abs {c['negabs']:4d} cmp {c['cmp']:4d} sel {c['sel']:4d} | arith {fp:5d} arith+cmp+sel {fp+c['cmp']+c['sel']:5d} mem {c['mem']:4d} calls {c['call']} loops {loops}")
