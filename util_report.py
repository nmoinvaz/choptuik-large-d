#!/usr/bin/env python3
"""Machine utilization for the runners' posts.

    python3 util_report.py                       one report: the time, cpu busy % (a 1 s sample), memory used, free and
                                                 total in GB (swap on macOS), then every python process (pid, working
                                                 set GB, cpu seconds, the command's head) and their total

Windows: GlobalMemoryStatusEx and GetSystemTimes through ctypes, Win32_Process for the per-process lines (PowerShell).
macOS: top -l 2 and vm.swapusage. Linux: /proc/meminfo and /proc/stat. Standard library only, so it runs in every
runner's venv. The per-process figure is the resident set, which undercounts a process that is partly swapped out (the
Mac's T94 workers: 1 GB resident, 4-5 GB footprint). Every comms post from a runner carries the report (the
coordinator's entry of 2026-11-01 to win, mac and win2), so that idle capacity is seen and filled.
"""
import os
import platform
import re
import subprocess
import sys
import time

GB = 1024.0 ** 3
WIN = platform.system() == "Windows"
MAC = platform.system() == "Darwin"


def _size_gb(s):
    """'31G', '3388M', '299M', '17573.00M' -> GB"""
    m = re.match(r"([\d.]+)([KMGT])", s)
    v, u = float(m.group(1)), m.group(2)
    return v * {"K": 1 / 1024 ** 2, "M": 1 / 1024, "G": 1, "T": 1024}[u]


def memory():
    """(total_gb, used_gb, free_gb, swap_text); free is what a new process can take without paging"""
    if WIN:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(st)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        total, free = st.ullTotalPhys / GB, st.ullAvailPhys / GB
        page = (st.ullTotalPageFile - st.ullAvailPageFile) / GB
        return total, total - free, free, f"commit {page:.1f} of {st.ullTotalPageFile / GB:.1f} GB"
    if MAC:
        total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / GB
        top = subprocess.run(["top", "-l", "1", "-n", "0"], capture_output=True, text=True).stdout
        m = re.search(r"PhysMem: (\S+) used .*?(\S+) unused", top)
        used, free = _size_gb(m.group(1)), _size_gb(m.group(2))
        sw = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout
        m = re.search(r"total = (\S+)\s+used = (\S+)", sw)
        swap = f"swap used {_size_gb(m.group(2)):.1f} of {_size_gb(m.group(1)):.1f} GB"
        return total, used, free, swap
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":")
            info[k] = int(v.split()[0]) / 1024 ** 2
    total, free = info["MemTotal"], info["MemAvailable"]
    swap = f"swap used {info['SwapTotal'] - info['SwapFree']:.1f} of {info['SwapTotal']:.1f} GB"
    return total, total - free, free, swap


def cpu_percent(sample=1.0):
    """busy % of all logical cores over `sample` seconds"""
    if WIN:
        import ctypes

        class FILETIME(ctypes.Structure):
            _fields_ = [("lo", ctypes.c_ulong), ("hi", ctypes.c_ulong)]

        def times():
            idle, kern, user = FILETIME(), FILETIME(), FILETIME()
            ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user))
            f = lambda t: (t.hi << 32) | t.lo
            return f(idle), f(kern) + f(user)        # kernel time includes idle time
        i0, t0 = times()
        time.sleep(sample)
        i1, t1 = times()
        return 100.0 * (1.0 - (i1 - i0) / max(t1 - t0, 1))
    if MAC:
        top = subprocess.run(["top", "-l", "2", "-n", "0", "-s", str(max(1, int(sample)))],
                             capture_output=True, text=True).stdout
        idle = re.findall(r"CPU usage:.*?([\d.]+)% idle", top)[-1]
        return 100.0 - float(idle)

    def stat():
        with open("/proc/stat") as f:
            v = [int(x) for x in f.readline().split()[1:]]
        return v[3] + v[4], sum(v)                   # idle + iowait, total
    i0, t0 = stat()
    time.sleep(sample)
    i1, t1 = stat()
    return 100.0 * (1.0 - (i1 - i0) / max(t1 - t0, 1))


def _cputime_s(s):
    """ps cputime 'DD-HH:MM:SS', 'HH:MM:SS' or 'MM:SS' -> seconds"""
    days = 0
    if "-" in s:
        d, s = s.split("-")
        days = int(d)
    parts = [float(x) for x in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def _head(cmd):
    """the informative part of a python command line"""
    if "multiprocessing" in cmd:
        return "spawn worker"
    toks = cmd.split()
    for i, t in enumerate(toks):
        if t.endswith(".py"):
            return " ".join([os.path.basename(t)] + toks[i + 1:i + 7])
    return " ".join(toks[:6])[:70]


def python_procs():
    """[(pid, gb, cpu_s, head)] for every python process but this one"""
    rows = []
    me = os.getpid()
    if WIN:
        ps = ("Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | ForEach-Object { "
              "'{0}|{1}|{2}|{3}' -f $_.ProcessId, $_.WorkingSetSize, "
              "[int](($_.KernelModeTime + $_.UserModeTime) / 1e7), $_.CommandLine }")
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if line.count("|") >= 3:
                pid, ws, cpu, cmd = line.split("|", 3)
                if int(pid) != me:
                    rows.append((int(pid), int(ws) / GB, float(cpu), _head(cmd)))
    else:
        out = subprocess.run(["ps", "-axo", "pid=,rss=,cputime=,args="], capture_output=True, text=True).stdout
        for line in out.splitlines():
            parts = line.split(None, 3)
            if len(parts) != 4 or int(parts[0]) == me:
                continue
            if os.path.basename(parts[3].split()[0]).startswith("python"):      # the interpreter, not a path
                rows.append((int(parts[0]), int(parts[1]) * 1024 / GB, _cputime_s(parts[2]), _head(parts[3])))
    return sorted(rows, key=lambda r: -r[1])


def report_line():
    total, used, free, extra = memory()
    cpu = cpu_percent()
    stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    return f"{stamp}  cpu {cpu:.0f}%  mem used {used:.1f} of {total:.1f} GB, free {free:.1f} GB, {extra}"


def report():
    lines = [report_line()]
    rows = python_procs()
    for pid, gb, cpu, head in rows:
        lines.append(f"  pid {pid:<7d} {gb:5.2f} GB  cpu {cpu:8.0f} s  {head}")
    lines.append(f"  python total {sum(r[1] for r in rows):.1f} GB in {len(rows)} processes")
    return "\n".join(lines)


def main(argv):
    if len(argv) > 1:
        print(__doc__)
        return 2
    print(report())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
