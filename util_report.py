#!/usr/bin/env python3
"""Machine utilization for the runners' posts, and a memory guard for a run that is allowed to die.

    python3 util_report.py                       one report: the time, cpu busy % (a 1 s sample), memory used, free and
                                                 total in GB (swap on macOS), then every python process (pid, working
                                                 set GB, cpu seconds, the command's head) and their total
    python3 util_report.py --guard MIN_FREE_GB --log FILE [--interval S] PID...
                                                 every S seconds (default 300) append the report's first line to FILE;
                                                 when free memory is below MIN_FREE_GB, kill the process trees of the
                                                 PIDs (and only those), append a "stopped" line and exit 3; exit 0 when
                                                 every PID has ended on its own
    python3 util_report.py --selftest            the guard fires on a sleeping child when the threshold is above the
                                                 machine's RAM (exit 3, the child is dead) and does not fire at 0 (the
                                                 child is alive); exits 1 if either is otherwise

Windows: GlobalMemoryStatusEx and GetSystemTimes through ctypes, Win32_Process for the per-process lines (PowerShell).
macOS: top -l 2 and vm.swapusage. Linux: /proc/meminfo and /proc/stat. Standard library only, so it runs in every
runner's venv. The per-process figure is the resident set, which undercounts a process that is partly swapped out (the
Mac's T94 workers: 1 GB resident, 4-5 GB footprint); the free-memory figure is the one the guard reads. Every comms
post from a runner carries the report (the coordinator's entry of 2026-11-01 to win, mac and win2), so that idle
capacity is seen and filled.
"""
import os
import platform
import re
import signal
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


def alive(pid):
    if WIN:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def kill_tree(pid):
    """the process and its descendants (spawn workers outlive a killed main otherwise)"""
    if WIN:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True)
        return
    out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout
    for child in out.split():
        kill_tree(int(child))
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def guard(min_free_gb, log, interval, pids, max_iter=None):
    """0 every PID ended, 3 fired (the PIDs' trees killed)"""
    n = 0
    while max_iter is None or n < max_iter:
        n += 1
        total, used, free, extra = memory()
        line = report_line()
        with open(log, "a") as f:
            f.write(line + "\n")
        if free < min_free_gb:
            for pid in pids:
                kill_tree(pid)
            with open(log, "a") as f:
                f.write(f"stopped {' '.join(map(str, pids))}: free {free:.1f} GB < {min_free_gb} GB\n")
            return 3
        if not any(alive(p) for p in pids):
            with open(log, "a") as f:
                f.write("guarded processes ended\n")
            return 0
        if max_iter is None or n < max_iter:
            time.sleep(interval)
    return 0


def selftest():
    import tempfile
    total = memory()[0]
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "guard.log")
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        time.sleep(0.5)
        fired = guard(total + 1.0, log, 0, [child.pid], max_iter=1)
        for _ in range(50):
            if child.poll() is not None:
                break
            time.sleep(0.1)
        dead = child.poll() is not None
        child2 = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        time.sleep(0.5)
        quiet = guard(0.0, log, 0, [child2.pid], max_iter=1)
        alive2 = child2.poll() is None
        kill_tree(child2.pid)
        child2.wait()
        with open(log) as f:
            lines = f.read().count("\n")
    print(f"selftest: threshold above RAM -> exit {fired} (want 3), child dead {dead} (want True); "
          f"threshold 0 -> exit {quiet} (want 0), child alive {alive2} (want True); {lines} log lines")
    return 0 if (fired, dead, quiet, alive2) == (3, True, 0, True) else 1


def main(argv):
    if "--selftest" in argv:
        return selftest()
    if "--guard" in argv:
        args = argv[1:]
        min_free = float(args[args.index("--guard") + 1])
        log = args[args.index("--log") + 1]
        interval = float(args[args.index("--interval") + 1]) if "--interval" in args else 300.0
        skip = set()
        for flag in ("--guard", "--log", "--interval"):
            if flag in args:
                i = args.index(flag)
                skip.update((i, i + 1))
        pids = [int(a) for i, a in enumerate(args) if i not in skip]
        if not pids:
            print(__doc__)
            return 2
        print(f"guard: stop {pids} when free memory < {min_free} GB, every {interval:.0f} s, log {log}", flush=True)
        return guard(min_free, log, interval, pids)
    if len(argv) > 1:
        print(__doc__)
        return 2
    print(report())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
