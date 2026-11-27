#!/bin/bash
# One clean-room trial of a single prompt: a fresh headless Claude Code session that sees only an empty work folder.
#   trial.sh ID PROMPT_FILE MODEL BUDGET_USD HOURS [CPUS]
#   SP_TURNS=N trial.sh ...   also stops the session after N agentic turns (claude -p --max-turns); HOURS stays as a
#                             backstop on the wall clock, so a turn limit is not eaten by a long build
#   SP_RUST=1 trial.sh ...    also mounts this machine's stable Rust toolchain read-only at /opt/rust and the crates of
#                             rust/vendor (made by vendor.sh) at /opt/vendor, with cargo set to build from them offline
#   SP_PROBE=FILE trial.sh ...  runs bash FILE in the same sandbox instead of the session (a build check)
# Start it from a login shell that exports the account's token (RUN.md). Everything the trial writes stays in trials/ID.
# The session runs inside bubblewrap: /usr and /etc read-only, its own /home/user, /work and /tmp, its own process
# list, and none of this machine's /home. So it cannot read a checkout, a memory folder or a CLAUDE.md, and it cannot
# see or signal another job. It keeps the network, which the API needs: score.py lists every command that used it.
# Its only tools are Bash and the file tools (no web search, no web fetch, no agents). The environment is emptied
# but for the token, so no other secret of the shell reaches it.
{
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ID="$1"; PROMPT="$2"; MODEL="$3"; BUDGET="$4"; HOURS="$5"; CPUS="${6:-8-11}"
: "${CLAUDE_CODE_OAUTH_TOKEN:?start from a login shell that exports the token, see RUN.md}"
T="$ROOT/trials/$ID"
[ -e "$T" ] && { echo "trial $ID exists already"; exit 2; }
[ -d "$ROOT/pylib" ] || { echo "no $ROOT/pylib"; exit 2; }
mkdir -p "$T/work" "$T/home"
cp "$PROMPT" "$T/prompt.md"
# a control for the canary: a planted instruction file, which the session must find and report
if [ -n "${SP_PLANT:-}" ]; then cp "$SP_PLANT" "$T/work/CLAUDE.md"; echo "$SP_PLANT" > "$T/planted"; fi
claude --version > "$T/claude_version"
echo "$MODEL" > "$T/model"
echo "${SP_TURNS:-none}" > "$T/max_turns"
TURNS=()
if [ -n "${SP_TURNS:-}" ]; then TURNS=(--max-turns "$SP_TURNS"); fi
RUST=(); SPATH=/usr/bin:/bin
if [ "${SP_RUST:-}" = 1 ]; then
  TC="$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu"
  [ -d "$TC/bin" ] && [ -d "$ROOT/rust/vendor" ] || { echo "no toolchain at $TC or no $ROOT/rust/vendor"; exit 2; }
  mkdir -p "$T/home/.cargo"
  printf '%s\n' '[source.crates-io]' 'replace-with = "vendored-sources"' '' '[source.vendored-sources]' \
    'directory = "/opt/vendor"' '' '[net]' 'offline = true' > "$T/home/.cargo/config.toml"
  { "$TC/bin/cargo" --version; "$TC/bin/rustc" --version; ls "$ROOT/rust/vendor"; } > "$T/rust_versions"
  RUST=(--ro-bind "$TC" /opt/rust --ro-bind "$ROOT/rust/vendor" /opt/vendor --setenv CARGO_HOME /home/user/.cargo)
  SPATH=/opt/rust/bin:/usr/bin:/bin
fi
python3 -c "import sys; sys.path.insert(0, '$ROOT/pylib'); import numpy, scipy, mpmath, sympy; print('python', sys.version.split()[0], 'numpy', numpy.__version__, 'scipy', scipy.__version__, 'mpmath', mpmath.__version__, 'sympy', sympy.__version__)" > "$T/python_versions"
TEXT="$(cat "$T/prompt.md")"
CMD=(/usr/bin/claude -p "$TEXT" --model "$MODEL" --output-format stream-json --verbose
  --tools "Bash,Read,Write,Edit,Glob,Grep" --permission-mode auto --max-budget-usd "$BUDGET" --strict-mcp-config "${TURNS[@]}")
if [ -n "${SP_PROBE:-}" ]; then cp "$SP_PROBE" "$T/work/probe.sh"; CMD=(/usr/bin/bash /work/probe.sh); fi
date -u '+%Y-%m-%d %H:%M:%S' > "$T/start_utc"
set +e
(
  for v in $(compgen -e); do
    case "$v" in CLAUDE_CODE_OAUTH_TOKEN) ;; *) unset "$v" 2>/dev/null ;; esac
  done
  exec /usr/bin/nice -n 10 /usr/bin/taskset -c "$CPUS" /usr/bin/timeout --signal=TERM --kill-after=60 "${HOURS}h" \
    /usr/bin/bwrap --ro-bind /usr /usr --symlink usr/bin /bin --symlink usr/sbin /sbin --symlink usr/lib /lib --symlink usr/lib64 /lib64 \
      --ro-bind /etc /etc --ro-bind /run/systemd/resolve /run/systemd/resolve --proc /proc --dev /dev --tmpfs /tmp \
      --tmpfs /home --bind "$T/home" /home/user --bind "$T/work" /work --ro-bind "$ROOT/pylib" /opt/pylib "${RUST[@]}" \
      --chdir /work --setenv HOME /home/user --setenv PATH "$SPATH" --setenv PYTHONPATH /opt/pylib --setenv LANG C.UTF-8 \
      --unshare-pid --unshare-ipc --unshare-uts --hostname trial --die-with-parent \
    "${CMD[@]}"
) > "$T/transcript.jsonl" 2> "$T/stderr.log"
echo $? > "$T/exit_code"
date -u '+%Y-%m-%d %H:%M:%S' > "$T/end_utc"
# the token must never stay in a file: replace it wherever the session printed it
python3 - "$T" <<'PY'
import os, pathlib, re, sys
tok, n = os.environ["CLAUDE_CODE_OAUTH_TOKEN"].encode(), 0
own = re.compile(rb"(CLAUDE_CODE_[A-Z_]*TOKEN=)[^\s\"\\]+")          # the session's own tokens, if it printed its environment
for p in pathlib.Path(sys.argv[1]).rglob("*"):
    if p.is_file() and not p.is_symlink():
        b = p.read_bytes()
        c = own.sub(rb"\1REDACTED", b.replace(tok, b"REDACTED"))
        if c != b:
            p.write_bytes(c)
            n += 1
pathlib.Path(sys.argv[1], "token_redacted_files").write_text(f"{n}\n")
PY
echo "trial $ID: exit $(cat "$T/exit_code"), $(cat "$T/start_utc") to $(cat "$T/end_utc") UTC, token redacted in $(cat "$T/token_redacted_files") files"
exit
}
