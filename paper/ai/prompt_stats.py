"""Counts of the prompts typed into this project's Claude Code sessions, by form, length and time.

    python3 paper/ai/prompt_stats.py TRANSCRIPT_DIR            # writes paper/ai/prompt_stats.csv, prints the tables
    python3 paper/ai/prompt_stats.py TRANSCRIPT_DIR --sample 100 --seed 7   # a random sample with labels, to check by hand
    python3 paper/ai/prompt_stats.py --selftest [--mutate=form|drop|dup]
    uv run --with pdfplumber python3 paper/ai/prompt_stats.py TRANSCRIPT_DIR --chat ~/Source/choptuik-og-history.pdf
                        # adds the chat where the project began, read from its printed copy (the user's prompts are
                        # set in their own typeface); writes prompt_times.csv for the figure of all sessions

TRANSCRIPT_DIR is Claude Code's folder of session transcripts (.jsonl) for this project on the coordinator machine.
The transcripts are private, so only the counts are stored. A prompt is a `type == "user"` record that is not meta,
not a side chain and not a tool result, or a queued command. Text that a session, the harness or a watcher wrote is
dropped (DROP), as is a prompt that is only an image, a path or a link (POINTER), sessions about other things are left out (OTHER), and a record copied into a resumed transcript is
counted once.

The form of a prompt is read from its first words by the rules of form(). The same rules are applied to the 145
prompts that changed the course of the work, listed in SOURCES.md, so the two groups can be compared.
"""
import argparse, csv, json, random, re, statistics, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
# sessions in the folder that are not about this project (a job search, macOS settings, a circuit board, artwork)
OTHER = {"1b2113d1", "a4352ed2", "f1e09559", "6908571c", "5e36a0d5", "2fb521f0"}
DROP = [re.compile(p) for p in (
    r"^This session is being continued from a previous conversation",            # context summary
    r"^\s*<(hc[ >]|agent-message|cross-session-message|teammate-message|artifact-content)",   # another session
    r"^\s*<(task-notification|system-reminder|local-command|command-name|command-message|bash-input|bash-stdout)",
    r"^\[Request interrupted", r"^Caveat: The messages below",
    r"^(Curate the notes|Reply with the single word OK|Coordinator: the notes rules changed)",   # automation
    r"^You are \w+, an? (agent|runner)", r"^#\d+: ",                             # a brief or a task written by a session
    r"^\w+ #\d+ .{0,40}alert:", r"^\w+ \(\d\d:\d\d UTC\)",                       # watcher alerts
)]
TAGS = re.compile(r"<(system-reminder|local-command-stdout|command-args|command-name|command-message|pasted_content[^>]*)>.*?</\1>|<hc from=.*", re.S)

LEAD = r"^(?:(?:ok|okay|so|and|also|but|then|hmm|well|yes|no|actually)[,.]? )?"
CHECK = re.compile(LEAD + r"(how'?s? (is |are )?.{0,30}\b(going|doing|looking|coming)\b|hows? (it|things)\b|"
                   r"(any|what'?s the|whats the|what is the) (progress|updates?|news|status|eta)|any new (findings|results)|status\b|"
                   r"(give me|show me) (a |an |the )?(status|update)|where are we\b|check on\b|summary$)", re.I)
GO = re.compile(r"^(yes|yep|yeah|ya|ok|okay|sure|go ahead|go for it|do it|do that|do both|do all|sounds good|"
                r"approved?|agreed|correct|right|both|all( of them)?|either|let'?s (do|go|proceed)\b|(do |option |number |#)?\d+\b)", re.I)
ON = re.compile(r"^(please )?(continue|proceed|keep going|carry on)( please| your work| subagents)?[.!]?$", re.I)
WH = re.compile(r"\b(what|why|how|when|where|which|who|whose)\b", re.I)
WH0 = re.compile(LEAD + r"(what|why|how|where|which|who|whose)\b", re.I)
AUX = re.compile(LEAD + r"(is|are|was|were|does|did|can|could|should|would|will|shall|isn'?t|aren'?t|doesn'?t|wouldn'?t|shouldn'?t|couldn'?t|any|"
                 r"do (we|you|i|they|it|any)|don'?t (we|you|they))\b", re.I)
POINTER = re.compile(r"^(\[Image #\d+\]\s*)+$|^[~/][\w./@+\-]+$|^https?://\S+$")
# each Claude Code session of the project by the part of the work it served; any other is a short session
COMPONENT = {**dict.fromkeys(["43f19197", "11bcbd25", "afb99455", "0726f216"], "coordinator sessions"),
             **dict.fromkeys(["6d6cbe24", "ebec4b64"], "the libraries"),
             **dict.fromkeys(["0c8fc979", "22958cb5", "c88d10f3", "9fac6212", "12451890"], "channel, tracker and housekeeping"),
             **dict.fromkeys(["cc3aa2b5", "3bc27918"], "the Mac runner"),
             "cb202ff5": "this paper, the film and the one-prompt trial"}
ROWS = ["the chat", "coordinator sessions", "the libraries", "channel, tracker and housekeeping", "the Mac runner",
        "this paper, the film and the one-prompt trial", "short sessions"]
UNTIL = "2026-11-26T07:20:00Z"        # the counts stop here, so a rerun gives the same numbers as the paper
CHAT_END = "2026-10-26T15:53:00Z"      # the handoff to the first Claude Code coordinator (tx:43f19197's first record)
DAY0 = "2026-10-18"


def component(sid):
    return COMPONENT.get(sid[:8], "short sessions")


def chat_prompts(pdf):
    """The user's prompts in the printed chat: lines set in the prompt typeface, joined while they run on."""
    import pdfplumber
    out, cur = [], None
    for p in pdfplumber.open(pdf).pages:
        for line in p.extract_text_lines():
            c = line["chars"][0]
            if c["fontname"].endswith("Roman-Regular") and c["size"] > 11 and line["x0"] > 100:
                if cur is not None:
                    cur.append(line["text"])
                else:
                    cur = [line["text"]]
                    out.append(cur)
            elif line["top"] > 25:                 # anything but the page header ends a prompt
                cur = None
    return [" ".join(x) for x in out]


FORMS = ["instruction or statement", "question: why, what or how", "question: whether, can we, should we", "go-ahead or choice", "check-in"]


def form(text, mutate=""):
    """One of FORMS, from the surface of the prompt. A go-ahead is short: a long reply that starts with yes is read by what follows."""
    t = re.sub(r"^[\s.,:;>\-]+", "", text.strip().replace("\u2019", "'"))
    words = len(t.split())
    if CHECK.search(t) and words <= 16:
        return FORMS[4]
    if (ON.search(t) or (GO.search(t) and words <= 8)) and mutate != "form":
        return FORMS[3]
    first = re.split(r"(?<=[.!?])\s", t, maxsplit=1)[0]      # a prompt is read by its first sentence
    if AUX.search(first):
        return FORMS[2]
    if WH0.search(first):
        return FORMS[1]
    if first.endswith("?") or (first == t and "?" in t):
        return FORMS[1] if WH.search(first) else FORMS[2]
    return FORMS[0]


def text_of(msg):
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c):
            return None
        return "\n".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
    return None


def prompts(lines_by_session, mutate=""):
    """[{session, t, text}] in time order, from {session id: iterable of JSON lines}."""
    kept, seen = [], set()
    for sid in sorted(lines_by_session):
        if sid[:8] in OTHER:
            continue
        for line in lines_by_session[sid]:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            t = None
            if e.get("type") == "user" and not e.get("isMeta") and not e.get("isSidechain"):
                t = text_of(e.get("message", {}))
            elif e.get("type") == "attachment" and (e.get("attachment") or {}).get("type") == "queued_command":
                p = e["attachment"].get("prompt")
                t = p if isinstance(p, str) else text_of({"content": p})
            if t is None or (mutate != "drop" and any(r.search(t) for r in DROP)):
                continue
            t = TAGS.sub("", t).strip()
            key = (e.get("timestamp"), t[:200])
            if not t or POINTER.search(t) or (key in seen and mutate != "dup"):
                continue
            seen.add(key)
            if (e.get("timestamp") or "") >= UNTIL:
                continue
            kept.append({"session": sid[:8], "t": e.get("timestamp") or "", "text": t})
    return sorted(kept, key=lambda r: r["t"])


def pushes():
    """The 145 course-changing prompts of SOURCES.md, as [(kind, text)]."""
    rows, on = [], False
    for line in (HERE / "SOURCES.md").read_text().splitlines():
        if line.startswith("## The 145 prompts"):
            on = True
        elif on and line.startswith("## "):
            break
        elif on and line.startswith("| ") and not line.startswith(("| when", "|---")):
            c = [x.strip() for x in line.strip("|").split("|", 3)]
            rows.append((c[2], c[3]))
    return rows


def days(t):
    from datetime import datetime
    a = datetime.fromisoformat(t.replace("Z", "+00:00"))
    b = datetime.fromisoformat(DAY0 + "T00:00:00+00:00")
    return round((a - b).total_seconds() / 86400, 4)


def components(P, chat):
    rows = [("component", "the chat", len(chat))] if chat is not None else []
    c = Counter(component(p["session"]) for p in P)
    rows += [("component", k, c[k]) for k in ROWS[1:]]
    for k in ROWS[1:]:
        ts = [p["t"][:10] for p in P if component(p["session"]) == k]
        rows.append(("component sessions", k, len({p["session"] for p in P if component(p["session"]) == k})))
        rows.append(("component dates", k, f"{min(ts)} to {max(ts)}"))
    rows.append(("component", "all", sum(r[2] for r in rows if r[0] == "component")))
    if chat is not None:
        f = Counter(form(x) for x in chat)
        rows += [("form, the chat", k, f[k]) for k in FORMS]
        rows.append(("words, the chat", "median", statistics.median(len(x.split()) for x in chat)))
    return rows


def tables(P, push):
    words = [len(p["text"].split()) for p in P]
    days = Counter(p["t"][:10] for p in P)
    hours = Counter(p["t"][:13] for p in P)
    hours_per_day = Counter(h[:10] for h in hours)
    out = [("count", "prompts", len(P)), ("count", "sessions", len({p["session"] for p in P})),
           ("count", "first day", min(days)), ("count", "last day", max(days)), ("count", "days with a prompt", len(days)),
           ("count", "prompts per day, median", statistics.median(days.values())), ("count", "prompts per day, most", max(days.values())),
           ("count", "hours with a prompt", len(hours)), ("count", "hours with a prompt per day, median", statistics.median(hours_per_day.values())),
           ("words", "median", statistics.median(words)), ("words", "5 or fewer", sum(w <= 5 for w in words)),
           ("words", "6 to 20", sum(5 < w <= 20 for w in words)), ("words", "21 to 50", sum(20 < w <= 50 for w in words)),
           ("words", "over 50", sum(w > 50 for w in words))]
    f = Counter(form(p["text"]) for p in P)
    g = Counter(form(t) for _, t in push)
    out += [("form, all prompts", k, f[k]) for k in FORMS] + [("form, course-changing prompts", k, g[k]) for k in FORMS]
    out += [("course-changing prompts", "count", len(push)), ("course-changing prompts", "median words", statistics.median(len(t.split()) for _, t in push))]
    return out


def selftest(mutate):
    user = lambda t, text, **k: json.dumps({"type": "user", "timestamp": t, "message": {"content": text}, **k})
    planted = {"aaaaaaaa-1": [user("2026-01-01T10:00:00Z", "why does the solver stall at 78?"), user("2026-01-01T10:05:00Z", "yes do it"),
                              user("2026-01-01T11:00:00Z", "can we speed it up with SIMD?"), user("2026-01-01T11:30:00Z", "just jump to 100"),
                              user("2026-01-02T09:00:00Z", "how\u2019s it going?"), user("2026-01-02T09:10:00Z", "continue to D = 200"),
                              user("2026-01-02T09:20:00Z", "<task-notification>a run ended</task-notification>"), user("2026-01-02T09:30:00Z", "a meta record", isMeta=True),
                              json.dumps({"type": "user", "timestamp": "2026-01-02T09:40:00Z", "message": {"content": [{"type": "tool_result", "content": "x"}]}})],
               "bbbbbbbb-2": [user("2026-01-01T10:00:00Z", "why does the solver stall at 78?")],          # the copy in a resumed transcript
               "1b2113d1-3": [user("2026-01-03T10:00:00Z", "help me with my resume")]}
    P = prompts(planted, mutate)
    got = [form(p["text"], mutate) for p in P]
    want = [FORMS[1], FORMS[3], FORMS[2], FORMS[0], FORMS[4], FORMS[0]]
    ok = got == want and len(pushes()) == 145
    print(f"selftest: {len(P)} prompts kept of 11 records, forms {'as planted' if got == want else 'WRONG'}, {len(pushes())} course-changing prompts read: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    a = argparse.ArgumentParser()
    a.add_argument("dir", nargs="?"); a.add_argument("--out", default=str(HERE / "prompt_stats.csv"))
    a.add_argument("--sample", type=int, default=0); a.add_argument("--seed", type=int, default=7)
    a.add_argument("--selftest", action="store_true"); a.add_argument("--mutate", default="")
    a.add_argument("--chat", default="")
    a = a.parse_args()
    if a.selftest:
        return selftest(a.mutate)
    P = prompts({f.stem: f.open() for f in sorted(Path(a.dir).glob("*.jsonl"))})
    if a.sample:
        for p in random.Random(a.seed).sample(P, a.sample):
            print(f"{form(p['text'])[:14]:14} | {p['text'][:140]!r}")
        return 0
    chat = chat_prompts(a.chat) if a.chat else None
    rows = tables(P, pushes()) + components(P, chat)
    if chat is not None:
        with open(HERE / "prompt_times.csv", "w", newline="") as o:
            w = csv.writer(o, lineterminator="\n")
            w.writerow(["day", "row", "form"])
            for p in P:
                w.writerow([days(p["t"]), ROWS.index(component(p["session"])), FORMS.index(form(p["text"]))])
    with open(a.out, "w", newline="") as o:
        csv.writer(o, lineterminator="\n").writerows([("table", "row", "value")] + rows)
    for r in rows:
        print(f"{r[0]:32} {r[1]:42} {r[2]}")
    return 0


sys.exit(main())
