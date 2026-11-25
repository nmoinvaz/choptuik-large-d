#!/usr/bin/env python3
"""The narration of quest_large_d.py: the words the voice says for a caption, their sound, and the film's sound track.

    python voice.py lines DIR                        # every line of the scenes' records in DIR, as it is spoken
    python voice.py clips DIR                        # make the sound of every line in DIR that has none yet
    python voice.py track DIR VIDEOS OUT.wav S ...   # the sound track of the scenes S ..., joined in that order
    python voice.py selftest [--mutate=formula|overrun|pace|stale]

A scene's record is DIR/<Scene>.json, written by quest_large_d.py when the scene is rendered: the time each line starts
at, its words, and the scene's length. `track` lays each line's clip at its time on the joined film (VIDEOS holds the
scenes' .mp4 files, whose lengths give each scene's offset). It fails if a record does not match its video, if a line
would run into the next one or past the end of its scene, or if a clip's pace is far from speech.

The voice is Kokoro (hexgrad, Kokoro-82M, Apache 2.0), a text-to-speech model run on this machine through kokoro-onnx
(https://github.com/thewh1teagle/kokoro-onnx), with espeak-ng for the phonemes. build.sh fetches its two files into
media/voice/model. QUEST_VOICE names the speaker (am_michael by default; "off" leaves the film silent).

A formula in a caption is read by say_math(): a single symbol, `X = number`, a power of ten, or an entry of SAID. A
formula with no reading is an error, so a new one needs an entry. `selftest` runs the reader and the track's checks on
planted input with known answers; each --mutate breaks that input in one way and must exit 1.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import wave
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
VOICE = os.environ.get("QUEST_VOICE", "am_michael")
MODEL = HERE / "media" / "voice" / "model"
RATE = 24000                                  # Kokoro's samples a second
GAP = 0.35                                    # seconds of quiet kept after a line
PACE = (5.0, 22.0)                            # characters a second: a clip outside this is not its line, spoken
HEADROOM = 0.5                                # clips are stored at half scale: some speakers peak above 1
FIELD_A = "<a>"                               # the field a: espeak reads a lone "a" as the article, so its name is put in
SOUND = {FIELD_A: "ˈeɪ", "<record>": "ɹᵻkˈɔːɹd", "<replay>": "ɹiːplˈeɪ",        # words given their phonemes: espeak picks
         "<close>": "klˈoʊs", "<to close>": "klˈoʊz"}                           # the wrong one of two readings for them

SYMBOL = {r"\Delta": "Delta", r"\sigma": "sigma", r"\varepsilon": "epsilon", "a": FIELD_A}
SAID = {                                      # the formulas the rules of say_math() would not read well
    r"(1 - x)^{\sigma}": "one minus x, to the power sigma",
    r"4 \times 5 \times 32": "4 times 5 times 32",
    r"D_0": "D zero",
    r"F(z)": "F of z",
    r"F(z) = 0": "F of z equals zero",
    r"\Delta \approx 3.445": "Delta is about 3.445",
    r"\Delta(D)": "Delta of D",
    r"\varepsilon G^2": "epsilon G squared",
    r"dz": "d z",
    r"e^{\Delta}": "e to the power Delta",
    r"x - c = 0": "x minus c equals zero",
}
RESPELL = [("Gundlach", "Gundlock"), ("Cochelin", "Coshlann"),     # names espeak reads wrongly as written
           ("and record its", "and <record> its"), ("Replay a failure", "<replay> a failure"),    # and the phrases with a
           ("marks close in", "marks <to close> in"), ("Delta, close every", "Delta, <to close> every"),    # word of SOUND
           ("already close", "already <close>")]
NUM = r"\d+(?:\.\d+)?"


def say_math(m):
    """The words for one formula of a caption."""
    m = " ".join(m.split())
    if m in SAID:
        return SAID[m]
    if m in SYMBOL:
        return SYMBOL[m]
    if re.fullmatch(r"[A-Za-z]", m):
        return m
    k = re.fullmatch(rf"(?:({NUM}) \\times )?10\^\{{(-?)(\d+)\}}", m)
    if k:
        return (f"{k[1]} times " if k[1] else "") + ("ten to the minus " if k[2] else "ten to the power ") + k[3]
    k = re.fullmatch(rf"([A-Za-z]) (=|\\approx) ({NUM})", m)
    if k:
        return f"{say_math(k[1])} {'equals' if k[2] == '=' else 'about'} {k[3]}"
    raise ValueError(f"no reading for the formula ${m}$: add one to SAID in voice.py")


def spoken(tex):
    """The words the voice says for a caption written in LaTeX."""
    s = re.sub(r"\$([^$]*)\$", lambda k: say_math(k[1]), tex)
    for name, sound in RESPELL:
        s = s.replace(name, sound)
    s = re.sub(r"(?<![.\d])0\.0{5,}\d+\b", lambda k: f"{(Decimal(k[0]) * 10 ** 6).normalize():f} millionths", s)
    s = re.sub(r"(?<![.\d])\b(1[89]|20)(?!00)(\d\d)\b(?![.\d])", r"\1 \2", s)     # a year, read as two numbers
    s = re.sub(r"(?<=[.!?]) \(([^()]*)\)", r" \1", s)                             # a whole sentence in brackets
    s = re.sub(r" \(([^()]*)\)", r", \1,", s)                                     # an aside in brackets
    s = " ".join(re.sub(r",(?=[:;.,!?])", "", s).split())
    left = re.sub(r"[A-Za-z0-9 .,;:'?!-]", "", "".join(k for k in re.split(r"(<[a-z ]+>)", s) if k not in SOUND))
    if left:
        raise ValueError(f"cannot be spoken ({left!r} left over): {tex}")
    return s


_kokoro = None


def kokoro():
    global _kokoro
    if _kokoro is None:
        import logging
        from kokoro_onnx import Kokoro
        logging.getLogger("phonemizer").setLevel(logging.ERROR)     # it warns on every line whose word count changes
        _kokoro = Kokoro(str(MODEL / "kokoro-v1.0.onnx"), str(MODEL / "voices-v1.0.bin"))
    return _kokoro


def phonemes(words):
    """A line's phonemes (espeak-ng, American English), with the words of SOUND put in as given."""
    tok, out = kokoro().tokenizer, ""
    for part in re.split(r"(<[a-z ]+>(?:'s)?)", words):
        part = part.strip()
        if part.startswith("<"):
            out += " " + SOUND[part.removesuffix("'s")] + ("z" if part.endswith("'s") else "")
            continue
        lead = re.match(r"[,.;:!?]*", part)[0]
        rest = part[len(lead):].strip()
        out += lead + (" " + tok.phonemize(rest) if rest else "")
    return out.strip()


def clip(words, voice=None):
    """The sound of a line, made once for each speaker: its file and its length in seconds."""
    voice = voice or VOICE
    key = hashlib.sha1(f"kokoro-v1.0|{voice}|{words}".encode()).hexdigest()[:16]
    path = HERE / "media" / "voice" / voice / f"{key}.wav"
    if not path.exists():
        import numpy as np
        audio, rate = kokoro().create(phonemes(words), voice=voice, speed=1.0, is_phonemes=True)
        assert rate == RATE and len(audio) > 0
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(".part")
        with wave.open(str(part), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes((np.clip(HEADROOM * audio, -1.0, 1.0) * 32767).astype("<i2").tobytes())
        part.replace(path)
    with wave.open(str(path), "rb") as w:
        return path, w.getnframes() / w.getframerate()


def record(folder, scene, end, fps, lines):
    """Write a scene's record: called by the film when the scene has been built."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{scene}.json").write_text(json.dumps({"scene": scene, "end": round(end, 4), "fps": fps, "lines": lines}, indent=1) + "\n")


def place(scenes):
    """Each line's start on the joined film. scenes is [(name, video seconds, record)], each line of a record with its
    clip's length in "dur". Returns the starts [(seconds, line)], the problems found and the film's length."""
    at, starts, bad = 0.0, [], []
    for name, length, rec in scenes:
        if abs(length - rec["end"]) > 1.5 / rec["fps"]:
            bad.append(f"{name}: its record ends at {rec['end']:.3f} s and its video at {length:.3f} s. Render it again.")
        L = rec["lines"]
        for i, ln in enumerate(L):
            stop = L[i + 1]["t"] if i + 1 < len(L) else rec["end"]
            if ln["t"] + ln["dur"] > stop + 1e-3:
                bad.append(f"{name} line {i + 1}: {ln['dur']:.2f} s of speech in a slot of {stop - ln['t']:.2f} s: {ln['say'][:50]}")
            if not PACE[0] <= len(ln["say"]) / ln["dur"] <= PACE[1]:
                bad.append(f"{name} line {i + 1}: {len(ln['say'])} characters in {ln['dur']:.2f} s is not speech: {ln['say'][:50]}")
            starts.append((at + ln["t"], ln))
        at += length
    return starts, bad, at


def records(folder, names=None):
    folder = Path(folder)
    names = names or sorted(p.stem for p in folder.glob("*.json"))
    return [json.loads((folder / f"{n}.json").read_text()) for n in names]


def track(folder, videos, out, names):
    import numpy as np
    scenes = []
    for rec in records(folder, names):
        mp4 = Path(videos) / f"{rec['scene']}.mp4"
        length = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(mp4)],
                                      capture_output=True, text=True, check=True).stdout)
        for ln in rec["lines"]:
            ln["clip"], ln["dur"] = clip(ln["say"])
        scenes.append((rec["scene"], length, rec))
    starts, bad, total = place(scenes)
    if bad:
        print("\n".join(["track: FAIL"] + bad))
        return False
    pcm = np.zeros(int(round(total * RATE)), dtype=np.float32)
    for start, ln in starts:
        with wave.open(str(ln["clip"]), "rb") as w:
            a = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32)
        i = int(round(start * RATE))
        pcm[i:i + len(a)] += a[:len(pcm) - i]
    pcm *= 0.89 * 32767 / max(float(np.abs(pcm).max()), 1.0)                      # one gain for the whole film
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.astype("<i2").tobytes())
    speech = sum(ln["dur"] for _, ln in starts)
    print(f"track: {len(starts)} lines by {VOICE}, {speech:.0f} s of speech in {total:.0f} s, to {out}")
    return True


def selftest(mutate):
    cases = [(r"That interval is the echo period, $\Delta$.", "That interval is the echo period, Delta."),
             (r"In our world $\Delta \approx 3.445$, so", "In our world Delta is about 3.445, so"),
             (r"At $D = 131$ it is already $10^{20}$.", "At D equals 131 it is already ten to the power 20."),
             (r"by about $3 \times 10^{-11}$ near $D \approx 200$", "by about 3 times ten to the minus 11 near D about 200"),
             (r"give $a$, and the horizon data lose $a$'s half", "give <a>, and the horizon data lose <a>'s half"),
             (r"the method (Cochelin, 1994): expand $z$", "the method, Coshlann, 19 94: expand z"),
             (r"than the last. (This one is real, at $D = 38$.)", "than the last. This one is real, at D equals 38."),
             (r"by 0.44. Its true size is 0.00000037.", "by 0.44. Its true size is 0.37 millionths."),
             (r"Gundlach's 1997 construction, with 1000 terms", "Gundlock's 19 97 construction, with 1000 terms"),
             (r"with the true $\Delta$, close every one. It is already close", "with the true Delta, <to close> every one. It is already <close>")]
    if mutate == "formula":
        cases.append((r"the factor $\Gamma^3$", "the factor Gamma cubed"))
    ok = True
    for tex, want in cases:
        try:
            got = spoken(tex)
        except ValueError as e:
            got = f"error: {e}"
        if got != want:
            ok = False
            print(f"  got  {got}\n  want {want}")
    print(f"selftest: {len(cases)} captions read, {'PASS' if ok else 'FAIL'}")

    def rec(end, *lines):
        return {"end": end, "fps": 30, "lines": [{"t": t, "say": "x" * n, "dur": d} for t, n, d in lines]}
    scenes = [("A", 20.2 if mutate == "stale" else 20.0, rec(20.0, (1.0, 60, 6.5 if mutate == "overrun" else 5.0), (7.0, 120, 10.0))),
              ("B", 12.0, rec(12.0, (0.5, 100, 0.5 if mutate == "pace" else 8.0)))]
    starts, bad, total = place(scenes)
    fits = not bad and [s for s, _ in starts] == [1.0, 7.0, 20.5] and total == 32.0
    print(f"selftest: planted film of 2 scenes and 3 lines, {'PASS' if fits else 'FAIL'}")
    for b in bad:
        print(f"  {b}")
    return ok and fits


def main():
    a = sys.argv[1:]
    if a[:1] == ["selftest"]:
        sys.exit(0 if selftest(a[1].split("=", 1)[1] if len(a) > 1 and a[1].startswith("--mutate=") else "") else 1)
    if len(a) == 2 and a[0] in ("lines", "clips"):
        n = chars = off = 0
        seconds = 0.0
        for rec in records(a[1]):
            for ln in rec["lines"]:
                n, chars = n + 1, chars + len(ln["say"])
                if a[0] == "clips":
                    d = clip(ln["say"])[1]
                    seconds += d
                    if not PACE[0] <= len(ln["say"]) / d <= PACE[1]:
                        off += 1
                        print(f"{rec['scene']}: {len(ln['say'])} characters in {d:.2f} s is not speech: {ln['say'][:60]}")
                else:
                    print(f"{rec['scene']:14s} {ln['say']}")
        print(f"{n} lines, {chars} characters" + (f", {seconds:.0f} s of speech by {VOICE}, {off} off pace" if a[0] == "clips" else ""))
        sys.exit(1 if off else 0)
    if len(a) >= 5 and a[0] == "track":
        sys.exit(0 if track(a[1], a[2], a[3], a[4:]) else 1)
    sys.exit(__doc__)


if __name__ == "__main__":
    main()
