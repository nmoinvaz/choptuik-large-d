#!/usr/bin/env python3
"""The quest to large D: a film that explains the large-D calculation through the evolution of the solver, with
captions and a voice that reads them.

    video/build.sh [l|m|h]        # every scene, joined into video/dist/quest_large_d.mp4 (l: 480p15, h: 1080p30)

One Manim scene per station, in the order of SCENES. The curves are the repository's own: Delta(D) from
hp/data/delta_all.csv (computed rows), the highest D against time from paper/ai/main.tex and paper/ai/frontier.csv,
and the equations' grip on each harmonic from hp/data/held_grip.csv. The statements follow ALGORITHM.md, the notes and
paper/ai/main.tex. Curves marked "sketch" on screen are illustrations, not data.

Manim Community (The Manim Community Developers, https://www.manim.community), after 3Blue1Brown's manim. The formulas
are compiled by tectonic through the stand-ins in video/tools (this Manim's multi-part MathTex needs dvisvgm, so every
formula here is one string). The voice is voice.py's: say() starts a caption's line as the caption appears, and keeps
the caption until the line has been spoken.
"""
import csv
import math
import os
import re
from pathlib import Path

from manim import *

import voice

ROOT = Path(__file__).resolve().parent.parent
config.tex_template = TexTemplate(tex_compiler="pdflatex", output_format=".pdf")
config.background_color = "#0d1117"

BLUE_, GOLD_, RED_, GREEN_, GREY_, ORANGE_, PURPLE_ = "#58C4DD", "#FFD866", "#FF6B6B", "#7CE38B", "#8B949E", "#F78C6C", "#C792EA"
DEL4 = 3.4454                                   # Delta at D = 4, to four decimals (hp/data/delta_all.csv)
SCENES = ["S00_Title", "S01a_Wave", "S01b_Echo", "S03a_Dot", "S03b_Cone", "S03c_Numbers", "S03d_Join", "S03e_Newton", "S04_Walk", "S05_FirstWall", "S06_Amplifier",
          "S07_Basin", "S08_Taylor", "S09_FiveWalls", "S10_Climb", "S11_Basics", "S12_Outro"]


def delta_points():
    """(D, Delta) of the computed rows of the Delta table."""
    with open(ROOT / "hp/data/delta_all.csv") as f:
        return sorted((float(r["D"]), float(r["Delta_q0"])) for r in csv.DictReader(f) if r["kind"] != "series")


def frontier_points():
    """(day, highest D): the paper's points from the log, then the pushes of paper/ai/frontier.csv."""
    tex = (ROOT / "paper/ai/main.tex").read_text()
    block = tex[tex.index("ylabel={highest $D$ reached at large $D$}"):]
    block = block[block.index("coordinates {"):block.index("};")]
    pts = [(float(a), float(b)) for a, b in re.findall(r"\(([\d.]+),([\d.]+)\)", block)]
    with open(ROOT / "paper/ai/frontier.csv") as f:
        pts += [(float(r["day"]), float(r["D"])) for r in csv.DictReader(f)][1:]
    return pts


def grip(D_near):
    """The smallest singular value of each held harmonic's rows at the stored D nearest D_near."""
    rows = list(csv.DictReader(open(ROOT / "hp/data/held_grip.csv")))
    D0 = min({float(r["D"]) for r in rows}, key=lambda d: abs(d - D_near))
    out = {}
    for r in rows:
        if float(r["D"]) == D0:
            out[int(r["harmonic"])] = min(out.get(int(r["harmonic"]), 1.0), float(r["sigma_exact"]))
    return D0, out


def harmonics(D_near):
    """The centre field's odd harmonics at the stored solution nearest D_near: D and {k: (cos, sin)}."""
    rows = list(csv.DictReader(open(ROOT / "video/harmonics.csv")))
    D0 = min({float(r["D"]) for r in rows}, key=lambda d: abs(d - D_near))
    return D0, {int(r["k"]): (float(r["cos"]), float(r["sin"])) for r in rows if float(r["D"]) == D0}


def wave_of(co, th, ks=None):
    """The wave at echo phase th from the harmonics ks (all of them by default)."""
    return sum(a * math.cos(k * th) + b * math.sin(k * th) for k, (a, b) in co.items() if ks is None or k in ks)


def sci(v):
    """A number as LaTeX with two digits: 3.8\\times10^{-4}."""
    e = math.floor(math.log10(abs(v)))
    return f"{v / 10 ** e:.1f}" + r"\times10^{" + str(e) + "}"


def held_points():
    """(D, sqrt(D - 2) times harmonic 5's cosine coefficient) of the full solutions the held fit was made on."""
    with open(ROOT / "video/held_points.csv") as f:
        return [(float(r["D"]), float(r["h5"])) for r in csv.DictReader(f)]


def held_fit(name="held_fit"):
    """Harmonic 5's fitted curve, the same quantity against D, from the fit's stored Chebyshev series (hp/held_fit.py)."""
    import json

    import numpy as np
    d = json.load(open(ROOT / "hp/data" / f"{name}.json"))
    c = np.polynomial.chebyshev.Chebyshev(d["coeffs"]["5a"], domain=d["domain"])
    return lambda D: float(c(1.0 / (D - 2.0)))


def z_row(y, fields="pmac", horizon=64, W=11.0):
    """The row z to scale (W for 769 numbers), centred: the centre data, five nodes with a block of 32 numbers per
    field, the horizon data and Delta. Returns its parts by name and the whole as "all"."""
    unit, cols = W / 769.0, {"p": BLUE_, "m": BLUE_, "a": PURPLE_, "c": PURPLE_}
    x = -unit * (64 + 5 * 32 * len(fields) + horizon + 6) / 2

    def block(n, col, op, letter=None):
        nonlocal x
        r = Rectangle(width=n * unit, height=0.7, stroke_color=WHITE, stroke_width=1.2, fill_color=col, fill_opacity=op)
        r.move_to([x + n * unit / 2, y, 0])
        x += n * unit
        return VGroup(r, MathTex(letter, font_size=24).move_to(r)) if letter else r

    out = {"centre": block(64, BLUE_, 0.55), "nodes": [{f: block(32, cols[f], 0.35, f) for f in fields} for _ in range(5)],
           "horizon": block(horizon, ORANGE_, 0.55), "delta": block(6, GOLD_, 0.9)}
    out["all"] = VGroup(out["centre"], *[b for nd in out["nodes"] for b in nd.values()], out["horizon"], out["delta"])
    return out


def log10G(D):
    """Decades of the horizon's amplification, G = 0.7^-(D-3)."""
    return (D - 3.0) * math.log10(1.0 / 0.7)


def stairs(ax, pts, **kw):
    """A step curve through (x, y) points: flat to the next x, then up."""
    P = [ax.c2p(*pts[0])]
    for (_, y0), (x1, y1) in zip(pts, pts[1:]):
        P += [ax.c2p(x1, y0), ax.c2p(x1, y1)]
    return VMobject(**kw).set_points_as_corners(P)


CAP = TexTemplate(tex_compiler="pdflatex", output_format=".pdf")              # captions: lines up to 13 units wide at size 34
CAP.add_to_preamble(r"\setlength{\textwidth}{385pt}\setlength{\linewidth}{385pt}\setlength{\hsize}{385pt}")
CAP_LONG = TexTemplate(tex_compiler="pdflatex", output_format=".pdf")         # the same width at size 30, for the longest
CAP_LONG.add_to_preamble(r"\setlength{\textwidth}{436pt}\setlength{\linewidth}{436pt}\setlength{\hsize}{436pt}")
CPS = float(os.environ.get("QUEST_CPS", "12"))    # characters a second a caption is given to be read, after a second to find it
LEAD = 0.5                                        # the share of a caption's reading time given before its animation starts
WATCHING = 0.4                                    # a second of animation counts this much toward reading
LINES = os.environ.get("QUEST_LINES") == "1"      # only collect the voice's lines (build.sh s): no sound is made


class Film(Scene):
    stage = ""

    def setup(self):
        self.cap = None
        self.owed = 0.0                           # reading time the caption on screen is still owed
        self.lines = []                           # the voice's lines: when each starts, its words, its length
        self.quiet_at = 0.0                       # the scene time at which the voice has finished its line
        if self.stage:
            self.add(Tex(self.stage, font_size=26, color=GREY_).to_corner(UL, buff=0.35))

    def settle(self):
        """Wait until the caption's reading time is spent and the voice has finished its line."""
        need, self.owed = max(self.owed, self.quiet_at - self.renderer.time), 0.0
        if need > 0.05:
            self.wait(need)

    def speak(self, words):
        """Start the voice on a line, once its last line is over. A caption's line is started by say()."""
        self.settle()
        seconds = 0.0 if LINES or voice.VOICE == "off" else voice.clip(words)[1]
        self.lines.append({"t": round(self.renderer.time, 4), "say": words, "dur": round(seconds, 3)})
        if seconds:
            self.quiet_at = self.renderer.time + seconds + voice.GAP

    def tear_down(self):
        self.settle()                             # a scene never ends in the middle of a line
        folder = "lines" if LINES else f"{config.pixel_height}p{round(config.frame_rate)}"
        voice.record(Path(config.media_dir) / "voice" / folder, type(self).__name__, self.renderer.time, config.frame_rate, self.lines)
        super().tear_down()

    def play(self, *args, **kwargs):
        """Scene.play, with the caption's time kept: a caption is never taken away before it has been read and spoken."""
        leaving = self.cap is not None and any(isinstance(a, FadeOut) and self.cap in a.mobject.get_family() for a in args)
        if leaving:
            self.settle()
        t0 = self.renderer.time
        super().play(*args, **kwargs)
        if self.cap is not None:
            still = len(args) == 1 and isinstance(args[0], Wait)
            self.owed -= (self.renderer.time - t0) * (1.0 if still else WATCHING)

    def say(self, text, hold=None, color=WHITE):
        """A caption at the foot of the frame, read aloud. It is owed a second plus its length at CPS: all of it now, or
        with hold given, at least LEAD of it now and the rest (an animation's time counting WATCHING) before it is
        replaced. It also stays until the voice has finished it."""
        n = len(re.sub(r"\\[a-zA-Z]+|[${}^_\\]", "", text))
        new = Tex(text, font_size=34, color=color, tex_template=CAP) if n <= 200 else Tex(text, font_size=30, color=color, tex_template=CAP_LONG)
        if new.width > 13.3:
            new.scale_to_fit_width(13.3)
        new.to_edge(DOWN, buff=0.35)
        self.speak(voice.spoken(text))            # waits out the last caption first
        old, self.cap = self.cap, None
        self.play(FadeIn(new, shift=0.15 * UP), *([] if old is None else [FadeOut(old)]), run_time=0.5)
        self.cap = new
        self.owed = 1.0 + n / CPS
        self.wait(self.owed if hold is None else max(hold, LEAD * self.owed))

    def foot(self, text, at=None):
        """A reference, small and grey: under the station's name unless a place is given."""
        lines = [text] if isinstance(text, str) else text
        f = VGroup(*[Tex(t, font_size=20, color=GREY_) for t in lines]).arrange(DOWN, buff=0.06, aligned_edge=LEFT)
        return f.to_corner(UL, buff=0.35).shift(0.42 * DOWN) if at is None else f.move_to(at)

    def quote(self, text):
        """One of the prompts that pushed the work, top right."""
        return Tex(r"``\textit{" + text + r"}''", font_size=28, color=GOLD_).to_corner(UR, buff=0.35)

    def frontier_axes(self, x_max=35, y_max=280, x_len=10.6, y_len=4.6):
        ax = Axes(x_range=[0, x_max, 5], y_range=[0, y_max, 50], x_length=x_len, y_length=y_len, tips=False,
                  axis_config={"color": GREY_, "stroke_width": 2, "include_numbers": True, "font_size": 22})
        ax.shift(0.35 * UP)
        xl = Tex("days", font_size=24, color=GREY_).next_to(ax.x_axis, RIGHT, buff=0.15)
        yl = Tex("highest $D$ reached", font_size=24, color=GREY_).next_to(ax.y_axis, UP, buff=0.12).shift(0.9 * RIGHT)
        return ax, VGroup(xl, yl)

    def front(self, ax, pts, a, b, **kw):
        """The frontier from day a to day b, joined to its level at day a."""
        level = [p for p in pts if p[0] <= a][-1][1] if any(p[0] <= a for p in pts) else pts[0][1]
        seg = [(a, level)] + [p for p in pts if a < p[0] <= b]
        seg.append((b, seg[-1][1]))
        return stairs(ax, seg, color=BLUE_, stroke_width=4, **kw)


class S00_Title(Film):
    def construct(self):
        P = []
        for i in range(1600):
            v = 9.0 * i / 1599
            P.append([-7.0 + 14.0 * (1 - math.exp(-v)), 3.1 * math.sin(2 * math.pi * v / 1.5), 0])
        wave = VMobject(color=BLUE_, stroke_width=3, stroke_opacity=0.22).set_points_as_corners(P)
        title = Tex(r"The quest to large $D$", font_size=92)
        sub = Tex(r"Choptuik's echo period at large $D$,\\explained through the evolution of our solver", font_size=34, color=GREY_)
        cred = Tex(r"Nathan Moinvaziri \quad $\cdot$ \quad x.com/nmnvzr \quad $\cdot$ \quad made with Claude Fable 5.2",
                   font_size=28)
        VGroup(title, sub).arrange(DOWN, buff=0.45).shift(0.5 * UP)
        cred.to_edge(DOWN, buff=0.9)
        self.play(Create(wave), run_time=2.5)
        self.speak("The quest to large D. Choptuik's echo period at large D, explained through the evolution of our solver.")
        self.play(Write(title), run_time=1.6)
        self.play(FadeIn(sub, shift=0.2 * UP), FadeIn(cred, shift=0.2 * UP), run_time=1.0)
        self.wait(3.2)
        self.settle()
        self.play(FadeOut(VGroup(wave, title, sub, cred)), run_time=0.8)


class S01a_Wave(Film):
    stage = r"1 \quad A wave falls inward"

    def construct(self):
        O, R0, base, H, q = [-2.4, 1.55, 0.0], 1.45, -1.25, 0.5, 0.5       # the view from above, and the side view under it
        g = lambda v: math.exp(-0.5 * v * v)
        xs = [-1.8 + 3.6 * k / 360 for k in range(361)]

        def ring(r, col, op):
            return Circle(radius=max(r, 0.02), color=col, stroke_width=5, stroke_opacity=op).move_to(O)

        def humps(r, col, w, h, op):
            """the wave's height along a line through the centre: the ring cut twice, at -r and +r"""
            P = [[O[0] + x, base + h * (g((x - r) / w) + g((x + r) / w)), 0] for x in xs]
            return VMobject(color=col, stroke_width=4, stroke_opacity=op).set_points_as_corners(P)

        def links(sh):
            """thin lines from the ring's two edges down to the two humps"""
            if not sh or sh[0][1] == GREY_ or not 0.1 < sh[0][0] < 1.75:
                return VGroup()
            return VGroup(*[DashedLine([O[0] + sg * sh[0][0], O[1], 0], [O[0] + sg * sh[0][0], base, 0], color=GREY_, stroke_width=1.5,
                                       stroke_opacity=0.6, dash_length=0.06) for sg in (-1, 1)])

        state = {"f": lambda: []}                       # the shells now: (radius, colour, width, height, opacity), the main one first
        top = always_redraw(lambda: VGroup(*[ring(r, c, op) for r, c, w, h, op in reversed(state["f"]())]))
        side = always_redraw(lambda: VGroup(*[humps(r, c, w, h, op) for r, c, w, h, op in reversed(state["f"]())]))
        lk = always_redraw(lambda: links(state["f"]()))
        centre = Dot(O, color=WHITE, radius=0.06)
        guides = VGroup(*[Circle(radius=r, color=GREY_, stroke_width=1, stroke_opacity=0.25).move_to(O) for r in (0.5, 1.0, 1.45)])
        axis = Line([O[0] - 1.8, base, 0], [O[0] + 1.8, base, 0], color=GREY_, stroke_width=1.5)
        mid = DashedLine([O[0], O[1], 0], [O[0], base - 0.08, 0], color=WHITE, stroke_width=1, stroke_opacity=0.35, dash_length=0.05)
        cl = Tex("the centre", font_size=22, color=GREY_).next_to([O[0], base, 0], DOWN, buff=0.12)
        va = Tex(r"space, seen\\from above", font_size=22, color=GREY_).move_to([-5.75, O[1], 0])
        vs = Tex(r"the same wave\\from the side:\\its height on a line\\through the centre", font_size=22, color=GREY_).move_to([-5.75, base + 0.45, 0])
        sk = Tex("(a sketch)", font_size=18, color=GREY_).next_to(va, DOWN, buff=0.1)
        case = lambda text, col: Tex(text, font_size=34, color=col).move_to([3.5, 2.5, 0])
        legend = VGroup(Tex("falling in", font_size=24, color=BLUE_), Tex("heading out", font_size=24, color=GOLD_),
                        Tex("escaping", font_size=24, color=GREY_)).arrange(DOWN, buff=0.12, aligned_edge=LEFT).move_to([3.5, 1.45, 0])
        u = ValueTracker(0.0)

        def weak():
            v = u.get_value()
            if v < 0.45:
                return [(R0 * (1 - v / 0.45), BLUE_, 0.2, H, 1.0)]
            r = 2.1 * (v - 0.45) / 0.55
            return [(r, GOLD_, 0.2, H, max(0.0, min(1.0, (2.0 - r) / 0.5)))]

        def strong():
            r = R0 - (R0 - 0.3) * u.get_value()
            return [(r, BLUE_, 0.2, H * (1 + 0.6 * u.get_value()), 1.0)]

        self.play(FadeIn(VGroup(centre, guides, va, sk)), run_time=0.8)
        state["f"] = weak
        self.add(top)
        self.say(r"Start with the physics. A ball-shaped wave of energy falls inward on its own centre, from every side at once. Here it is seen from above.", hold=0.3)
        self.play(u.animate.set_value(0.1), run_time=1.4, rate_func=linear)
        self.play(Create(axis), Create(mid), FadeIn(cl), FadeIn(vs), run_time=0.8)
        self.add(side, lk)
        self.say(r"Below it, the same wave seen from the side: its height along a line through the centre. The ring is cut twice, so it shows as two humps, one on each side, closing in.", hold=0.3)
        self.play(u.animate.set_value(0.2), run_time=1.6, rate_func=linear)
        # weak: through the centre and out again
        lab = case("a weak wave", GREEN_)
        self.play(FadeIn(lab), FadeIn(legend[0]), FadeIn(legend[1]), run_time=0.4)
        self.say(r"A weak wave passes through the centre and flies back out. Nothing is left behind.", hold=0.2)
        self.play(u.animate.set_value(1.0), run_time=5.0, rate_func=linear)
        self.play(FadeOut(lab), run_time=0.4)
        # strong: a black hole
        u.set_value(0.0)
        state["f"] = strong
        lab = case("a strong wave", RED_)
        self.play(FadeIn(lab), run_time=0.5)
        self.say(r"A strong wave is trapped by its own gravity. It leaves a black hole at the centre.", hold=0.2)
        self.play(u.animate.set_value(1.0), run_time=2.6, rate_func=linear)
        hole = Circle(radius=0.3, color=GOLD_, stroke_width=3, fill_color=BLACK, fill_opacity=1.0).move_to(O)
        pit = Rectangle(width=0.6, height=1.15, color=GOLD_, stroke_width=3, fill_color=BLACK, fill_opacity=1.0).move_to([O[0], base + 0.575, 0])
        hl = Tex("a black hole", font_size=22, color=GOLD_).next_to(pit, RIGHT, buff=0.15)
        state["f"] = lambda: []
        self.play(GrowFromCenter(hole), GrowFromCenter(pit), FadeIn(hl), run_time=0.8)
        self.wait(2.0)
        self.play(FadeOut(VGroup(hole, pit, hl, lab)), run_time=0.6)
        # tuned between: in, through the centre, part escapes, the rest falls back in, smaller each round
        r_k = [R0 * q ** k for k in range(9)]
        T_k = [0.0]
        for k in range(8):
            T_k.append(T_k[-1] + 1.5 * r_k[k])            # a round: fall to the centre, then out to the next round's start
        tt = ValueTracker(0.0)

        def tuned():
            t, out = tt.get_value(), []
            k = max(i for i in range(9) if T_k[i] <= t)
            if k < 8:
                sdt, w = t - T_k[k], max(0.2 * q ** k, 0.03)
                out.append((r_k[k] - sdt, BLUE_, w, H, 1.0) if sdt < r_k[k] else (sdt - r_k[k], GOLD_, w, H, 1.0))
            for j in range(8):                             # the part that escaped at the end of round j
                r = r_k[j + 1] + (t - T_k[j + 1])
                if t >= T_k[j + 1] and r < 1.9:
                    out.append((r, GREY_, max(0.2 * q ** j, 0.03), 0.5 * H, max(0.0, 1.0 - (r - r_k[j + 1]) / 1.3)))
            return out

        state["f"] = tuned
        lab = case("tuned exactly between", GOLD_)
        self.play(FadeIn(lab), FadeIn(legend[2]), run_time=0.4)
        self.say(r"Tune the strength exactly between the two, and the wave does neither.")
        self.say(r"It falls in, passes through the centre, and heads back out. Part of it escapes for good.", hold=0.2)
        self.play(tt.animate.set_value(T_k[1]), run_time=4.5, rate_func=linear)
        self.say(r"The rest is pulled back by its own gravity and falls in again, smaller and faster. And again. And again.", hold=0.2)
        self.play(tt.animate.set_value(T_k[-1] + 0.6), run_time=6.0, rate_func=linear)
        self.say(r"Look at the side view as it plays again: each round, the humps come back just as tall, only narrower. Every round is a copy of the one before, shrunk.", hold=0.3)
        tt.set_value(0.0)
        self.play(tt.animate.set_value(T_k[-1] + 0.6), run_time=9.0, rate_func=linear)
        state["f"] = lambda: []
        star = Dot(O, color=RED_, radius=0.1)
        spot = Dot([O[0], base, 0], color=RED_, radius=0.1)
        self.play(FadeOut(centre), FadeIn(star, scale=3), FadeIn(spot, scale=3), Flash(star, color=RED_, flash_radius=0.5), run_time=0.8)
        sl = Tex("the collapse", font_size=26, color=RED_).next_to(star, UP, buff=0.15)
        self.play(FadeIn(sl), run_time=0.4)
        self.say(r"That repeating is the echoing. The rounds pile up at the centre at one single instant. That instant, at that point, is the collapse.")
        words = VGroup(Tex(r"the centre: the middle point\\the wave falls toward", font_size=28, tex_environment="flushleft"),
                       Tex(r"the collapse: the instant the\\echoes pile up there", font_size=28, color=RED_, tex_environment="flushleft"))
        words.arrange(DOWN, buff=0.4, aligned_edge=LEFT).move_to([3.5, -0.1, 0])
        self.play(FadeIn(words, shift=0.2 * UP), run_time=0.6)
        fn = self.foot("M. W. Choptuik, Phys. Rev. Lett. 70, 9 (1993)", at=[3.5, -1.55, 0])
        self.play(FadeIn(fn), run_time=0.3)
        self.say(r"Choptuik found this in 1993, and that every echo is the last one shrunk by the same factor. The whole film is about that factor.")
        self.play(FadeOut(VGroup(guides, axis, mid, cl, va, vs, sk, star, spot, sl, lab, legend, words, fn, self.cap)), run_time=0.7)


class S01b_Echo(Film):
    stage = r"2 \quad The echo"

    def construct(self):
        ax = Axes(x_range=[0, 1, 0.25], y_range=[-1.25, 1.25, 1], x_length=10.6, y_length=3.4, tips=False,
                  axis_config={"color": GREY_, "stroke_width": 2, "include_ticks": False}).shift(0.55 * UP)
        tl = Tex(r"time $\rightarrow$", font_size=26, color=GREY_).next_to(ax.c2p(0, -1.25), DR, buff=0.12)
        star = DashedLine(ax.c2p(1, -1.25), ax.c2p(1, 1.25), color=RED_, stroke_width=3)
        sl = Tex("the collapse", font_size=26, color=RED_).next_to(star, UP, buff=0.12)
        s = ValueTracker(0.0)

        def wave():
            P = []
            for i in range(900):
                v = 12.0 * i / 899
                P.append(ax.c2p(1 - math.exp(-v), math.sin(2 * math.pi * (s.get_value() + v) / DEL4)))
            return VMobject(color=BLUE_, stroke_width=3.5).set_points_as_corners(P)

        curve = wave()
        self.play(Create(ax), FadeIn(tl), run_time=1.0)
        self.say(r"Now stand at the centre of that tuned wave, and record its strength there, the field, as time runs toward the collapse.", hold=0.6)
        self.play(Create(curve), Create(star), FadeIn(sl), run_time=2.6)
        self.say(r"The field echoes. Each echo is a copy of the last, squeezed toward the moment of collapse.")
        live = always_redraw(wave)
        self.remove(curve)
        self.add(live)
        zoom = Tex(r"magnified $\times$", font_size=30, color=GOLD_)
        num = Integer(1, font_size=38, color=GOLD_)
        num.add_updater(lambda m: m.set_value(round(math.exp(s.get_value()))))
        VGroup(zoom, num).arrange(RIGHT, buff=0.15).next_to(ax, UP, buff=0.1).shift(3.2 * LEFT)
        num.add_updater(lambda m: m.next_to(zoom, RIGHT, buff=0.15))
        self.play(FadeIn(zoom), FadeIn(num), run_time=0.5)
        self.say(r"Zoom in on the collapse by just the right factor, and the picture repeats exactly.", hold=0.4)
        self.play(s.animate.set_value(DEL4), run_time=4.0, rate_func=smooth)
        self.wait(0.6)
        self.play(s.animate.set_value(2 * DEL4), run_time=3.2, rate_func=smooth)
        self.wait(0.8)
        num.clear_updaters()
        self.play(FadeOut(VGroup(live, star, sl, tl, zoom, num, ax)), run_time=0.7)

        ax2 = Axes(x_range=[0, 4 * DEL4, DEL4], y_range=[-1.25, 1.25, 1], x_length=10.6, y_length=3.4, tips=False,
                   axis_config={"color": GREY_, "stroke_width": 2, "include_ticks": True}).shift(0.55 * UP)
        tau = Tex(r"logarithmic time $\tau = -\ln(\textrm{time left})$", font_size=26, color=GREY_)
        tau.next_to(ax2.c2p(4 * DEL4, -1.25), DL, buff=0.12)
        per = ax2.plot(lambda t: math.sin(2 * math.pi * t / DEL4), x_range=[0, 4 * DEL4, 0.02], color=BLUE_, stroke_width=3.5)
        br = BraceBetweenPoints(ax2.c2p(DEL4, 1.05), ax2.c2p(2 * DEL4, 1.05), direction=UP, color=GOLD_)
        bl = MathTex(r"\Delta", font_size=44, color=GOLD_).next_to(br, UP, buff=0.1)
        self.play(Create(ax2), FadeIn(tau), run_time=0.8)
        self.play(Create(per), run_time=2.0)
        self.say(r"Measure time by how often the time left has shrunk, logarithmic time, and the echoes come at even intervals. That interval is the echo period, $\Delta$.", hold=0.5)
        self.play(GrowFromCenter(br), FadeIn(bl), run_time=0.8)
        self.wait(2.6)
        self.say(r"The zoom factor is $e^{\Delta}$. In our world $\Delta \approx 3.445$, so every echo is about " + f"{math.exp(DEL4):.0f}" + " times smaller than the last.")
        self.play(FadeOut(VGroup(ax2, tau, per, br, bl, self.cap)), run_time=0.7)


CUTS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]     # the centre, the five nodes of the large-D chain, the horizon
OFFS = [0.26, -0.2, 0.22, -0.17, 0.15, -0.24]   # a sketch's mismatches, one per piece
PCOL = [BLUE_, GREEN_, GREEN_, GREEN_, GREEN_, ORANGE_]


def profile(v):
    """A smooth curve standing for one field along x (a sketch)."""
    return 0.5 * math.sin(2.6 * v + 0.4) + 0.22 * math.cos(7.0 * v)


def piece(line, i, e, lift=1.0):
    """Piece i of the solution along x, off by e times its mismatch. The horizon's piece runs inward."""
    a, b = CUTS[i], CUTS[i + 1]
    ks = range(41) if i < 5 else range(40, -1, -1)
    P = [line.n2p(a + (b - a) * k / 40) + UP * (lift + profile(a + (b - a) * k / 40) + e * OFFS[i]) for k in ks]
    return VMobject(color=PCOL[i], stroke_width=4).set_points_as_corners(P)


def gaps(line, e, lift=1.0):
    """The mismatch at each node: where the piece that arrives and the piece that leaves disagree."""
    out = VGroup()
    for i in range(5):
        v = CUTS[i + 1]
        y0, y1 = lift + profile(v) + e * OFFS[i], lift + profile(v) + e * OFFS[i + 1]
        out.add(Line(line.n2p(v) + UP * y0, line.n2p(v) + UP * y1, color=RED_, stroke_width=6))
    return out


def x_line(y=-1.9, height=2.2, nodes="the five nodes"):
    """The x axis from the centre to the horizon as the bottom edge of the frame that also holds the image of one echo:
    the five node columns as upright lines, the centre and the horizon as its coloured sides. Returns the axis and
    the group (frame, columns, sides, end labels, node label, upright label)."""
    line = NumberLine(x_range=[0, 1, 0.1], length=10.0, color=GREY_, include_ticks=False).shift(y * UP)
    frame = Rectangle(width=10.0, height=height, color=GREY_, stroke_width=1.5).move_to([0, y + height / 2, 0])
    cols = VGroup(*[Line(line.n2p(v), line.n2p(v) + height * UP, color=WHITE, stroke_width=1.5, stroke_opacity=0.45) for v in CUTS[1:6]])
    sides = VGroup(Line(line.n2p(0), line.n2p(0) + height * UP, color=BLUE_, stroke_width=5),
                   Line(line.n2p(1), line.n2p(1) + height * UP, color=ORANGE_, stroke_width=5))
    ends = VGroup(Tex(r"the centre, $x = 0$", font_size=24, color=BLUE_).next_to(line.n2p(0), DOWN, buff=0.15),
                  Tex(r"the horizon, $x = 1$", font_size=24, color=ORANGE_).next_to(line.n2p(1), DOWN, buff=0.15))
    nl = Tex(nodes, font_size=22, color=GREY_).next_to(line.n2p(0.5), DOWN, buff=0.15)
    up = Tex(r"the value of one field $\uparrow$", font_size=20, color=GREY_).next_to(frame.get_corner(UL), UP, buff=0.08).align_to(frame, LEFT)
    return line, VGroup(frame, cols, sides, ends, nl, up)


def low_axes():
    """Delta against D at low D, where single points can be told apart."""
    ax = Axes(x_range=[2, 14, 2], y_range=[2.2, 3.6, 0.2], x_length=9.6, y_length=4.2, tips=False,
              axis_config={"color": GREY_, "stroke_width": 2, "include_numbers": True, "font_size": 22},
              y_axis_config={"decimal_number_config": {"num_decimal_places": 1}}).shift(0.5 * UP)
    xl = Tex("spacetime dimensions $D$", font_size=26, color=GREY_).next_to(ax.x_axis, DOWN, buff=0.3)
    yl = MathTex(r"\Delta", font_size=36, color=GREY_).next_to(ax.y_axis, UP, buff=0.1)
    return ax, VGroup(xl, yl)


def strip_cells(RC, RW, RH, co, ncol=28, nrow=16):
    """One field over one echo as an image: x across, the time within the echo upward, gold where the field is positive
    and blue where it is negative. A sketch built on the real centre wave: p is close to x times p1 at the nodes
    (hp/plot_wave_family.py, step 4)."""
    top = max(abs(wave_of(co, 2 * math.pi * k / 200)) for k in range(200))
    cells = VGroup()
    for i in range(ncol):
        xv = (i + 0.5) / ncol
        for j in range(nrow):
            val = xv * wave_of(co, 2 * math.pi * (j + 0.5) / nrow) / top
            cell = Rectangle(width=RW / ncol + 0.01, height=RH / nrow + 0.01, stroke_width=0,
                             fill_color=interpolate_color(ManimColor(BLUE_), ManimColor(GOLD_), 0.5 + 0.5 * val),
                             fill_opacity=min(1.0, 0.12 + 0.95 * abs(val)))
            cells.add(cell.move_to([RC[0] - RW / 2 + xv * RW, RC[1] - RH / 2 + (j + 0.5) * RH / nrow, 0]))
    return cells


class S03a_Dot(Film):
    stage = r"3 \quad The question"

    def construct(self):
        pts = delta_points()
        ax, lab = low_axes()
        at = lambda D: min(pts, key=lambda q: abs(q[0] - D))
        D4 = at(4.0)
        ours = Dot(ax.c2p(*D4), color=GOLD_, radius=0.11)
        ol = Tex(f"our world: $D = 4$, $\\Delta = {D4[1]:.3f}$", font_size=28, color=GOLD_).next_to(ours, RIGHT, buff=0.2)
        self.play(Create(ax), FadeIn(lab), run_time=1.0)
        self.play(FadeIn(ours, scale=2), FadeIn(ol), run_time=0.6)
        self.say(r"All of that was one collapse, in one world. Our world has $D = 4$ dimensions, three of space and one of time, and its echo period is the one dot on this graph.")
        curve = VMobject(color=BLUE_, stroke_width=3, stroke_opacity=0.35).set_points_as_corners(
            [ax.c2p(*q) for q in pts if 3.9 <= q[0] <= 13.0])
        dots = VGroup(*[Dot(ax.c2p(*at(D)), color=BLUE_, radius=0.07) for D in range(5, 14)])
        self.play(Create(curve), LaggedStart(*[FadeIn(d, scale=2.5) for d in dots], lag_ratio=0.18), run_time=2.4)
        self.say(r"The same collapse can be worked out for a world with any number of dimensions $D$. Each $D$ has its own echo period $\Delta$.")
        self.say(r"So this graph is not a picture of a collapse. Each dot stands for a whole collapse, in a world with that $D$.")
        self.say(r"And each dot is one computation, called a solve: for that $D$, find the echoing solution and read off its $\Delta$.")
        end = at(13.0)
        arrow = Arrow(ax.c2p(13.3, end[1] - 0.02), ax.c2p(15.2, end[1] - 0.09), color=GOLD_, stroke_width=5, buff=0)
        far = Tex(r"on to $D = 100,\ 200,\ 400,\ \ldots$", font_size=28, color=GOLD_).next_to(arrow, UP, buff=0.55).align_to([6.8, 0, 0], RIGHT)
        self.play(GrowArrow(arrow), FadeIn(far), run_time=1.0)
        self.say(r"The quest: follow $\Delta(D)$ far off the right-hand edge of this graph, into worlds with hundreds of dimensions.")
        self.say(r"That takes a solver: a program that finds $\Delta$ for a given $D$. First, what one solve has to find. Then, how each wall on the way reshaped the solver.")
        self.play(FadeOut(VGroup(ax, lab, curve, dots, ours, ol, arrow, far, self.cap)), run_time=0.7)


class S03b_Cone(Film):
    stage = r"4 \quad The map, the horizon and $x$"

    def construct(self):
        C, T = [-0.6, 2.55, 0.0], 4.3
        P = lambda r, tt: [C[0] + r, C[1] - T + tt, 0.0]          # the map: distance r across, time tt upward
        O, K, W = [-4.55, 1.5, 0.0], 1.1 / T, 7.3                 # the view from above: its centre and scale
        yb, XS, Hs, ws = -1.2, 1.6, 0.5, 0.07                     # the side view under it: baseline, half width, hump height and width
        xs = [-XS + 2 * XS * k / 480 for k in range(481)]
        bell = lambda v: math.exp(-0.5 * v * v)
        t = ValueTracker(0.0)
        now = lambda: t.get_value()
        centre = Line(P(0, 0), C, color=BLUE_, stroke_width=4)
        base = Line(P(0, 0), P(W, 0), color=GREY_, stroke_width=2)
        tl = Tex(r"time $\uparrow$", font_size=22, color=GREY_).next_to(centre, LEFT, buff=0.1)
        rl = Tex(r"distance from the centre $\rightarrow$", font_size=22, color=GREY_).next_to(base, DOWN, buff=0.42).align_to(base, RIGHT)
        ml = Tex("a map: where, and when", font_size=22, color=GREY_).to_corner(UR, buff=0.35)
        vl = Tex("space, seen from above", font_size=22, color=GREY_).move_to([O[0], 3.15, 0])
        dotc = Dot(O, color=WHITE, radius=0.06)
        saxis = Line([O[0] - XS, yb, 0], [O[0] + XS, yb, 0], color=GREY_, stroke_width=1.5)
        smid = DashedLine(O, [O[0], yb - 0.08, 0], color=WHITE, stroke_width=1, stroke_opacity=0.35, dash_length=0.05)
        dl = VGroup(Tex("the centre", font_size=20, color=BLUE_).next_to([O[0], yb, 0], DOWN, buff=0.12),
                    Tex("side view", font_size=20, color=GREY_).next_to([O[0] - XS + 0.45, yb, 0], DOWN, buff=0.12))
        nowline = always_redraw(lambda: DashedLine(P(0, now()), P(W, now()), color=WHITE, stroke_width=2, stroke_opacity=0.75))
        nowl = always_redraw(lambda: Tex("now", font_size=20).next_to(P(W, now()), UP, buff=0.04).shift(0.3 * LEFT))

        def humps(shells):
            """the side view of shells (distance from the centre, colour, opacity, width): each a pair of humps"""
            out = VGroup()
            for r, col, op, w in shells:
                pts = [[O[0] + v, yb + Hs * (bell((v - K * r) / w) + bell((v + K * r) / w)), 0] for v in xs]
                out.add(VMobject(color=col, stroke_width=4, stroke_opacity=op).set_points_as_corners(pts))
            return out

        def shell(r0, col):
            """a shell of the wave that starts at distance r0: its ring from above, its humps from the side, its track on the map"""
            ring = always_redraw(lambda: Circle(radius=K * (r0 - now()), color=col, stroke_width=5).move_to(O)
                                 if r0 - now() > 0.03 else VGroup())
            hump = always_redraw(lambda: humps([(r0 - now(), col, 1.0, ws)]) if r0 - now() > 0.03 else VGroup())
            track = always_redraw(lambda: Line(P(r0, 0), P(r0 - min(now(), r0) - 1e-3, min(now(), r0) + 1e-3), color=col, stroke_width=4))
            head = always_redraw(lambda: Dot(P(r0 - min(now(), r0), min(now(), r0)), color=col, radius=0.08))
            return VGroup(ring, hump, track, head)

        self.play(FadeIn(VGroup(dotc, vl, saxis, smid, dl)), Create(centre), Create(base), FadeIn(VGroup(tl, rl, ml)), run_time=1.2)
        self.add(nowline, nowl)
        self.say(r"Go back to one collapse and look closer. On the left are the two views from before: space seen from above, and the same wave seen from the side.")
        self.say(r"On the right is something new: a map, with distance from the centre across and time upward. The white line on the map marks the moment shown on the left.")
        A = shell(2.9, BLUE_)
        self.add(A)
        self.say(r"A shell of the wave falls inward. It moves at the speed of light, and nothing moves faster. On the map, that is a diagonal.", hold=0.3)
        self.play(t.animate.set_value(2.9), run_time=3.6, rate_func=linear)
        self.play(Flash(dotc, color=BLUE_, flash_radius=0.3), Indicate(centre, color=BLUE_), run_time=0.8)
        cl = Tex("the centre", font_size=22, color=BLUE_).next_to(P(0, 0), DOWN, buff=0.1)
        self.play(FadeIn(cl), run_time=0.3)
        self.say(r"The centre is the wave's middle point. It stays put, so on the map it is the upright line at the left.")
        star = Dot(C, color=RED_, radius=0.1)
        sl = Tex("the collapse", font_size=24, color=RED_).next_to(star, UP, buff=0.1)
        hit = always_redraw(lambda: VGroup(Dot(O, color=RED_, radius=0.1), Tex("the collapse", font_size=22, color=RED_).next_to(O, UP, buff=0.15),
                                           Dot([O[0], yb, 0], color=RED_, radius=0.1)) if now() > T - 0.02 else VGroup())
        self.add(hit)
        self.play(t.animate.set_value(T), run_time=1.6, rate_func=linear)
        self.play(FadeIn(star, scale=3), FadeIn(sl), Flash(dotc, color=RED_, flash_radius=0.4), run_time=0.8)
        self.say(r"The collapse is the instant the rounds of echoing pile up at the centre. On the map it is one point: the top of that line.")
        # which shells can take part?
        self.say(r"Now rewind and ask: which parts of the wave can still take part in the collapse?", hold=0.2)
        self.play(t.animate.set_value(0.0), run_time=1.2)
        B = shell(T, ORANGE_)
        self.add(B)
        self.wait(0.6)
        self.play(t.animate.set_value(T), run_time=4.2, rate_func=linear)
        self.play(Flash(star, color=ORANGE_), Flash(dotc, color=ORANGE_, flash_radius=0.4), run_time=0.7)
        self.say(r"This shell reaches the centre exactly at the collapse. It is the last one that arrives in time.")
        self.play(t.animate.set_value(0.0), FadeOut(A), run_time=1.2)
        Cs = shell(5.4, PURPLE_)
        self.add(Cs)
        self.say(r"A shell that starts farther out is still on its way when the collapse happens.", hold=0.2)
        self.play(t.animate.set_value(T), run_time=4.2, rate_func=linear)
        late = VGroup(Tex("too late", font_size=24, color=PURPLE_).next_to(P(5.4 - T, T), DOWN, buff=0.2).shift(0.5 * RIGHT),
                      Tex("too late", font_size=22, color=PURPLE_).move_to([O[0], O[1] - K * (5.4 - T) - 0.3, 0]))
        self.play(FadeIn(late), run_time=0.4)
        self.say(r"It comes too late to matter. And since nothing outruns the wave, nothing else from out there can matter either.")
        # the horizon: the closing edge of what can still matter
        self.play(FadeOut(late), FadeOut(Cs), t.animate.set_value(0.0), run_time=1.2)
        inside_map = Polygon(P(0, 0), P(T, 0), C, stroke_width=0, fill_color=ORANGE_, fill_opacity=0.16)

        def inside():
            r = K * (T - now())
            if T - now() <= 0.03:
                return VGroup()
            return VGroup(Circle(radius=r, stroke_width=0, fill_color=ORANGE_, fill_opacity=0.16).move_to(O),
                          Rectangle(width=2 * r, height=0.95, stroke_width=0, fill_color=ORANGE_, fill_opacity=0.16).move_to([O[0], yb + 0.475, 0]),
                          Line([O[0] - r, yb, 0], [O[0] - r, yb + 0.95, 0], color=ORANGE_, stroke_width=3),
                          Line([O[0] + r, yb, 0], [O[0] + r, yb + 0.95, 0], color=ORANGE_, stroke_width=3))

        inside_view = always_redraw(inside)
        hz = Tex("the horizon", font_size=26, color=ORANGE_).next_to(Line(P(T, 0), C).get_center(), RIGHT, buff=0.3)
        hv = Tex("the horizon", font_size=22, color=ORANGE_).move_to([O[0], O[1] + K * T + 0.2, 0])
        self.add(inside_view)
        self.play(FadeIn(inside_map), FadeIn(hz), FadeIn(hv), run_time=0.7)
        self.say(r"So at every moment there is an edge: a sphere around the centre, closing in at the speed of light. From the side it is two walls closing on the centre. Only what lies inside can still affect the collapse.", hold=0.3)
        self.play(FadeOut(hv), t.animate.set_value(0.72 * T), run_time=4.0, rate_func=linear)
        self.wait(1.0)
        self.say(r"That closing edge is the horizon, also called the light cone of the collapse. On the map it is the sloping line. It is not a black hole's horizon.")
        # the band between a falling shell and the horizon, and why it is not the whole problem
        self.play(t.animate.set_value(0.0), run_time=1.2)
        r0 = 1.4
        legs, turns, tt, L = [(0.0, r0, r0, 0.0, BLUE_)], [], r0, T - r0                # (from time, to time, from r, to r)
        for _ in range(7):
            s_ = 0.22 * L                                # each round turns back at the same x, 0.22 / 0.78
            legs += [(tt, tt + s_, 0.0, s_, GOLD_), (tt + s_, tt + 2 * s_, s_, 0.0, BLUE_)]
            turns.append((tt + s_, s_))
            tt, L = tt + 2 * s_, L - 2 * s_
        width = lambda time: ws * min(1.0, max((T - time) / (T - r0), 0.3))             # the rounds grow narrower with the time left

        def r_at(time):
            for t0, t1, ra, rb, c_ in legs:
                if t0 <= time <= t1:
                    return ra + (rb - ra) * (time - t0) / (t1 - t0), c_
            return 0.0, BLUE_

        def escaped():
            """the parts that have escaped by now: (distance, colour, opacity, width)"""
            return [(se + now() - te, GREY_, 0.6 * max(0.0, 1.0 - (now() - te) / 1.8), width(te)) for te, se in turns
                    if now() > te and K * (se + now() - te) < XS]

        def zigzag():
            g = VGroup()
            for t0, t1, ra, rb, c_ in legs:
                f = min(1.0, (now() - t0) / (t1 - t0))
                if f > 0.02:
                    g.add(Line(P(ra, t0), P(ra + (rb - ra) * f, t0 + (t1 - t0) * f), color=c_, stroke_width=4))
            for te, se in turns:                                                          # the part that escapes
                d = min(now() - te, 0.8 * (T - te - se), T - te - 0.02)
                if d > 0.02:
                    g.add(Line(P(se, te), P(se + d, te + d), color=GREY_, stroke_width=2))
            return g

        zig = always_redraw(zigzag)
        zr = always_redraw(lambda: VGroup(*[Circle(radius=K * r, color=c_, stroke_width=3, stroke_opacity=op).move_to(O) for r, c_, op, w in escaped()],
                                          Circle(radius=max(K * r_at(now())[0], 0.03), color=r_at(now())[1], stroke_width=5).move_to(O),
                                          humps(escaped() + [(r_at(now())[0], r_at(now())[1], 1.0, width(now()))])))
        zk = VGroup(Tex("falling in", font_size=22, color=BLUE_), Tex("heading out", font_size=22, color=GOLD_),
                    Tex("escaping", font_size=22, color=GREY_), Tex("(a sketch)", font_size=18, color=GREY_))
        zk.arrange(DOWN, buff=0.08, aligned_edge=LEFT).move_to(P(5.9, 2.6))
        self.add(zig, zr)
        tp = 0.2 * r0                                                                     # a moment early in the fall
        self.play(t.animate.set_value(tp), run_time=1.0, rate_func=linear)
        band = Polygon(P(r0, 0), P(T, 0), P(T - r0, r0), P(0, r0), stroke_width=0, fill_color=BLUE_, fill_opacity=0.5)
        bandl = Tex(r"the band between a falling\\shell and the horizon", font_size=22, color=BLUE_).move_to(P(5.7, 1.0))
        ann = Annulus(inner_radius=K * (r0 - tp), outer_radius=K * (T - tp), color=BLUE_, fill_opacity=0.5, stroke_width=0).move_to(O)
        gap = VGroup(*[Rectangle(width=K * (T - r0), height=0.95, stroke_width=0, fill_color=BLUE_, fill_opacity=0.5).move_to(
            [O[0] + sg * K * (r0 - tp + (T - r0) / 2), yb + 0.475, 0]) for sg in (-1, 1)])
        cut = Line(P(r0 - tp, tp), P(T - tp, tp), color=WHITE, stroke_width=7)
        self.play(FadeIn(band), FadeIn(bandl), FadeIn(ann), FadeIn(gap), FadeIn(cut), run_time=0.6)
        self.say(r"Look at the band between a falling shell and the horizon, in blue in all three views: a ring-shaped region from above, two gaps from the side, and a slanted strip on the map.")
        self.say(r"The thick white line is the band at this moment. It holds the wave that is still on its way in. Is that band all the solver needs?")
        self.play(band.animate.set_fill(opacity=0.15), FadeOut(bandl), FadeOut(ann), FadeOut(gap), FadeOut(cut), FadeIn(zk), run_time=0.5)
        self.say(r"No, because a shell is not finished when it reaches the centre. It passes through and heads back out. Part escapes across the horizon, and what gravity turns back falls in again.", hold=0.3)
        self.play(t.animate.set_value(0.985 * T), run_time=9.0, rate_func=linear)
        self.wait(0.6)
        self.say(r"That zig-zag is the echoing, drawn on the map: round after round, each smaller, right up to the collapse. From the side, the humps shuttle between the centre and the closing walls, and the grey ones slip out.")
        self.say(r"So everything in the band ends up at the centre, and comes back out across the whole inside. The solver needs all of it, from the centre to the horizon.")
        zig.clear_updaters()
        zr.clear_updaters()
        self.play(FadeOut(VGroup(zig, zr, zk, band)), run_time=0.5)
        # x: how far out, as a fraction of the way to the horizon
        self.play(t.animate.set_value(0.0), run_time=1.2)
        rays = VGroup(*[Line(P(v * T, 0), C, color=GREY_, stroke_width=1.5) for v in CUTS[1:6]])
        marks = always_redraw(lambda: VGroup(
            *[Circle(radius=K * v * (T - now()), color=GREY_, stroke_width=1.5).move_to(O) for v in CUTS[1:6]],
            *[Line([O[0] + sg * K * v * (T - now()), yb - 0.1, 0], [O[0] + sg * K * v * (T - now()), yb + 0.1, 0], color=WHITE, stroke_width=2)
              for v in CUTS[1:6] for sg in (-1, 1)]) if T - now() > 0.2 else VGroup())
        ruler = always_redraw(lambda: VGroup(Line(P(0, now()), P(T - now(), now()), color=WHITE, stroke_width=3),
                                             *[Line(P(v * (T - now()), now() - 0.09), P(v * (T - now()), now() + 0.09), color=WHITE, stroke_width=2.5)
                                               for v in [0.0] + CUTS[1:6] + [1.0]]))
        x0 = Tex(r"$x = 0$", font_size=24, color=BLUE_).next_to(cl, DOWN, buff=0.04)
        x1 = Tex(r"$x = 1$", font_size=24, color=ORANGE_).next_to(P(T, 0), DOWN, buff=0.1)
        xh = Tex(r"$x = 0.5$", font_size=22, color=GREY_).next_to(P(0.5 * T, 0), DOWN, buff=0.1)
        self.play(LaggedStart(*[Create(r) for r in rays], lag_ratio=0.1), FadeIn(marks), FadeIn(ruler), FadeIn(VGroup(x0, x1, xh)), run_time=1.4)
        self.say(r"Inside the horizon, say where a point is by how far out it is, as a fraction: $x = 0$ at the centre, $x = 1$ on the horizon, $x = 0.5$ halfway.")
        self.say(r"As the horizon closes in, those marks close in with it.", hold=0.2)
        self.play(t.animate.set_value(0.8 * T), run_time=5.0, rate_func=linear)
        self.say(r"Each round of echoing is the last one, shrunk. In $x$ the shrinking is taken out, so every echo looks the same.")
        self.say(r"Keep the horizon in mind. At large $D$, nearly every wall ahead begins at that edge.")
        self.say(r"So the solver works in $x$, between the centre at 0 and the horizon at 1.")
        for m in (nowline, nowl, B, hit, inside_view, marks, ruler):
            m.clear_updaters()
        self.play(FadeOut(VGroup(dotc, dl, vl, saxis, smid, centre, base, tl, rl, ml, cl, star, sl, B, hit, inside_map, inside_view, hz, rays, marks, ruler,
                                 x0, x1, xh, nowline, nowl, self.cap)), run_time=0.7)


class S03c_Numbers(Film):
    stage = r"5 \quad The solution, written as numbers"

    def construct(self):
        D38, co = harmonics(38.0)
        top = max(abs(wave_of(co, 2 * math.pi * k / 200)) for k in range(200))
        # 1. what a solution is: every field at every point inside the horizon, echo stacked on echo
        C, T, Wd = [-4.9, 2.5, 0.0], 4.4, 3.7
        pt = lambda v, t: [C[0] + v * Wd * (1 - t / T), C[1] - T + t, 0.0]
        centre = Line(pt(0, 0), C, color=BLUE_, stroke_width=4)
        cone = Line(pt(1, 0), C, color=ORANGE_, stroke_width=4)
        star = Dot(C, color=RED_, radius=0.09)
        sl = Tex("the collapse", font_size=24, color=RED_).next_to(star, RIGHT, buff=0.15)
        c0 = Tex(r"the centre\\$x = 0$", font_size=22, color=BLUE_).next_to(pt(0, 0), DOWN, buff=0.1)
        c1 = Tex(r"the horizon\\$x = 1$", font_size=22, color=ORANGE_).next_to(pt(1, 0), DOWN, buff=0.1)
        sk = Tex("the map (a sketch)", font_size=20, color=GREY_).move_to([C[0] + 3.4, C[1] - 0.1, 0])
        left = lambda k: T * 0.62 ** k                                  # time left at half echo k
        tk = [T - left(k) for k in range(17)]
        bands = VGroup(*[Polygon(pt(0, tk[k]), pt(1, tk[k]), pt(1, tk[k + 1]), pt(0, tk[k + 1]), stroke_width=0,
                                 fill_color=GOLD_ if k % 2 == 0 else BLUE_, fill_opacity=0.45) for k in range(16)])
        swings = lambda tv: math.log(T / (T - tv)) / math.log(1 / 0.62)          # the swings gone by at time tv
        xc = C[0] - 0.62
        chirp = VGroup(*[VMobject(color=GOLD_ if k % 2 == 0 else BLUE_, stroke_width=3.5).set_points_as_corners(
            [[xc + 0.42 * math.sin(math.pi * swings(tk[k] + (tk[k + 1] - tk[k]) * i / 24)), C[1] - T + tk[k] + (tk[k + 1] - tk[k]) * i / 24, 0]
             for i in range(25)]) for k in range(16)])
        beside = VGroup(chirp, DashedLine([xc, C[1] - T, 0], [xc, C[1], 0], color=GREY_, stroke_width=1, dash_length=0.05),
                        Tex(r"the wave\\at the centre\\over time", font_size=18, color=GREY_).move_to([-6.53, 0.3, 0]))
        themap = VGroup(centre, cone, star, sl, c0, c1, sk)
        self.play(FadeIn(themap), run_time=0.9)
        self.say(r"What must it find there? What the wave is doing at every point inside the horizon, at every moment. That is a solution.", hold=0.4)
        self.play(LaggedStart(*[FadeIn(bd) for bd in bands], lag_ratio=0.25), run_time=3.0)
        self.say(r"On the map, the swings of the wave stack up toward the collapse: crest, trough, crest, each a shrunken copy of the one before.")
        self.play(FadeIn(beside[1]), FadeIn(beside[2]), Create(chirp), run_time=3.0, rate_func=linear)
        self.say(r"Beside the map is the wave at the centre, drawn against time: the echo of station 2, turned upright. Each crest lines up with a gold band, and each trough with a blue one.")
        self.say(r"Only four or five bands can be seen, but there is no end to them. Each is a fixed fraction of the one before, so they crowd toward the collapse and soon are too thin to draw.")
        self.say(r"The sketch is gentle, too. In our world each echo is 31 times shorter than the last, so drawn to scale only the first would show.")
        # 2. one echo, redrawn with x across and the time within the echo upward
        RW, RH, RC = 4.6, 2.4, [3.3, 0.95, 0.0]
        vs, fs = [0.0] + CUTS[1:6] + [1.0], [0.0, 0.5, 1.0]
        tf = lambda f: T - left(2) * (left(4) / left(2)) ** f              # the time at fraction f of this echo
        gmap = VGroup(*[Line(pt(v, tf(0)), pt(v, tf(1)), color=WHITE, stroke_width=2) for v in vs],
                      *[Line(pt(0, tf(f)), pt(1, tf(f)), color=WHITE, stroke_width=2) for f in fs])
        corner = [RC[0] - RW / 2, RC[1] - RH / 2, 0.0]
        gbox = VGroup(*[Line([corner[0] + v * RW, corner[1], 0], [corner[0] + v * RW, corner[1] + RH, 0], color=WHITE, stroke_width=2) for v in vs],
                      *[Line([corner[0], corner[1] + f * RH, 0], [corner[0] + RW, corner[1] + f * RH, 0], color=WHITE, stroke_width=2) for f in fs])
        lo = Rectangle(width=RW, height=RH / 2, stroke_width=0, fill_color=GOLD_, fill_opacity=0.45).move_to([RC[0], RC[1] - RH / 4, 0])
        hi = Rectangle(width=RW, height=RH / 2, stroke_width=0, fill_color=BLUE_, fill_opacity=0.45).move_to([RC[0], RC[1] + RH / 4, 0])
        self.play(Create(gmap), run_time=1.0)
        self.say(r"Take one echo: a crest and a trough. On the map it is a slanted, shrinking patch.", hold=1.6)
        self.say(r"Redraw that patch with $x$ across and the time within the echo upward. The same patch becomes a plain rectangle.", hold=0.4)
        self.play(ReplacementTransform(gmap.copy(), gbox), ReplacementTransform(bands[2].copy(), lo), ReplacementTransform(bands[3].copy(), hi), run_time=2.4)
        xa = VGroup(Tex(r"$x = 0$", font_size=24, color=BLUE_).next_to([corner[0], corner[1], 0], DOWN, buff=0.1),
                    Tex(r"$x = 1$", font_size=24, color=ORANGE_).next_to([corner[0] + RW, corner[1], 0], DOWN, buff=0.1),
                    Tex(r"position $x \rightarrow$", font_size=22, color=GREY_).next_to([RC[0], corner[1], 0], DOWN, buff=0.1))
        ya = Tex(r"time within\\one echo $\uparrow$", font_size=22, color=GREY_).next_to([corner[0], RC[1], 0], LEFT, buff=0.15)
        self.play(FadeIn(xa), FadeIn(ya), run_time=0.6)
        self.wait(2.0)
        # 3. an image
        cells = strip_cells(RC, RW, RH, co)
        frame = Rectangle(width=RW, height=RH, color=WHITE, stroke_width=2).move_to(RC)
        key = Tex(r"gold: the field is positive \quad blue: negative", font_size=20, color=GREY_).next_to(frame, UP, buff=0.12)
        self.play(FadeOut(lo), FadeOut(hi), FadeOut(gbox), FadeIn(cells), FadeIn(frame), FadeIn(key), run_time=1.2)
        self.say(r"Think of the rectangle as an image. Each pixel is one place and one moment within the echo, and its colour is the strength of the wave there.")
        # 4. it is still a wave: read up one line
        xn = corner[0] + 0.5 * RW
        wig = VMobject(color=WHITE, stroke_width=4).set_points_as_corners(
            [[xn + 0.42 * wave_of(co, 2 * math.pi * k / 120) / top, corner[1] + RH * k / 120, 0] for k in range(121)])
        guide = DashedLine([xn, corner[1], 0], [xn, corner[1] + RH, 0], color=WHITE, stroke_width=1.5)
        self.play(Create(guide), run_time=0.5)
        self.play(Create(wig), run_time=1.6)
        self.say(r"It is still a wave. Read up any one line: the field rises and falls, one swing per echo.")
        self.play(FadeOut(themap), FadeOut(bands), FadeOut(beside), FadeOut(gmap), run_time=0.6)
        wax = Axes(x_range=[0, 2 * math.pi, math.pi], y_range=[-1.25 * top, 1.25 * top, top], x_length=4.6, y_length=1.5, tips=False,
                   axis_config={"color": GREY_, "stroke_width": 1.5, "include_ticks": False}).move_to([-3.9, 1.75, 0])
        wave = wax.plot(lambda t: wave_of(co, t), x_range=[0, 2 * math.pi, 0.02], color=WHITE, stroke_width=3.5)
        wl = Tex(r"that line, turned on its side: time within one echo $\rightarrow$", font_size=20, color=GREY_).next_to(wax, UP, buff=0.06)
        self.play(Create(wax), TransformFromCopy(wig, wave), FadeIn(wl), run_time=2.0)
        br = Brace(wax, DOWN, color=GOLD_)
        bl = Tex(r"one echo: in logarithmic time, the echo period $\Delta$", font_size=20, color=GOLD_).next_to(br, DOWN, buff=0.05)
        self.play(GrowFromCenter(br), FadeIn(bl), run_time=0.7)
        self.say(r"Turned on its side, it is an ordinary wave over one echo. So the height of the rectangle is one echo: the period $\Delta$ we are after.")
        # 5. four fields, four images
        layers = VGroup(*[frame.copy().set_stroke(width=2, opacity=0.6).shift(0.24 * j * UR) for j in (3, 2, 1)])
        ltags = VGroup(*[MathTex(n_, font_size=30, color=c_).move_to(frame.get_corner(UR) + 0.24 * j * UR + 0.22 * RIGHT + 0.12 * DOWN)
                         for j, n_, c_ in zip((0, 1, 2, 3), "pmac", (BLUE_, BLUE_, PURPLE_, PURPLE_))])
        fields = VGroup(Tex(r"$p,\ m$: the scalar wave, heading out and in", font_size=22, color=BLUE_),
                        Tex(r"$a,\ c$: the curved spacetime it travels through", font_size=22, color=PURPLE_))
        fields.arrange(DOWN, buff=0.12, aligned_edge=LEFT).next_to(frame, DOWN, buff=0.65)
        self.play(FadeIn(layers), FadeIn(ltags), FadeIn(fields), FadeOut(key), run_time=0.8)
        self.say(r"One image is not enough. Four quantities, called fields, are needed: two for the wave, its parts heading out and in, and two for the spacetime it bends. So a solution is four such images, like layers.")
        # 6. the computer keeps five columns of each
        ncol, nrow = 28, 16
        node_i = [int(v * ncol) for v in CUTS[1:6]]
        rest = VGroup(*[cells[i * nrow:(i + 1) * nrow] for i in range(ncol) if i not in node_i])
        nl = Tex("the five nodes", font_size=22).next_to(frame, UP, buff=0.85)
        self.play(rest.animate.set_fill(opacity=0.05), FadeOut(wig), FadeOut(guide), FadeOut(fields), FadeIn(nl), run_time=1.0)
        self.say(r"The computer does not keep whole images. Of each one it keeps five columns: five positions between the centre and the horizon, called the nodes.")
        # 7. a column is a wave, and a wave is 32 numbers
        self.play(FadeOut(br), FadeOut(bl), run_time=0.3)
        minis = VGroup()
        for k in (1, 3, 5):
            size = math.hypot(*co[k])
            m = Axes(x_range=[0, 2 * math.pi, math.pi], y_range=[-1.3, 1.3, 1], x_length=1.45, y_length=0.9, tips=False,
                     axis_config={"color": GREY_, "stroke_width": 1, "include_ticks": False})
            c = m.plot(lambda t, k=k, size=size: wave_of(co, t, ks=(k,)) / size, x_range=[0, 2 * math.pi, 0.02],
                       color=GOLD_, stroke_width=2.5)
            minis.add(VGroup(m, c, Tex(f"harmonic {k}\\\\size {size:.2g}", font_size=20, color=GREY_).next_to(m, DOWN, buff=0.05)))
        minis.arrange(RIGHT, buff=0.25).move_to([-3.9, -0.75, 0])
        plus = Tex(r"$=$ a sum of pure tones, each on its own scale:", font_size=20, color=GREY_).next_to(minis, UP, buff=0.1).align_to(minis, LEFT)
        self.play(FadeIn(plus), LaggedStart(*[FadeIn(m, shift=0.2 * DOWN) for m in minis], lag_ratio=0.3), run_time=1.6)
        self.say(f"Each column is a wave, and a wave is a sum of pure tones, its harmonics. Harmonic $k$ swings $k$ times per echo, and each is about ten times smaller than the last. (This one is the real wave at the centre, at $D = {D38:.0f}$.)")
        self.say(r"So a column is stored as its harmonics: 32 numbers. The smallest of them will be the first wall.")
        self.play(FadeOut(VGroup(wax, wave, wl, minis, plus, cells, frame, layers, ltags, xa, ya, nl)), run_time=0.7)
        # 8. each field is a table: 5 nodes across, 32 numbers down
        def table(col):
            g, cw, ch = VGroup(), 0.42, 0.22
            for i in range(5):
                for j in range(8):
                    g.add(Rectangle(width=cw, height=ch, stroke_color=WHITE, stroke_width=1, fill_color=interpolate_color(ManimColor(col), ManimColor('#0d1117'), 0.45), fill_opacity=1.0).move_to([i * cw, -j * ch, 0]))
            return g
        tabs = VGroup(*[table(c).shift(0.3 * k * UR) for k, c in zip((3, 2, 1, 0), (PURPLE_, PURPLE_, BLUE_, BLUE_))]).move_to([0, 0.95, 0])
        edge = tabs[0].get_right()[0] + 0.2
        tags = VGroup(*[MathTex(n, font_size=30, color=c_).move_to([edge, tabs[i].get_top()[1] - 0.12, 0])
                        for i, n, c_ in zip((3, 2, 1, 0), "pmac", (BLUE_, BLUE_, PURPLE_, PURPLE_))])
        tb = Brace(tabs[3], LEFT, color=GREY_)
        tbl = Tex(r"32 numbers\\(8 shown)", font_size=22, color=GREY_).next_to(tb, LEFT, buff=0.1)
        tt = Tex("5 nodes", font_size=22, color=GREY_).next_to(tabs[3], DOWN, buff=0.12)
        self.play(LaggedStart(*[FadeIn(tb_, shift=0.15 * DL) for tb_ in tabs], lag_ratio=0.3), FadeIn(tags), run_time=1.8)
        self.play(GrowFromCenter(tb), FadeIn(tbl), FadeIn(tt), run_time=0.6)
        self.say(r"So each field is a small table: 5 nodes across, 32 numbers down. Four fields, four tables: an array of $4 \times 5 \times 32$ numbers.")
        # 9. laid end to end, with the edges and the height: z
        W, x, blocks = 11.0, 0.0, VGroup()
        unit = W / 769.0
        parts = [("centre", 64, BLUE_)] + [("node", 128, GREY_)] * 5 + [("horizon", 64, ORANGE_), ("Delta", 6, GOLD_)]
        for name, n, col in parts:
            r = Rectangle(width=n * unit, height=0.7, stroke_color=WHITE, stroke_width=1.5, fill_color=col,
                          fill_opacity=0.55 if name != "node" else 0.18)
            r.move_to([x + n * unit / 2 - W / 2, 0.9, 0])
            if name == "node":
                for j, c in enumerate([BLUE_, BLUE_, PURPLE_, PURPLE_]):
                    sub = Rectangle(width=n * unit / 4, height=0.7, stroke_width=0.8, stroke_color=WHITE, fill_color=c,
                                    fill_opacity=0.35)
                    sub.move_to(r.get_left() + RIGHT * (j + 0.5) * n * unit / 4)
                    sub.add(MathTex("pmac"[j], font_size=24).move_to(sub))
                    r.add(sub)
            blocks.add(r)
            x += n * unit
        zl = MathTex(r"z", font_size=54).next_to(blocks, LEFT, buff=0.3)
        mid = Tex(r"five nodes $\times$ four fields $\times$ 32 numbers $= 640$", font_size=24).next_to(blocks[3], DOWN, buff=0.15)
        lc = Tex(r"centre data: 64 numbers\\that start the picture\\at $x = 0$", font_size=22, color=BLUE_).next_to(blocks[0], UP, buff=0.2).align_to(blocks[0], LEFT)
        lh = Tex(r"horizon data: 64 numbers\\that start it at $x = 1$", font_size=22, color=ORANGE_).next_to(blocks[6], UP, buff=0.2).align_to(blocks[6], RIGHT)
        ld = Tex(r"the height $\Delta$: 1", font_size=24, color=GOLD_).next_to(blocks[7], DOWN, buff=0.15).align_to(blocks[7], RIGHT)
        self.play(FadeOut(VGroup(tags, tb, tbl, tt)), ReplacementTransform(tabs, VGroup(*blocks[1:6])), run_time=1.6)
        self.play(FadeIn(mid), run_time=0.4)
        self.say(r"Lay the tables end to end, node by node.")
        self.play(FadeIn(blocks[0]), FadeIn(blocks[6]), FadeIn(blocks[7]), FadeIn(lc), FadeIn(lh), FadeIn(ld), run_time=0.9)
        self.say(r"Add the numbers that start the picture at its two edges, the centre data and the horizon data, and its height $\Delta$.")
        tot = Tex(r"$640 + 64 + 64 + 1 = 769$ numbers", font_size=38, color=GOLD_).move_to([0, -1.2, 0])
        self.play(FadeIn(zl), FadeIn(tot, shift=0.2 * UP), run_time=0.7)
        self.say(r"That row of 769 numbers is called $z$. It is everything the computer knows about one solution.")
        self.say(r"From here on, a guess at the solution is a guess at $z$. Solving means adjusting those numbers until the four images obey Einstein's equations.")
        self.play(FadeOut(VGroup(blocks, zl, mid, lc, lh, ld, tot, self.cap)), run_time=0.7)


class S03d_Join(Film):
    stage = r"6 \quad Checking a guess: do the pieces join?"

    def construct(self):
        D38, co = harmonics(38.0)
        # a guess z holds each image only on five columns and at its two edges
        RW, RH, RC, ncol, nrow = 10.0, 2.2, [0.0, 0.55, 0.0], 50, 12
        line, deco = x_line(y=RC[1] - RH / 2, height=RH, nodes="the five nodes: what $z$ stores")
        frame, cols, sides, ends, nl, up = deco
        cells = strip_cells(RC, RW, RH, co, ncol=ncol, nrow=nrow)
        col = lambda i: cells[i * nrow:(i + 1) * nrow]
        node_i = [round(v * ncol) for v in CUTS[1:6]]
        cells.set_fill(opacity=0.0)
        known = VGroup(*[col(i) for i in node_i])
        tup = Tex(r"time within one echo $\uparrow$", font_size=20, color=GREY_).move_to(up).align_to(frame, LEFT)
        self.add(cells)
        self.play(FadeIn(VGroup(line, frame, sides, ends, nl, tup)), run_time=0.8)
        for i in node_i:
            for j, c in enumerate(col(i)):
                c.set_fill(opacity=min(1.0, 0.12 + 0.95 * abs(((i + 0.5) / ncol) * wave_of(co, 2 * math.pi * (j + 0.5) / nrow) / max(abs(wave_of(co, 2 * math.pi * k / 200)) for k in range(200)))))
        self.play(FadeIn(known), run_time=0.8)
        self.say(r"How can we tell whether a guess $z$ obeys them? It holds each image only on five columns, and at its two edges. Everything depends on what lies in between.")
        eq = MathTex(r"x\,(x - c)\,\frac{\partial p}{\partial x} \;=\; \cdots", font_size=40).move_to([0, 2.75, 0])
        self.play(Write(eq), run_time=1.2)
        self.say(r"Einstein's equations, with the wave equation, say how every field changes from one $x$ to the next.")
        top = max(abs(wave_of(co, 2 * math.pi * k / 200)) for k in range(200))
        def lit(i):
            g = col(i)
            for j, c in enumerate(g):
                c.generate_target()
                c.target.set_fill(opacity=min(1.0, 0.12 + 0.95 * abs(((i + 0.5) / ncol) * wave_of(co, 2 * math.pi * (j + 0.5) / nrow) / top)))
            return [MoveToTarget(c) for c in g]
        def arrow(i0, i1, colr, y=None):
            xa_ = RC[0] - RW / 2 + (i0 + 0.5) * RW / ncol
            xb_ = RC[0] - RW / 2 + (i1 + 0.5) * RW / ncol
            yy = RC[1] + RH / 2 + 0.45 if y is None else y
            return Arrow([xa_, yy, 0], [xb_, yy, 0], color=colr, stroke_width=4, buff=0, max_tip_length_to_length_ratio=0.25)
        a1 = arrow(node_i[0], node_i[1], GREEN_)
        self.say(r"So starting from one column, the equations fill in the image as far as the next column. That filled-in stretch is a piece.", hold=0.4)
        self.play(GrowArrow(a1), LaggedStart(*[AnimationGroup(*lit(i)) for i in range(node_i[0] + 1, node_i[1])], lag_ratio=0.5), run_time=2.2)
        self.wait(2.6)
        arrows = VGroup(a1, *[arrow(node_i[k], node_i[k + 1], GREEN_) for k in (1, 2, 3)], arrow(0, node_i[0], BLUE_), arrow(ncol - 1, node_i[4], ORANGE_))
        self.say(r"Six pieces cover the whole image: one out from the centre, four from node to node, and one in from the horizon.", hold=0.4)
        self.play(GrowArrow(arrows[4]), LaggedStart(*[AnimationGroup(*lit(i)) for i in range(0, node_i[0])], lag_ratio=0.3), run_time=1.6)
        for k in (1, 2, 3):
            self.play(GrowArrow(arrows[k]), LaggedStart(*[AnimationGroup(*lit(i)) for i in range(node_i[k] + 1, node_i[k + 1])], lag_ratio=0.5), run_time=0.9)
        self.play(GrowArrow(arrows[5]), LaggedStart(*[AnimationGroup(*lit(i)) for i in range(ncol - 1, node_i[4], -1)], lag_ratio=0.3), run_time=1.6)
        self.wait(1.8)
        row = Line([RC[0] - RW / 2, RC[1] + 0.25 * RH, 0], [RC[0] + RW / 2, RC[1] + 0.25 * RH, 0], color=WHITE, stroke_width=3)
        self.say(r"Do the pieces fit together? To see, look along one row of the image: one field at one moment, drawn against $x$.", hold=0.4)
        self.play(Create(row), run_time=1.0)
        self.wait(1.6)
        # one row, in the same frame: the six pieces, the stored values, the gaps
        P = [piece(line, i, 1.0) for i in range(6)]
        dots = VGroup(*[Dot(P[i].get_start(), color=WHITE, radius=0.07) for i in (1, 2, 3, 4)], Dot(P[5].get_end(), color=WHITE, radius=0.07))
        self.play(FadeOut(cells), FadeOut(arrows), FadeOut(tup), FadeOut(row), FadeIn(cols), FadeIn(up), run_time=1.4)
        self.say(r"Keep the same frame: the centre on the left, the horizon on the right, and the five node columns as upright lines. Only the upward direction changes. It now shows the value of the field along that one row.")
        self.play(FadeIn(dots), run_time=0.5)
        self.say(r"The dots are the values $z$ stores at the five nodes.", hold=1.4)
        tl = Tex("Taylor steps", font_size=24, color=GREEN_).next_to(line.n2p(0.5), UP, buff=0.18)
        self.say(r"From node to node, a piece is carried by a Taylor step: a polynomial of high order that follows the equations across the gap.", hold=0.4)
        self.play(LaggedStart(*[Create(P[i]) for i in (1, 2, 3, 4)], lag_ratio=0.6), FadeIn(tl), run_time=2.6)
        self.wait(2.4)
        self.say(r"At the two ends that fails: the factor in front becomes zero, and the equation would divide by it.", hold=0.4)
        self.play(eq[0][0].animate.set_color(BLUE_), eq[0][1:6].animate.set_color(ORANGE_), run_time=0.6)
        self.play(Indicate(eq[0][0], color=BLUE_), Indicate(eq[0][1:6], color=ORANGE_), run_time=1.2)
        self.wait(2.2)
        self.say(r"$x = 0$ is the centre. And $c$ is the speed of light on our map, so $x - c = 0$ is the horizon: light there runs along the edge, not across it.")
        cl = Tex("series from the centre", font_size=24, color=BLUE_).next_to(line.n2p(0.15), UP, buff=0.18)
        hl = Tex("series from the horizon", font_size=24, color=ORANGE_).next_to(line.n2p(0.85), UP, buff=0.18)
        self.say(r"So at each end the piece is a power series, a long polynomial that is smooth there, started by that end's own numbers in $z$: the centre data and the horizon data.", hold=0.4)
        self.play(Create(P[0]), FadeIn(cl), run_time=1.3)
        self.play(Create(P[5]), FadeIn(hl), run_time=1.3)
        self.wait(2.6)
        g = gaps(line, 1.0)
        self.say(r"Each piece should land exactly on the stored values where it arrives. For a guessed $z$ it misses, and leaves a gap.", hold=0.4)
        self.play(LaggedStart(*[GrowFromCenter(sg) for sg in g], lag_ratio=0.2), run_time=1.4)
        self.wait(1.8)
        F = MathTex(r"F(z)", font_size=44, color=RED_)
        Ft = Tex("= all the gaps, one for each number in $z$", font_size=28, color=RED_).next_to(F, RIGHT, buff=0.25)
        VGroup(F, Ft).move_to([0, 2.75, 0])
        self.play(FadeOut(eq), run_time=0.4)
        self.play(FadeIn(F), FadeIn(Ft), run_time=0.6)
        self.say(r"Those gaps are called the residual, $F(z)$. Only the true fields, with the true $\Delta$, close every one of them.")
        root = MathTex(r"F(z) = 0", font_size=56, color=GOLD_).move_to([0, 2.75, 0])
        self.play(FadeOut(F), FadeOut(Ft), FadeIn(root, scale=1.2), run_time=0.7)
        self.say(r"That is what a solve is: find the $z$ with $F(z) = 0$. A solution is a root.")
        self.play(FadeOut(VGroup(line, deco, dots, tl, cl, hl, g, root, self.cap, *P)), run_time=0.7)


class S03e_Newton(Film):
    stage = r"7 \quad Newton's method"

    def construct(self):
        # one unknown: follow the slope down to zero
        ax = Axes(x_range=[1.0, 3.25, 0.5], y_range=[-2, 8, 2], x_length=6.4, y_length=4.3, tips=False,
                  axis_config={"color": GREY_, "stroke_width": 2, "include_numbers": True, "font_size": 20},
                  x_axis_config={"decimal_number_config": {"num_decimal_places": 1}}).move_to([-3.2, 0.6, 0])
        f = lambda v: v * v - 2.0
        curve = ax.plot(f, x_range=[1.0, 3.2, 0.01], color=BLUE_, stroke_width=4)
        yl = Tex("the miss", font_size=22, color=GREY_).next_to(ax.y_axis, UP, buff=0.08)
        xl = Tex("the unknown", font_size=22, color=GREY_).next_to(ax.x_axis, RIGHT, buff=0.1)
        rt = Dot(ax.c2p(math.sqrt(2.0), 0), color=GOLD_, radius=0.08)
        rl = Tex("the root", font_size=22, color=GOLD_).next_to(rt, UL, buff=0.12)
        self.play(Create(ax), FadeIn(yl), FadeIn(xl), Create(curve), run_time=1.4)
        self.play(FadeIn(rt, scale=2), FadeIn(rl), run_time=0.5)
        self.say(r"How do you find a root? Newton's method. First with a single unknown: we want the point where this curve crosses zero, where the miss is nothing.")
        xs = [3.0]
        for _ in range(4):
            xs.append(xs[-1] - f(xs[-1]) / (2.0 * xs[-1]))
        head = VGroup(Tex("guess", font_size=26, color=GREY_), Tex("miss", font_size=26, color=GREY_)).arrange(RIGHT, buff=1.5).move_to([3.6, 2.3, 0])
        rows = VGroup()
        def add_row(k):
            v = f(xs[k])
            miss = f"{v:.2f}" if v >= 1 else f"{v:.4f}" if v >= 1e-3 else f"{v:.7f}"
            r = VGroup(Tex(f"{xs[k]:.6f}", font_size=28), Tex(miss, font_size=28, color=RED_ if k < 3 else GREEN_))
            r[0].move_to([head[0].get_center()[0], 1.75 - 0.5 * k, 0])
            r[1].move_to([head[1].get_center()[0], 1.75 - 0.5 * k, 0])
            rows.add(r)
            return r
        d0 = Dot(ax.c2p(xs[0], f(xs[0])), color=WHITE, radius=0.08)
        self.play(FadeIn(head), FadeIn(d0, scale=2), FadeIn(add_row(0)), run_time=0.7)
        self.say(r"Start from a guess. The curve's slope there says how fast the miss changes. Follow that slope straight down to zero: that is the next guess.", hold=0.3)
        drawn = VGroup(d0)
        for k in range(4):
            tan = Line(ax.c2p(xs[k], f(xs[k])), ax.c2p(xs[k + 1], 0), color=GOLD_, stroke_width=3)
            up = DashedLine(ax.c2p(xs[k + 1], 0), ax.c2p(xs[k + 1], f(xs[k + 1])), color=GREY_, stroke_width=2)
            dk = Dot(ax.c2p(xs[k + 1], f(xs[k + 1])), color=WHITE, radius=0.08)
            self.play(Create(tan), run_time=1.2 if k == 0 else 0.7)
            self.play(Create(up), FadeIn(dk, scale=2), FadeIn(add_row(k + 1)), run_time=0.7 if k == 0 else 0.5)
            drawn.add(tan, up, dk)
            if k == 0:
                self.wait(2.6)
                self.say(r"Then do it again from there.", hold=0.3)
        self.say(r"Each round roughly squares the miss: 7, then 1.4, then 0.14, 0.002, 0.0000006. A few rounds give every digit the arithmetic can hold.")
        self.play(FadeOut(VGroup(ax, yl, xl, curve, rt, rl, drawn, head, rows)), run_time=0.6)
        # the catch: the slope only knows the curve nearby
        bx = Axes(x_range=[-2.6, 2.6, 1], y_range=[-1.6, 1.6, 1], x_length=7.4, y_length=3.6, tips=False,
                  axis_config={"color": GREY_, "stroke_width": 2, "include_numbers": True, "font_size": 20}).move_to([0, 0.75, 0])
        bcurve = bx.plot(math.atan, x_range=[-2.6, 2.6, 0.01], color=BLUE_, stroke_width=4)
        brt = Dot(bx.c2p(0, 0), color=GOLD_, radius=0.08)
        self.play(Create(bx), Create(bcurve), FadeIn(brt), run_time=1.2)
        self.say(r"But the slope only knows the curve close by. On a curve that flattens out, start too far from the root and the step overshoots.", hold=0.3)
        ys = [1.5]
        for _ in range(2):
            ys.append(ys[-1] - math.atan(ys[-1]) * (1.0 + ys[-1] ** 2))
        bd = VGroup(Dot(bx.c2p(ys[0], math.atan(ys[0])), color=WHITE, radius=0.08))
        self.play(FadeIn(bd[0], scale=2), run_time=0.4)
        for k in range(2):
            tan = Line(bx.c2p(ys[k], math.atan(ys[k])), bx.c2p(ys[k + 1], 0), color=RED_, stroke_width=3)
            up = DashedLine(bx.c2p(ys[k + 1], 0), bx.c2p(ys[k + 1], math.atan(ys[k + 1])), color=GREY_, stroke_width=2)
            dk = Dot(bx.c2p(ys[k + 1], math.atan(ys[k + 1])), color=WHITE, radius=0.08)
            self.play(Create(tan), run_time=1.1)
            self.play(Create(up), FadeIn(dk, scale=2), run_time=0.6)
            bd.add(tan, up, dk)
        worse = Tex(f"the miss grows: {abs(math.atan(ys[0])):.2f}, {abs(math.atan(ys[1])):.2f}, {abs(math.atan(ys[2])):.2f}", font_size=28, color=RED_).next_to(bx, UP, buff=0.1)
        self.play(FadeIn(worse), run_time=0.4)
        self.say(r"The miss gets bigger, not smaller. Newton's method needs a first guess that is already close. Remember that: it becomes a wall.")
        self.play(FadeOut(VGroup(bx, bcurve, brt, bd, worse)), run_time=0.6)
        # our problem: 769 unknowns, 769 misses, and a table of slopes
        line, deco = x_line()
        e = ValueTracker(1.0)
        shot = always_redraw(lambda: VGroup(*[piece(line, i, e.get_value()) for i in range(6)]))
        gp = always_redraw(lambda: gaps(line, e.get_value()))
        self.play(FadeIn(VGroup(line, deco)), FadeIn(shot), FadeIn(gp), run_time=0.8)
        self.say(r"Our problem is the same, with 769 unknowns, the numbers in $z$, and 769 misses, the gaps of $F(z)$. The single slope becomes a table of slopes.")
        zrow = VGroup(*[Square(side_length=0.26, stroke_color=WHITE, stroke_width=1, fill_color=GREY_, fill_opacity=0.25) for _ in range(16)]).arrange(RIGHT, buff=0.0).move_to([-3.6, 2.75, 0])
        zl = MathTex("z", font_size=36).next_to(zrow, LEFT, buff=0.15)
        J = VGroup(*[Square(side_length=0.2, stroke_color=GREY_, stroke_width=0.8, fill_color=GOLD_, fill_opacity=0.0) for _ in range(100)]).arrange_in_grid(rows=10, cols=10, buff=0.0).move_to([3.9, 2.0, 0])
        Jl = MathTex("J", font_size=40, color=GOLD_).next_to(J, LEFT, buff=0.2)
        self.play(FadeIn(zrow), FadeIn(zl), FadeIn(J), FadeIn(Jl), run_time=0.7)
        self.say(r"Nudge one number of $z$, and watch how every gap moves. That fills one column of the table. It is called the Jacobian, $J$.", hold=0.3)
        import random
        rnd = random.Random(7)
        for n, kcol in enumerate((2, 6, 11)):
            jc = VGroup(*[J[10 * r + (kcol * 10 // 16)] for r in range(10)])
            self.play(zrow[kcol].animate.set_fill(GOLD_, opacity=0.9), e.animate.set_value(1.0 + 0.12 * (-1) ** n), run_time=0.5)
            self.play(*[c.animate.set_fill(opacity=rnd.uniform(0.2, 0.95)) for c in jc], e.animate.set_value(1.0), zrow[kcol].animate.set_fill(GREY_, opacity=0.25), run_time=0.7)
        self.play(*[c.animate.set_fill(opacity=rnd.uniform(0.15, 0.95)) for c in J], run_time=1.0)
        self.wait(1.2)
        eq = MathTex(r"J\,dz = -F(z)", font_size=46, color=GOLD_).move_to([-0.2, 2.2, 0])
        res = Tex(r"largest gap: $10^{-2}$", font_size=30, color=RED_).move_to([-0.2, 1.45, 0])
        self.play(Write(eq), FadeIn(res), run_time=1.0)
        self.say(r"With all 769 columns, one linear solve gives the change $dz$ in every number at once that should close every gap. Apply it, and repeat.", hold=0.4)
        for scale, expo in [(0.35, 4), (0.06, 8), (0.0, 16), (0.0, 32)]:
            new = Tex(r"largest gap: $10^{-" + str(expo) + r"}$", font_size=30, color=RED_ if expo < 16 else GREEN_).move_to(res)
            self.play(e.animate.set_value(scale), Transform(res, new), run_time=1.0)
            self.wait(0.5)
        self.say(r"As with one unknown, each round roughly squares the misses, as long as the first guess for $z$ was close.")
        shot.clear_updaters()
        gp.clear_updaters()
        self.play(FadeOut(VGroup(line, deco, shot, gp, eq, res, zrow, zl, J, Jl)), run_time=0.5)
        pts = delta_points()
        ax2, lab = low_axes()
        at = lambda D: min(pts, key=lambda q: abs(q[0] - D))
        curve2 = VMobject(color=BLUE_, stroke_width=3, stroke_opacity=0.35).set_points_as_corners(
            [ax2.c2p(*q) for q in pts if 3.9 <= q[0] <= 13.0])
        dots = VGroup(*[Dot(ax2.c2p(*at(D)), color=BLUE_, radius=0.07) for D in (4, 5)])
        newd = Dot(ax2.c2p(*at(6.0)), color=GOLD_, radius=0.11)
        self.play(FadeIn(VGroup(ax2, lab, curve2, dots)), run_time=0.6)
        self.play(FadeIn(newd, scale=3), Flash(newd, color=GOLD_), run_time=0.8)
        self.say(r"The gaps are closed. That is one dot on the curve: its $\Delta$ is the last number in $z$. Now for the next $D$.")
        self.play(FadeOut(VGroup(ax2, lab, curve2, dots, newd, self.cap)), run_time=0.7)


class S04_Walk(Film):
    stage = r"8 \quad Walking in $D$"

    def construct(self):
        pts = [p for p in delta_points() if 3.9 <= p[0] <= 60.97]
        ax = Axes(x_range=[0, 65, 10], y_range=[1.6, 3.6, 0.4], x_length=10.6, y_length=4.4, tips=False,
                  axis_config={"color": GREY_, "stroke_width": 2, "include_numbers": True, "font_size": 22},
                  y_axis_config={"decimal_number_config": {"num_decimal_places": 1}}).shift(0.4 * UP)
        xl = Tex("$D$", font_size=28, color=GREY_).next_to(ax.x_axis, RIGHT, buff=0.15)
        yl = MathTex(r"\Delta", font_size=36, color=GREY_).next_to(ax.y_axis, UP, buff=0.1)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), run_time=0.8)
        at = lambda D: min(pts, key=lambda p: abs(p[0] - D))
        stops = [at(D) for D in (4, 7, 10, 13)]
        dot = Dot(ax.c2p(*stops[0]), color=GOLD_, radius=0.08)
        trail = VGroup(Dot(ax.c2p(*stops[0]), color=BLUE_, radius=0.05))
        self.play(FadeIn(dot, scale=2), run_time=0.5)
        self.say(r"Newton needs a first guess that is already close. The best one at hand is the solution just found, at a nearby $D$.", hold=0.3)
        for i, (a, b) in enumerate(zip(stops, stops[1:])):
            guess = ax.c2p(b[0], a[1] + (b[1] - a[1]) * 0.55)
            arr = Arrow(ax.c2p(*a), guess, color=GREY_, buff=0.05, stroke_width=3, max_tip_length_to_length_ratio=0.12)
            ghost = Dot(guess, color=GREY_, radius=0.06)
            lab = Tex("predict" if i == 0 else "", font_size=22, color=GREY_).next_to(ghost, UP, buff=0.1)
            self.play(GrowArrow(arr), FadeIn(ghost), FadeIn(lab), run_time=0.55)
            lab2 = Tex("correct" if i == 0 else "", font_size=22, color=GOLD_).next_to(ax.c2p(*b), DOWN, buff=0.15)
            self.play(dot.animate.move_to(ax.c2p(*b)), FadeOut(ghost), FadeIn(lab2), run_time=0.55)
            trail.add(Dot(ax.c2p(*b), color=BLUE_, radius=0.05))
            self.add(trail)
            self.play(FadeOut(arr), FadeOut(lab), FadeOut(lab2), run_time=0.3)
        self.say(r"Predict the solution at the next $D$, let Newton correct it, and repeat. This is continuation.", hold=0.4)
        rest = [p for p in pts if p[0] >= stops[-1][0]]
        path = VMobject(color=BLUE_, stroke_width=4).set_points_as_corners([ax.c2p(*p) for p in rest])
        full = VMobject(color=BLUE_, stroke_width=4).set_points_as_corners([ax.c2p(*p) for p in pts if p[0] <= stops[-1][0]])
        Dn = DecimalNumber(stops[-1][0], num_decimal_places=1, font_size=40, color=GOLD_)
        Dl = Tex("$D =$", font_size=40, color=GOLD_)
        VGroup(Dl, Dn).arrange(RIGHT, buff=0.2).move_to(ax.c2p(40, 3.2))
        Dn.add_updater(lambda m: m.set_value(ax.p2c(dot.get_center())[0]))
        self.play(FadeIn(full), FadeOut(trail), FadeIn(Dl), FadeIn(Dn), run_time=0.5)
        self.play(Create(path), MoveAlongPath(dot, path), run_time=5.0, rate_func=linear)
        Dn.clear_updaters()
        wall = DashedLine(ax.c2p(60.96, 1.6), ax.c2p(60.96, 3.6), color=RED_, stroke_width=4)
        self.play(Create(wall), Flash(dot, color=RED_), run_time=0.7)
        self.say(r"With numbers of about 32 digits, the walk climbed for a day. At $D = 60.96$ it stopped: the first wall.")
        self.play(FadeOut(VGroup(ax, xl, yl, dot, path, full, Dl, Dn, wall, self.cap)), run_time=0.7)


class S05_FirstWall(Film):
    stage = r"9 \quad The first wall: weak harmonics"

    def construct(self):
        pts = frontier_points()
        ax, lab = self.frontier_axes()
        self.play(Create(ax), FadeIn(lab), run_time=0.8)
        self.say(r"From here on, one chart keeps the score: the highest $D$ the solver has reached, against the days of work.", hold=0.4)
        c1 = self.front(ax, pts, 0.0, 4.0)
        self.play(Create(c1), run_time=1.6)
        w = Tex("60.96", font_size=26, color=RED_).next_to(ax.c2p(3.0, 61), UP, buff=0.1)
        self.play(FadeIn(w), run_time=0.4)
        self.wait(1.6)
        self.say(r"Why did it stop at 60.96? The answer is in the harmonics: the pure tones that add up to the wave over one echo.")
        chart = VGroup(ax, lab, c1, w)
        self.play(FadeOut(chart), run_time=0.5)
        Dw, co = harmonics(60.96)
        size = {k: math.hypot(*co[k]) for k in co}
        top = 1.25 * max(abs(wave_of(co, 2 * math.pi * k / 200)) for k in range(200))
        mk = lambda y, h, yr: Axes(x_range=[0, 2 * math.pi, math.pi], y_range=[-yr, yr, yr], x_length=5.6, y_length=h, tips=False,
                                   axis_config={"color": GREY_, "stroke_width": 1.5, "include_ticks": False}).move_to([-3.7, y, 0])
        wax = mk(2.1, 1.45, top)
        wl = Tex(f"the scalar wave at the centre over one echo, at $D = {Dw:.2f}$", font_size=22, color=GREY_).next_to(wax, UP, buff=0.06)
        wave = wax.plot(lambda t: wave_of(co, t), x_range=[0, 2 * math.pi, 0.01], color=GOLD_, stroke_width=4)
        self.play(Create(wax), FadeIn(wl), Create(wave), run_time=1.4)
        self.say(r"This is the real wave at the wall. It is the sum of the harmonics below, each drawn on a row of its own.", hold=0.4)
        rows = VGroup()
        for j, (k, col) in enumerate(zip((1, 3, 5), (BLUE_, GREEN_, PURPLE_))):
            ak = mk(0.42 - 1.2 * j, 0.72, 1.3)
            ck = ak.plot(lambda t, k=k: wave_of(co, t, ks=(k,)) / size[k], x_range=[0, 2 * math.pi, 0.01], color=col, stroke_width=3)
            text = "harmonic 1: one swing per echo" if k == 1 else f"harmonic {k}: {k} swings, drawn {size[1] / size[k]:.0f} times larger"
            lk = Tex(text, font_size=22, color=col).next_to(ak, UP, buff=0.13).align_to(ak, LEFT)
            rows.add(VGroup(ak, ck, lk))
            self.play(Create(ak), Create(ck), FadeIn(lk), run_time=1.1)
            self.wait(0.9)
        self.say(r"Harmonic $k$ swings $k$ times in one echo. Only odd ones appear, because this wave flips sign every half echo.")
        sizes = VGroup()
        for k in sorted(co):
            bar = Rectangle(width=0.42 * (7.0 + math.log10(size[k])), height=0.3, stroke_width=0, fill_color=GOLD_, fill_opacity=0.8)
            sizes.add(VGroup(Tex(f"harmonic {k}", font_size=24), bar, MathTex(sci(size[k]), font_size=26, color=GOLD_)).arrange(RIGHT, buff=0.2))
        sizes.arrange(DOWN, buff=0.16, aligned_edge=LEFT)
        for row in sizes:
            row[1].align_to(sizes[-1][1], LEFT)
            row[2].next_to(row[1], RIGHT, buff=0.15)
        sh = Tex("the size of each harmonic", font_size=24, color=GREY_)
        sbox = VGroup(sh, sizes).arrange(DOWN, buff=0.25).move_to([3.5, 0.45, 0])
        self.play(FadeIn(sh), LaggedStart(*[FadeIn(r, shift=0.2 * RIGHT) for r in sizes], lag_ratio=0.2), run_time=2.2)
        self.say(r"Each harmonic is about ten times smaller than the one before. Harmonic 13 is two millionths of harmonic 1.")
        self.say(r"The solver must still get every one of them right. And the small ones are the hard ones.")
        self.play(FadeOut(VGroup(wax, wl, wave, rows)), sbox.animate.move_to([-3.6, 0.45, 0]), run_time=0.8)
        D0, sig = grip(79.77)
        hs = sorted(sig)
        panel = VGroup()
        for i, h in enumerate(hs):
            dec = 30.0 + math.log10(sig[h])
            bar = Rectangle(width=dec * 0.2, height=0.3, stroke_width=0, fill_color=RED_, fill_opacity=0.85 - 0.14 * i)
            name = Tex(f"harmonic {h}", font_size=24)
            val = MathTex(r"10^{" + f"{math.log10(sig[h]):.0f}" + "}", font_size=28, color=RED_)
            panel.add(VGroup(name, bar, val).arrange(RIGHT, buff=0.2))
        panel.arrange(DOWN, buff=0.16, aligned_edge=LEFT)
        for row in panel:
            row[1].align_to(panel[0][1], LEFT)
            row[2].next_to(row[1], RIGHT, buff=0.15)
        head = Tex(f"how firmly the equations hold it, at $D = {D0:.1f}$", font_size=24, color=GREY_)
        box = VGroup(head, panel).arrange(DOWN, buff=0.25).move_to([3.5, 0.2, 0])
        self.play(FadeIn(head), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(r, shift=0.2 * RIGHT) for r in panel], lag_ratio=0.25), run_time=2.0)
        self.say(r"The equations barely pin the higher harmonics down. Their grip falls by decades per harmonic, and with $D$.")
        self.say(f"At $D = {D0:.1f}$ a full Newton step would move harmonic 13 by 0.44. Its true size is 0.00000037.")
        self.say(r"Rounding error, multiplied by how sensitive the solve is to it, reached 1 just before the stall. Newton was stepping on noise.")
        self.play(FadeOut(VGroup(sbox, box)), run_time=0.5)
        self.play(FadeIn(chart), run_time=0.5)
        fix = Tex(r"hold the weak harmonics on their smooth trend in $D$, solve for everything else", font_size=28, color=GREEN_)
        fix.move_to(ax.c2p(19, 235))
        self.play(FadeIn(fix, shift=0.2 * UP), run_time=0.6)
        self.say(r"The fix: stop asking the equations for what they cannot give. Hold those harmonics, and solve the rest.", hold=0.5)
        c2 = self.front(ax, pts, 4.0, 8.9)
        self.play(Create(c2), run_time=2.2)
        self.say(r"With those harmonics held, the walk moved again, to 67.8, 71 and 73.3. Then it slowed once more, for a reason that sits at the horizon.")
        self.play(FadeOut(VGroup(ax, lab, c1, c2, w, fix, self.cap)), run_time=0.7)


class S06_Amplifier(Film):
    stage = r"10 \quad The second wall: an amplifier at the horizon"

    def construct(self):
        f = MathTex(r"G \approx 0.7^{-(D-3)}", font_size=60, color=ORANGE_).move_to([-3.3, 2.2, 0])
        self.say(r"The piece that comes in from the horizon starts from the horizon data. Any error in those numbers is multiplied, on the way to the last node, by a factor $G$.", hold=0.4)
        self.play(Write(f), run_time=1.2)
        D = ValueTracker(10.0)
        Dl = Tex("$D =$", font_size=44)
        Dn = Integer(10, font_size=44)
        Dn.add_updater(lambda m: m.set_value(round(D.get_value())))
        Gl = Tex(r"$G \approx 10$", font_size=44, color=ORANGE_)
        Gn = Integer(1, font_size=30, color=ORANGE_)
        Gn.add_updater(lambda m: m.set_value(round(log10G(D.get_value()))))
        VGroup(Dl, Dn).arrange(RIGHT, buff=0.2).move_to([2.4, 2.45, 0])
        Gl.move_to([2.55, 1.75, 0])
        Dn.add_updater(lambda m: m.next_to(Dl, RIGHT, buff=0.2))
        Gn.add_updater(lambda m: m.next_to(Gl, UR, buff=0.02).shift(0.12 * DOWN))
        # digits: a bar of the decades G eats, against what each number type carries
        base, per = [-5.6, -0.45, 0], 11.2 / 130.0
        axis = Line(base, [base[0] + 130 * per, base[1], 0], color=GREY_, stroke_width=2)
        types = [("double", 16, GREY_), ("double-double", 32, BLUE_), ("quad-double", 64, GREEN_), ("octo-double", 127, PURPLE_)]
        marks = VGroup()
        for i, (name, dig, col) in enumerate(types):
            x = base[0] + dig * per
            tick = Line([x, base[1] - 0.12, 0], [x, base[1] + 1.05, 0], color=col, stroke_width=2.5)
            t = Tex(f"{name}\\\\{dig} digits", font_size=22, color=col).next_to(tick, UP, buff=0.08)
            marks.add(VGroup(tick, t))
        al = Tex("digits", font_size=22, color=GREY_).next_to(axis, RIGHT, buff=0.12)
        eaten = always_redraw(lambda: Rectangle(width=max(0.01, log10G(D.get_value()) * per), height=0.42, stroke_width=0,
                                                 fill_color=ORANGE_, fill_opacity=0.8).move_to(
            [base[0] + log10G(D.get_value()) * per / 2, base[1] + 0.3, 0]))
        el = Tex("digits lost to $G$", font_size=22, color=ORANGE_).next_to(axis, UP, buff=0.75).align_to(axis, LEFT)
        self.play(FadeIn(VGroup(Dl, Dn, Gl, Gn)), Create(axis), FadeIn(al), run_time=0.8)
        self.play(LaggedStart(*[FadeIn(m, shift=0.15 * UP) for m in marks], lag_ratio=0.2), FadeIn(eaten), FadeIn(el), run_time=1.6)
        self.say(r"An ordinary computer number, a double, holds 16 digits. For more, a number is stored as a sum of 2, 4 or 8 doubles: double-double, quad-double, octo-double.")
        def flag(digits, text, col, drop, side=LEFT):
            x = base[0] + digits * per
            stem = Line([x, base[1] - 0.05, 0], [x, base[1] - drop, 0], color=col, stroke_width=2)
            return VGroup(stem, Tex(text, font_size=22, color=col).next_to(stem, DOWN, buff=0.06).align_to(stem, side))

        D_at = lambda digits: 3.0 + digits / math.log10(1.0 / 0.7)        # the D at which G has eaten that many digits
        self.say(r"$G$ grows exponentially with $D$, and it eats those digits.", hold=0.3)
        self.play(D.animate.set_value(80), run_time=2.6, rate_func=linear)
        f1 = flag(log10G(80), r"$D \approx 80$: double-double gives out", BLUE_, 1.4)
        self.play(FadeIn(f1, shift=0.1 * DOWN), run_time=0.5)
        self.wait(1.0)
        self.play(D.animate.set_value(131), run_time=1.8, rate_func=linear)
        self.say(r"At $D = 131$ it is already $10^{20}$. Every stage of the solver needs longer numbers as $D$ climbs.", hold=2.2)
        self.play(D.animate.set_value(194), run_time=2.2, rate_func=linear)
        f2 = flag(log10G(194), r"$D \approx 194$: quad-double gives out", GREEN_, 1.0)
        self.play(FadeIn(f2, shift=0.1 * DOWN), run_time=0.5)
        self.wait(1.0)
        self.play(D.animate.set_value(260), run_time=2.2, rate_func=linear)
        self.wait(0.6)
        self.say(r"So the walk needed longer numbers the higher it went: double-double to about 80, quad-double to about 194, octo-double beyond.")
        # how far could octo-double go?
        self.say(r"How far could octo-double go? Let $D$ run on, until $G$ has eaten all of its 127 digits.", hold=0.3)
        self.play(D.animate.set_value(D_at(127)), run_time=7.0, rate_func=linear)
        ft = flag(127, f"$D \\approx {D_at(127):.0f}$: every digit gone", PURPLE_, 1.4, side=RIGHT)
        self.play(FadeIn(ft, shift=0.1 * DOWN), run_time=0.5)
        self.say(f"At $D \\approx {D_at(127):.0f}$, $G$ swallows all 127 digits. That is octo-double's ceiling: past it, not one digit of the horizon data survives.")
        Dn.clear_updaters()
        Gn.clear_updaters()
        f3 = flag(63.5, f"$D \\approx {D_at(63.5):.0f}$: the series in $D$ has none left (an estimate)", PURPLE_, 0.6)
        self.say(r"The working limit is nearer. The solver's series in $D$ is hit twice: its error grows like $\varepsilon G^2$, with $\varepsilon$ the rounding of the arithmetic. So its digits are gone at half the bar.", hold=0.4)
        self.play(FadeIn(f3, shift=0.1 * DOWN), run_time=0.5)
        self.say(f"For double-double that half-way point is $D \\approx {D_at(16):.0f}$, and it gave out near 80. For quad-double it is {D_at(32):.0f}, and it gave out near 194. For octo-double it is $D \\approx {D_at(63.5):.0f}$.")
        self.say(r"So with octo-double numbers, this form of the equations could never get far past 400, however long it ran.")
        self.play(FadeOut(VGroup(f, Dl, Dn, Gl, Gn, axis, marks, al, eaten, el, f1, f2, f3, ft, self.cap)), run_time=0.7)


class S07_Basin(Film):
    stage = r"11 \quad A week at 78: Newton's basin"

    def construct(self):
        pts = frontier_points()
        ax, lab = self.frontier_axes()
        c1 = self.front(ax, pts, 0.0, 8.9)
        self.add(ax, lab, c1)
        c2 = self.front(ax, pts, 8.9, 16.0)
        self.play(Create(c2), run_time=2.4, rate_func=linear)
        self.say(r"Longer numbers took the walk to 78. Then, for a week, nothing got past it.")
        q = self.quote("just jump to 100")
        self.play(FadeIn(q, shift=0.2 * DOWN), run_time=0.5)
        nl = NumberLine(x_range=[70, 102, 5], length=8.6, color=GREY_, include_numbers=True, font_size=22).move_to(ax.c2p(20.0, 150))
        back = SurroundingRectangle(nl, color=GREY_, stroke_width=1, fill_color="#0d1117", fill_opacity=0.92, buff=0.55)
        back.stretch_to_fit_height(2.2).shift(0.4 * UP)
        nlab = VGroup(Tex("$D$", font_size=24, color=GREY_).next_to(nl, RIGHT, buff=0.15),
                      Tex("sketch", font_size=20, color=GREY_).move_to(back.get_corner(UL) + 0.5 * RIGHT + 0.22 * DOWN))
        self.play(FadeIn(back), Create(nl), FadeIn(nlab), run_time=0.7)
        self.say(r"So the question became an experiment: jumps of every size, from two starting points.", hold=0.3)
        jumps = [(73.3, 79.0, True), (78.1, 79.3, True), (73.3, 82.0, False), (78.1, 85.0, False), (73.3, 90.0, False),
                 (78.1, 100.0, False)]
        drawn = VGroup()
        for a, b, ok in jumps:
            arc = ArcBetweenPoints(nl.n2p(a), nl.n2p(b), angle=-PI / 3.4, color=GREEN_ if ok else RED_, stroke_width=3)
            mark = (Dot(nl.n2p(b), color=GREEN_, radius=0.07) if ok
                    else Cross(Dot(nl.n2p(b), radius=0.1), stroke_color=RED_, stroke_width=4))
            self.play(Create(arc), run_time=0.45)
            self.play(FadeIn(mark, scale=1.5), run_time=0.2)
            drawn.add(arc, mark)
        lim = DashedLine(nl.n2p(79.3) + 0.25 * DOWN, nl.n2p(79.3) + 1.0 * UP, color=GOLD_, stroke_width=3)
        ll = Tex("79.3", font_size=24, color=GOLD_).next_to(lim, UP, buff=0.05).shift(0.35 * LEFT)
        self.play(Create(lim), FadeIn(ll), run_time=0.5)
        self.say(r"None landed past 79.3. The wall was not the size of the step. It was Newton's need for a close first guess: past 78, the region of good guesses, the basin, had become tiny.")
        self.say(r"A longer stride could not help. The prediction itself had to become far more accurate.")
        self.play(FadeOut(VGroup(ax, lab, c1, c2, q, nl, nlab, back, drawn, lim, ll, self.cap)), run_time=0.7)


class S08_Taylor(Film):
    stage = r"12 \quad Taylor continuation in $D$"

    def construct(self):
        f = MathTex(r"z(D_0 + h) = z_0 + z_1 h + z_2 h^2 + \cdots + z_{24} h^{24}", font_size=44).move_to([0, 2.55, 0])
        fn = self.foot(r"B. Cochelin, Computers \& Structures 53, 1181 (1994)")
        self.play(FadeIn(fn), run_time=0.3)
        self.say(r"Found in the literature, the asymptotic numerical method (Cochelin, 1994): expand the whole solution $z$ as a Taylor series in $D$.", hold=0.4)
        self.play(Write(f), run_time=1.8)
        self.wait(2.4)
        self.say(r"It is the Taylor step again, this time across dimensions instead of across $x$.")
        ax = Axes(x_range=[0, 2.0, 0.5], y_range=[0.2, 1.3, 0.5], x_length=9.6, y_length=3.3, tips=False,
                  axis_config={"color": GREY_, "stroke_width": 2, "include_ticks": False}).shift(0.25 * DOWN)
        xl = Tex(r"$D_0$", font_size=26, color=GOLD_).next_to(ax.c2p(0, 0.2), DOWN, buff=0.12)
        xr = Tex(r"$D \rightarrow$", font_size=26, color=GREY_).next_to(ax.c2p(2.0, 0.2), DOWN, buff=0.12)
        sk = Tex("sketch: a toy branch", font_size=20, color=GREY_).next_to(ax.c2p(0.03, 0.2), UR, buff=0.08)
        true = ax.plot(lambda h: 1.0 / (1.0 + 0.8 * h), x_range=[0, 2.0, 0.01], color=BLUE_, stroke_width=4)
        base = Dot(ax.c2p(0, 1.0), color=GOLD_, radius=0.08)

        def partial(n):
            def s(h):
                return sum((-0.8 * h) ** k for k in range(n + 1))
            end = next((k * 0.01 for k in range(201) if not 0.2 <= s(k * 0.01) <= 1.3), 2.0)
            return ax.plot(s, x_range=[0, end - 0.01, 0.01], color=GOLD_, stroke_width=3)

        self.play(Create(ax), FadeIn(xl), FadeIn(xr), FadeIn(sk), Create(true), FadeIn(base), run_time=1.4)
        cur = partial(1)
        ol = Tex("order 1", font_size=28, color=GOLD_).move_to(ax.c2p(1.75, 1.15))
        self.play(Create(cur), FadeIn(ol), run_time=0.9)
        self.say(r"At a solved $D_0$, all 24 coefficients come from the same table $J$ that Newton already uses.", hold=0.3)
        for n in (2, 4, 8, 16, 24):
            self.play(Transform(cur, partial(n)), Transform(ol, Tex(f"order {n}", font_size=28, color=GOLD_).move_to(ol)), run_time=0.75)
            self.wait(0.25)
        a = 0.95
        step = DashedLine(ax.c2p(a, 0.2), ax.c2p(a, 1.0 / (1.0 + 0.8 * a)), color=GREEN_, stroke_width=3)
        sl = Tex(r"the step: where the last term has\\fallen to $\varepsilon$ of the first", font_size=22, color=GREEN_, tex_environment="flushleft").next_to(ax.c2p(a, 0.285), RIGHT, buff=0.12)
        land = Dot(ax.c2p(a, 1.0 / (1.0 + 0.8 * a)), color=GREEN_, radius=0.08)
        self.play(Create(step), FadeIn(sl), FadeIn(land, scale=2), run_time=0.8)
        self.say(r"The series itself says how far it can be trusted. Step there, and the guess is good to many digits.")
        self.say(r"Then recompute the horizon data at every guess Newton tries, so their errors start as small as the arithmetic allows. Newton finishes the job.")
        self.play(FadeOut(VGroup(f, ax, xl, xr, sk, true, base, cur, ol, step, sl, land, fn)), run_time=0.6)
        pts = frontier_points()
        fx, lab = self.frontier_axes()
        c1 = self.front(fx, pts, 0.0, 16.0)
        self.play(FadeIn(VGroup(fx, lab, c1)), run_time=0.6)
        c2 = self.front(fx, pts, 16.0, 18.05)
        self.play(Create(c2), run_time=2.6)
        self.say(r"The wall at 78 fell. The walk, now a chain of such Taylor steps, passed 86, 96, 127 and 146 in two days.")
        self.play(FadeOut(VGroup(fx, lab, c1, c2, self.cap)), run_time=0.7)


class S09_FiveWalls(Film):
    stage = r"13 \quad Five walls in one day"

    def construct(self):
        pts = frontier_points()
        ax = Axes(x_range=[18.0, 18.9, 0.2], y_range=[140, 215, 10], x_length=6.2, y_length=5.0, tips=False,
                  axis_config={"color": GREY_, "stroke_width": 2, "include_numbers": True, "font_size": 20},
                  x_axis_config={"decimal_number_config": {"num_decimal_places": 1}}).to_edge(LEFT, buff=0.7).shift(0.3 * UP)
        ax.stretch_to_fit_width(5.4).to_edge(LEFT, buff=0.7)
        xl = Tex("day", font_size=22, color=GREY_).next_to(ax.x_axis, RIGHT, buff=0.1)
        yl = Tex("$D$", font_size=24, color=GREY_).next_to(ax.y_axis, UP, buff=0.1)
        q = self.quote("we need to keep going above 150")
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(q), run_time=0.8)
        walls = [(18.0, 18.12, "152.9", r"a stalled step, replayed from its\\saved input: two defects found"),
                 (18.12, 18.21, "162.3", r"a hypothesis tested before acting:\\precision ruled out"),
                 (18.21, 18.42, "175", r"the series loses its digits:\\build it in octo-double"),
                 (18.42, 18.56, "188", r"a hang: dump the stack\\before the kill"),
                 (18.56, 18.67, "193.8", r"quad-double's floor: correct in\\octo-double, 127 digits")]
        rows = VGroup()
        for i, (a, b, D, what) in enumerate(walls):
            row = VGroup(Tex(D, font_size=30, color=RED_), Tex(what, font_size=26, tex_environment="flushleft"))
            row.arrange(RIGHT, buff=0.3)
            rows.add(row)
        rows.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        x0 = max(row[0].get_right()[0] for row in rows) + 0.3
        for row in rows:
            row[1].shift((x0 - row[1].get_left()[0]) * RIGHT)
        rows.next_to(ax, RIGHT, buff=0.9).shift(0.15 * UP)
        self.say(r"Past 150 the walls came quickly. Each one fell to its own diagnosis.", hold=0.3)
        segs = VGroup()
        for i, (a, b, D, what) in enumerate(walls):
            seg = self.front(ax, pts, a, b)
            self.play(Create(seg), run_time=0.8)
            self.play(FadeIn(rows[i], shift=0.2 * LEFT), run_time=0.5)
            self.wait(1.7)
            segs.add(seg)
        last = self.front(ax, pts, 18.67, 18.9)
        self.play(Create(last), run_time=1.0)
        self.say(r"Replay a failure on its stored input. Test the guess first. Build the next precision before it is needed.")
        self.play(FadeOut(VGroup(ax, xl, yl, q, rows, segs, last, self.cap)), run_time=0.7)


class S10_Climb(Film):
    stage = r"14 \quad The slow climb"

    def construct(self):
        q = self.quote("Keep going to 400")
        self.play(FadeIn(q, shift=0.2 * DOWN), run_time=0.5)
        # the weak harmonics again, and where their held values come from
        roles = VGroup(Tex("harmonics 1 and 3: solved for", font_size=30, color=BLUE_),
                       Tex("harmonics 5 to 13: held at given values", font_size=30, color=GREEN_),
                       Tex("harmonics 15 and up: set to zero", font_size=30, color=GREY_))
        roles.arrange(DOWN, buff=0.25, aligned_edge=LEFT).move_to([0, 0.9, 0])
        self.play(FadeIn(roles), run_time=0.6)
        self.say(r"Recall the first wall. Harmonics 5 to 13 of the wave are tiny, and the equations barely hold them: the weak harmonics. Past $D = 73$ the chain does not solve for them at all. It holds them at given values and solves for the rest.")
        self.say(r"Where do those values come from?", hold=1.4)
        self.play(FadeOut(roles), run_time=0.4)
        hp_, fit8 = held_points(), held_fit()
        hax = Axes(x_range=[30, 270, 40], y_range=[0.008, 0.042, 0.01], x_length=10.4, y_length=4.1, tips=False,
                   axis_config={"color": GREY_, "stroke_width": 2, "include_numbers": True, "font_size": 20},
                   y_axis_config={"decimal_number_config": {"num_decimal_places": 2}}).shift(0.35 * UP)
        hx = Tex("$D$", font_size=26, color=GREY_).next_to(hax.x_axis, RIGHT, buff=0.12)
        hy = Tex(r"harmonic 5 of the wave at the centre, times $\sqrt{D - 2}$", font_size=22, color=GREY_).next_to(hax, UP, buff=0.05).align_to(hax, LEFT).shift(0.4 * RIGHT)
        dots = VGroup(*[Dot(hax.c2p(D, v), radius=0.035, color=BLUE_) for D, v in hp_])
        self.play(Create(hax), FadeIn(hx), FadeIn(hy), run_time=1.0)
        self.say(f"Up to $D = 73$ the solver could still solve for every one of them, and did, {len(hp_)} times. Here is harmonic 5 in each of those full solutions.", hold=0.4)
        self.play(LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.02), run_time=2.6)
        self.wait(2.2)
        c_in = hax.plot(fit8, x_range=[hp_[0][0], 73.0, 0.25], color=GOLD_, stroke_width=3.5)
        self.say(r"The points lie on a smooth curve, because the solutions change gently with $D$. Fit a polynomial through them.", hold=0.4)
        self.play(Create(c_in), run_time=1.6)
        self.wait(2.0)
        edge = DashedLine(hax.c2p(73, 0.008), hax.c2p(73, 0.042), color=GREY_, stroke_width=2)
        el = Tex("73: the last full solution", font_size=20, color=GREY_).next_to(hax.c2p(73, 0.040), RIGHT, buff=0.12)
        c_out = hax.plot(fit8, x_range=[73.0, 264.0, 0.5], color=GOLD_, stroke_width=3.5, stroke_opacity=0.7)
        self.say(r"Then follow that curve onward, past 73, where the solver can no longer check it.", hold=0.4)
        self.play(Create(edge), FadeIn(el), run_time=0.6)
        self.play(Create(c_out), run_time=3.0, rate_func=linear)
        mark = Dot(hax.c2p(200, fit8(200.0)), color=GREEN_, radius=0.08)
        ml = Tex("at $D = 200$, harmonic 5 is held here", font_size=22, color=GREEN_).next_to(mark, UP, buff=0.15)
        self.play(FadeIn(mark, scale=2), FadeIn(ml), run_time=0.5)
        self.say(r"At every $D$ the chain reads harmonic 5 off this curve and holds it there. Harmonics 7 to 13 each have a curve of their own. That is the fit.")
        self.say(r"A fit is a guess. Inside 38 to 73, fits of degree 6, 8 and 10 agree to about $10^{-12}$, then slowly part. Chains run on each of them differ in $\Delta$ by about $3 \times 10^{-11}$ near $D = 200$: that spread is the error bar.")
        self.play(FadeOut(VGroup(hax, hx, hy, dots, c_in, c_out, edge, el, mark, ml)), run_time=0.6)
        # the climb
        pts = frontier_points()
        ax, lab = self.frontier_axes()
        c1 = self.front(ax, pts, 0.0, 18.9)
        self.play(FadeIn(VGroup(ax, lab, c1)), run_time=0.6)
        self.say(r"On those held values the chain climbed on.", hold=0.3)
        c2 = self.front(ax, pts, 18.9, 28.3)
        self.play(Create(c2), run_time=4.0, rate_func=linear)
        self.say(r"215. Then 242.6. Each flat stretch a wall of its own, each passed in days.")
        # the fold: a branch that turns back
        fold_ax = Axes(x_range=[0, 1, 1], y_range=[0, 1, 1], x_length=3.4, y_length=2.2, tips=False,
                       axis_config={"color": GREY_, "stroke_width": 1.5, "include_ticks": False}).move_to(ax.c2p(9.5, 215))
        P = [fold_ax.c2p(0.85 - 0.75 * (2 * t - 1) ** 2, 0.08 + 0.84 * t) for t in [k / 60 for k in range(61)]]
        br = VMobject(color=RED_, stroke_width=4).set_points_as_corners(P)
        fl = Tex("sketch: a fold. The path of solutions turns back.", font_size=20, color=GREY_).next_to(fold_ax, DOWN, buff=0.08)
        fx = Tex("$D$", font_size=20, color=GREY_).next_to(fold_ax.x_axis, RIGHT, buff=0.08)
        self.play(Create(fold_ax), FadeIn(fx), Create(br), FadeIn(fl), run_time=1.6)
        self.say(r"At 259.9 the chain met a fold: the path of solutions of the computer's own equations turned back on itself. It sat there for four days.")
        # what made it: the kink's power outgrew the series
        self.play(FadeOut(VGroup(fold_ax, br, fl, fx)), run_time=0.4)
        Dv = ValueTracker(200.0)
        sig = lambda: 0.5 * Dv.get_value() + 1.6                 # the horizon exponent on the chain, about n/2 + 2.6
        nl = NumberLine(x_range=[95, 140, 5], length=4.6, color=GREY_, include_numbers=True, font_size=18,
                        numbers_to_include=[100, 110, 120, 130, 140]).move_to(ax.c2p(8.4, 178))
        nt = Tex(r"power $\sigma$ of the kink at the horizon\\{\footnotesize numerical evidence (notes/LARGE\_D\_BASICS.md)}", font_size=20, color=GREY_).next_to(nl, DOWN, buff=0.3)
        kept = Line(nl.n2p(130) + 0.5 * DOWN, nl.n2p(130) + 1.15 * UP, color=GOLD_, stroke_width=4)
        kl = Tex("130 terms kept", font_size=20, color=GOLD_).next_to(kept, UP, buff=0.05)
        mk = always_redraw(lambda: Triangle(color=RED_ if sig() > 130 else BLUE_, fill_opacity=1.0).scale(0.11).rotate(PI).move_to(nl.n2p(sig()) + 0.2 * UP))
        ml2 = always_redraw(lambda: Tex(f"$D = {Dv.get_value():.1f}$", font_size=20, color=RED_ if sig() > 130 else BLUE_).next_to(nl.n2p(sig()) + 0.3 * UP, UP, buff=0.12))
        self.play(Create(nl), FadeIn(nt), Create(kept), FadeIn(kl), FadeIn(mk), FadeIn(ml2), run_time=1.0)
        self.say(r"Why a fold? Near the horizon every wrong solution has a kink: it goes like $(1 - x)^{\sigma}$. And the power $\sigma$ grows with $D$, about half of $D$.")
        self.say(r"The horizon's series is how the solver demands smoothness there. But it was cut off after 130 terms, and a kink of power $\sigma$ first shows at term $\sigma$.", hold=0.3)
        self.play(Dv.animate.set_value(259.9), run_time=5.0, rate_func=linear)
        self.wait(0.8)
        self.say(r"Near $D = 257$ the power passed 130. The cut-off series could no longer see the kink, so it no longer demanded smoothness, and at 259.9 the path folded.")
        mk.clear_updaters()
        ml2.clear_updaters()
        c3 = self.front(ax, pts, 28.3, 34.2)
        self.say(r"So the cure was to tie the number of terms to $D$. The fold was gone, and the chain reached 263.6.", hold=0.3)
        self.play(Create(c3), run_time=3.0, rate_func=linear)
        self.wait(1.6)
        self.play(FadeOut(VGroup(nl, nt, kept, kl, mk, ml2, q)), run_time=0.4)
        rate = VGroup(Tex("4 hours for each unit of $D$", font_size=34, color=RED_),
                      Tex("$D = 400$: 23 days away", font_size=34, color=RED_)).arrange(DOWN, buff=0.2).move_to(ax.c2p(11, 215))
        self.play(FadeIn(rate, shift=0.2 * UP), run_time=0.6)
        self.say(r"But every unit of $D$ now cost four hours. No table had shown the time left to the goal.")
        q2 = self.quote("we should stop doing our calculations at large d until we can find a better method")
        q2.scale_to_fit_width(min(q2.width, 9.0)).to_corner(UR, buff=0.35)
        self.play(FadeIn(q2, shift=0.2 * DOWN), run_time=0.5)
        self.say(r"The chain was paused.")
        self.play(FadeOut(VGroup(ax, lab, c1, c2, c3, rate, q2, self.cap)), run_time=0.7)


class S11_Basics(Film):
    stage = r"15 \quad Back to basics"

    def construct(self):
        q = self.quote("go back to basics if you have to")
        self.play(FadeIn(q, shift=0.2 * DOWN), run_time=0.5)
        pts = delta_points()
        ax = Axes(x_range=[200, 265, 10], y_range=[1.62, 1.70, 0.02], x_length=6.0, y_length=3.6, tips=False,
                  axis_config={"color": GREY_, "stroke_width": 2, "include_numbers": True, "font_size": 20},
                  y_axis_config={"decimal_number_config": {"num_decimal_places": 2}}).to_edge(LEFT, buff=0.8).shift(0.3 * UP)
        xl = Tex("$D$", font_size=24, color=GREY_).next_to(ax.x_axis, RIGHT, buff=0.1)
        yl = MathTex(r"\Delta", font_size=30, color=GREY_).next_to(ax.y_axis, UP, buff=0.1)
        tail = [p for p in pts if p[0] >= 200]
        curve = VMobject(color=BLUE_, stroke_width=4).set_points_as_corners([ax.c2p(*p) for p in tail])
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), run_time=0.7)
        self.play(Create(curve), run_time=2.0)
        self.say(r"Was the wall in the physics, or in our equations? A fresh derivation, made before reading our notes:", hold=3.2)
        notes = VGroup(Tex(r"the chain's $\Delta$ leaves the smooth\\curve from $D \approx 254$", font_size=28, color=RED_, tex_environment="flushleft"),
                       Tex(r"nothing singular in the\\continuum near $D \approx 265$", font_size=28, color=GREEN_, tex_environment="flushleft"),
                       Tex(r"the limits belonged to\\the discrete system", font_size=28, tex_environment="flushleft"))
        notes.arrange(DOWN, buff=0.4, aligned_edge=LEFT).next_to(ax, RIGHT, buff=0.9)
        for n in notes:
            self.play(FadeIn(n, shift=0.2 * LEFT), run_time=0.5)
            self.wait(1.5)
        self.say(r"The mistake was to treat large $D$ as a continuation problem at all.")
        self.play(FadeOut(VGroup(ax, xl, yl, curve, notes, q)), run_time=0.6)

        g = MathTex(r"G \approx 0.7^{-(D-3)}", font_size=56, color=ORANGE_).move_to([0, 2.3, 0])
        self.play(Write(g), run_time=1.0)
        fn = self.foot([r"C. Gundlach, Phys. Rev. D 55, 695 (1997), arXiv:gr-qc/9604019, Eq. (19)",
                        "our record of the search: issue 1089 of the repository"])
        self.play(FadeIn(fn), run_time=0.3)
        self.say(r"And the amplifier $G$, the reason for ever longer numbers? Three searches of the published work on this solution were sent out. One, reading Gundlach's 1997 construction of it, came back with what $G$ is.")
        self.play(g.animate.scale(0.55).move_to([5.8, 3.35, 0]), run_time=0.6)
        # which equation governs each field (README, "Large D"; hp/src/model.rs; the last one as in form S)
        rows = [(r"p,\ m", BLUE_, r"x\,(x-c)\,\partial_x p = -\omega x\,\partial_\theta p + \cdots", "the wave equation (mirrored for $m$)", 2.25),
                (r"c", PURPLE_, r"x\,\partial_x c = (n-1)\,c\,(a^2-1)", "the slicing condition", 1.55),
                (r"a", GREEN_, r"2x\,\partial_x a = a\,(p^2+m^2) - (n-1)\,a\,(a^2-1)", "the Hamiltonian constraint", 0.75),
                (r"a", GOLD_, r"\omega\,\partial_\theta B = \big[(n-1)+q\big]\,B - (n-1), \quad B = 1/a^2", "the momentum constraint", -0.3)]
        E = []
        for sym, col, eq, name, y in rows:
            sy = MathTex(sym, font_size=36, color=col).move_to([-6.2, y, 0])
            E.append([sy, MathTex(eq, font_size=30).move_to([0, y, 0]).align_to([-5.4, 0, 0], LEFT), Tex(name, font_size=26, color=col)])
        xr = max(e[1].get_right()[0] for e in E) + 0.45
        for (sy, eq, nm), (_, _, _, _, y) in zip(E, rows):
            nm.move_to([0, y, 0]).align_to([xr, 0, 0], LEFT)
        da = Tex(r"along $x$: how $a$ changes with position. Mass curves space.", font_size=22, color=GREEN_)
        db = Tex(r"through the echo: how $a$ changes in time. Energy flow changes the mass inside.", font_size=22, color=GOLD_)
        da.move_to([0, 0.75 - 0.4, 0]).align_to([-5.4, 0, 0], LEFT)
        db.move_to([0, -0.3 - 0.45, 0]).align_to([-5.4, 0, 0], LEFT)
        lg = VGroup(Tex(r"$n = D - 2$ \quad $\omega = 2\pi/\Delta$ \quad $\theta$: the time within one echo, as an angle", font_size=20, color=GREY_),
                    Tex(r"$q = \big((x-c)\,p^2 + (x+c)\,m^2\big)/x$ \quad the last line takes $\partial_x a$ from the line above it", font_size=20, color=GREY_))
        lg.arrange(DOWN, buff=0.1, aligned_edge=LEFT).move_to([0, -1.45, 0]).align_to([-5.4, 0, 0], LEFT)
        e1, e2, ea, eb = (VGroup(*e) for e in E)
        self.play(FadeIn(e1), run_time=0.5)
        self.play(FadeIn(e2), FadeIn(lg[0]), run_time=0.5)
        self.say(r"Step back to the equations. Einstein's equations are several equations at once. The wave equation moves the two parts of the wave, $p$ and $m$. One of Einstein's equations, the slicing condition, gives $c$.")
        self.play(FadeIn(ea), FadeIn(da), run_time=0.5)
        self.play(FadeIn(eb), FadeIn(db), FadeIn(lg[1]), run_time=0.5)
        self.say(r"And two of them give $a$, the field that carries the mass. The Hamiltonian constraint says how $a$ changes from one $x$ to the next. The momentum constraint says how $a$ changes through the echo.")
        self.say(r"They are called constraints because the fields must obey them at every moment. For a true solution the two agree, so a solver needs only one of them.")
        used = Tex("our solver used this one", font_size=22, color=GREEN_).next_to(da, RIGHT, buff=0.4)
        left = Tex("and never looked at this one", font_size=22, color=RED_).next_to(db, RIGHT, buff=0.3)
        self.play(FadeIn(used), run_time=0.4)
        self.play(FadeIn(left), run_time=0.4)
        self.say(r"Ours used the first. It stepped $a$ along $x$ like the other fields: the pieces of station 6. It never looked at the second.")
        self.play(Indicate(ea[1], color=GREEN_), run_time=1.0)
        self.say(r"But the first one alone lets in a stray solution, which acts like a speck of mass at the centre. From the horizon to the last node it grows by exactly $G$. Every rounding error in the horizon data becomes a speck of it.")
        self.play(Indicate(eb[1], color=GOLD_), run_time=1.0)
        self.say(r"The second forbids it. A fixed speck of mass does not shrink with the echoes, so it cannot repeat from one echo to the next. We had left out the one equation that rules it out.")
        self.say(r"So the solver spent its digits holding the stray mass down: 40 of them by $D = 260$.")
        self.play(FadeOut(VGroup(e1, e2, ea, eb, da, db, lg, used, left)), g.animate.scale(1 / 0.55).move_to([0, 2.3, 0]), run_time=0.6)
        # z without a
        A = z_row(0.6)
        zl = MathTex("z", font_size=50).next_to(A["all"], LEFT, buff=0.3)
        cnt = Tex("769 numbers", font_size=32, color=GOLD_).next_to(A["all"], DOWN, buff=0.35)
        zt = VGroup(Tex("centre data", font_size=20, color=BLUE_).next_to(A["centre"], UP, buff=0.1),
                    Tex("five nodes, four fields each", font_size=20, color=GREY_).next_to(A["all"], UP, buff=0.1),
                    Tex("horizon data", font_size=20, color=ORANGE_).next_to(A["horizon"], UP, buff=0.1))
        self.play(FadeIn(A["all"]), FadeIn(zl), FadeIn(cnt), FadeIn(zt), run_time=0.8)
        self.say(r"Gundlach's 1997 construction of the critical solution does it the other way: it takes $a$ straight from the momentum constraint, at every $x$. Here is $z$, the row of numbers the solver has to find.", hold=0.5)
        a_blocks = VGroup(*[nd["a"] for nd in A["nodes"]])
        self.play(*[bk[0].animate.set_fill(RED_, opacity=0.85) for bk in a_blocks], run_time=0.8)
        self.wait(0.8)
        S = z_row(0.6, fields="pmc", horizon=32)
        moves = [FadeOut(a_blocks), ReplacementTransform(A["centre"], S["centre"]), ReplacementTransform(A["horizon"], S["horizon"]),
                 ReplacementTransform(A["delta"], S["delta"])]
        moves += [ReplacementTransform(na[k], ns[k]) for na, ns in zip(A["nodes"], S["nodes"]) for k in "pmc"]
        zt2 = VGroup(Tex("centre data", font_size=20, color=BLUE_).next_to(S["centre"], UP, buff=0.1),
                     Tex("five nodes, three fields each", font_size=20, color=GREY_).next_to(S["all"], UP, buff=0.1),
                     Tex("horizon data", font_size=20, color=ORANGE_).next_to(S["horizon"], UP, buff=0.1).shift(0.25 * RIGHT))
        self.play(*moves, Transform(cnt, Tex("577 numbers", font_size=32, color=GOLD_).move_to(cnt)), Transform(zt, zt2),
                  zl.animate.next_to(S["all"], LEFT, buff=0.3), run_time=1.8)
        self.say(r"What does that do to $z$? Each node loses its table for $a$, and the horizon data lose $a$'s half. 769 numbers become 577.")
        self.say(r"$a$ is not thrown away. Wherever a piece needs it, it is worked out from $p$, $m$ and $c$ by one small solve around the echo. It is no longer something to guess.")
        self.play(g.animate.set_opacity(0.25), run_time=0.6)
        gone = Cross(g, stroke_color=GREEN_, stroke_width=5)
        self.play(Create(gone), run_time=0.5)
        self.say(r"And with $a$ no longer guessed, the stray mass has nowhere to live. $G$ has nothing to amplify.")
        self.play(FadeOut(VGroup(S["all"], zl, cnt, zt)), run_time=0.5)
        # why it is faster
        why = VGroup(Tex(r"a guess for $D = 250$ misses by: \quad old form $10^{192}$ \quad new form $10^{-11}$", font_size=28, color=GREEN_),
                     Tex(r"$D = 300$: one direct solve, 24 minutes \quad $D = 400$: one direct solve, 20 minutes", font_size=28, color=GREEN_),
                     Tex(r"577 unknowns instead of 769, and no horizon data to recompute", font_size=28, color=GREEN_),
                     Tex(r"at $D = 300$, quad-double numbers were enough: 4 minutes", font_size=28, color=GREEN_),
                     Tex(r"first results (issue 1088), not yet reviewed", font_size=20, color=GREY_))
        why.arrange(DOWN, buff=0.38, aligned_edge=LEFT).move_to([0, 0.1, 0])
        self.play(FadeIn(why[0], shift=0.2 * UP), FadeIn(why[4]), run_time=0.6)
        self.say(r"Why is it faster? First, a guess made far away now works. Take $D = 250$, guessed from solutions below 246. In the old form that guess misses by $10^{192}$: the stray mass, blown up by $G$. In the new form it misses by $10^{-11}$.")
        self.play(FadeIn(why[1], shift=0.2 * UP), run_time=0.6)
        self.say(r"So Newton can start from far away, and there is no walk: one direct solve for $D = 300$, in 24 minutes, and one for $D = 400$, in 20. The chain had needed four hours for every single unit of $D$.")
        self.play(FadeIn(why[2], shift=0.2 * UP), FadeIn(why[3], shift=0.2 * UP), run_time=0.6)
        self.say(r"And each solve is lighter. There are a quarter fewer unknowns, the horizon data need no recomputing, and the numbers can be shorter: at $D = 300$, quad-double was enough for the table $J$, and the solve took 4 minutes.")
        self.say(r"These are first results, not yet reviewed. The weak harmonics were held as before. Released one at a time at $D = 300$, they moved $\Delta$ by $10^{-11}$.")
        self.say(r"And $D = 1000$ does not work yet. The horizon's series loses its digits there: the next wall, once more at the horizon. A new variable for that series is being tested.")
        self.play(FadeOut(VGroup(g, gone, why, fn, self.cap)), run_time=0.7)


class S12_Outro(Film):
    def construct(self):
        pts = delta_points()
        ax = Axes(x_range=[0, 420, 100], y_range=[1.4, 3.8, 0.4], x_length=10.6, y_length=4.2, tips=False,
                  axis_config={"color": GREY_, "stroke_width": 2, "include_numbers": True, "font_size": 22},
                  y_axis_config={"decimal_number_config": {"num_decimal_places": 1}}).shift(0.6 * UP)
        xl = Tex("spacetime dimensions $D$", font_size=26, color=GREY_).next_to(ax.x_axis, DOWN, buff=0.3)
        yl = MathTex(r"\Delta", font_size=36, color=GREY_).next_to(ax.y_axis, UP, buff=0.1)
        good = [p for p in pts if p[0] <= 254.0]
        curve = VMobject(color=BLUE_, stroke_width=4).set_points_as_corners([ax.c2p(*p) for p in good])
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), run_time=0.8)
        self.play(Create(curve), run_time=3.0)
        n = Tex(f"{len(good)} computed points, $D = {good[0][0]:.3f}$ to ${good[-1][0]:.0f}$", font_size=28, color=BLUE_)
        n.move_to(ax.c2p(200, 2.6))
        arrow = Arrow(ax.c2p(262, good[-1][1]), ax.c2p(415, good[-1][1]), color=GOLD_, stroke_width=5, buff=0)
        goal = Tex("400, 2000, \\ldots", font_size=30, color=GOLD_).next_to(arrow, UP, buff=0.12)
        self.play(FadeIn(n), run_time=0.5)
        self.say(r"Each wall was a fact about the solver, not about the solution. Each one reshaped the method.", hold=0.4)
        self.play(GrowArrow(arrow), FadeIn(goal), run_time=1.0)
        self.wait(2.6)
        self.say(r"The quest continues.")
        end = VGroup(Tex("code, data and every number's source:", font_size=30, color=GREY_),
                     Tex("github.com/nmoinvaz/choptuik-large-d", font_size=40),
                     Tex(r"Nathan Moinvaziri \quad $\cdot$ \quad x.com/nmnvzr \quad $\cdot$ \quad made with Claude Fable 5.2", font_size=26, color=GREY_),
                     Tex(r"animated with Manim \quad $\cdot$ \quad voice: Kokoro, a synthetic voice", font_size=22, color=GREY_))
        end.arrange(DOWN, buff=0.35)
        self.play(FadeOut(VGroup(ax, xl, yl, curve, n, arrow, goal, self.cap)), run_time=0.7)
        self.play(FadeIn(end, shift=0.2 * UP), run_time=0.8)
        self.wait(4.0)
        self.play(FadeOut(end), run_time=0.8)
