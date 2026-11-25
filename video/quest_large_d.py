#!/usr/bin/env python3
"""The quest to large D: a silent film that explains the large-D calculation through the evolution of the solver.

    video/build.sh [l|m|h]        # every scene, joined into video/dist/quest_large_d.mp4 (l: 480p15, h: 1080p30)

One Manim scene per station, in the order of SCENES. The curves are the repository's own: Delta(D) from
hp/data/delta_all.csv (computed rows), the highest D against time from paper/ai/main.tex and paper/ai/frontier.csv,
and the equations' grip on each harmonic from hp/data/held_grip.csv. The statements follow ALGORITHM.md, the notes and
paper/ai/main.tex. Curves marked "sketch" on screen are illustrations, not data.

Manim Community (The Manim Community Developers, https://www.manim.community), after 3Blue1Brown's manim. The formulas
are compiled by tectonic through the stand-ins in video/tools (this Manim's multi-part MathTex needs dvisvgm, so every
formula here is one string).
"""
import csv
import math
import re
from pathlib import Path

from manim import *

ROOT = Path(__file__).resolve().parent.parent
config.tex_template = TexTemplate(tex_compiler="pdflatex", output_format=".pdf")
config.background_color = "#0d1117"

BLUE_, GOLD_, RED_, GREEN_, GREY_, ORANGE_, PURPLE_ = "#58C4DD", "#FFD866", "#FF6B6B", "#7CE38B", "#8B949E", "#F78C6C", "#C792EA"
DEL4 = 3.4454                                   # Delta at D = 4, to four decimals (hp/data/delta_all.csv)
SCENES = ["S00_Title", "S01_Echo", "S02_Question", "S03a_Dot", "S03b_Cone", "S03c_Numbers", "S03d_Join", "S03e_Newton", "S04_Walk", "S05_FirstWall", "S06_Amplifier",
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


def log10G(D):
    """Decades of the horizon's amplification, G = 0.7^-(D-3)."""
    return (D - 3.0) * math.log10(1.0 / 0.7)


def stairs(ax, pts, **kw):
    """A step curve through (x, y) points: flat to the next x, then up."""
    P = [ax.c2p(*pts[0])]
    for (_, y0), (x1, y1) in zip(pts, pts[1:]):
        P += [ax.c2p(x1, y0), ax.c2p(x1, y1)]
    return VMobject(**kw).set_points_as_corners(P)


class Film(Scene):
    stage = ""

    def setup(self):
        self.cap = None
        if self.stage:
            self.add(Tex(self.stage, font_size=26, color=GREY_).to_corner(UL, buff=0.35))

    def say(self, text, hold=None, color=WHITE):
        """A caption at the foot of the frame, held long enough to read."""
        new = Tex(text, font_size=34, color=color)
        if new.width > 13.0:
            new.scale_to_fit_width(13.0)
        new.to_edge(DOWN, buff=0.35)
        if self.cap is None:
            self.play(FadeIn(new, shift=0.15 * UP), run_time=0.5)
        else:
            self.play(FadeOut(self.cap), FadeIn(new, shift=0.15 * UP), run_time=0.5)
        self.cap = new
        n = len(re.sub(r"\\[a-zA-Z]+|[${}^_\\]", "", text))
        self.wait(hold if hold is not None else max(1.8, n / 15.0))

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
        self.play(Write(title), run_time=1.6)
        self.play(FadeIn(sub, shift=0.2 * UP), FadeIn(cred, shift=0.2 * UP), run_time=1.0)
        self.wait(3.2)
        self.play(FadeOut(VGroup(wave, title, sub, cred)), run_time=0.8)


class S01_Echo(Film):
    stage = r"1 \quad The echo"

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
        self.say(r"Tune a collapsing wave of scalar field to the very edge of forming a black hole.", hold=0.6)
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
        self.say(r"Zoom in on the collapse by a factor $e^{\Delta}$, and the picture repeats.", hold=0.4)
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
        self.say(r"In logarithmic time the solution is periodic. Its period is the echo period $\Delta$.", hold=0.5)
        self.play(GrowFromCenter(br), FadeIn(bl), run_time=0.8)
        self.wait(2.6)
        self.say(r"In four dimensions $\Delta \approx 3.445$: every echo is about " + f"{math.exp(DEL4):.0f}" + " times smaller than the last.")
        self.play(FadeOut(VGroup(ax2, tau, per, br, bl, self.cap)), run_time=0.7)


class S02_Question(Film):
    stage = r"2 \quad The question"

    def construct(self):
        pts = delta_points()
        ax = Axes(x_range=[0, 270, 50], y_range=[1.4, 3.8, 0.4], x_length=10.6, y_length=4.4, tips=False,
                  axis_config={"color": GREY_, "stroke_width": 2, "include_numbers": True, "font_size": 22},
                  y_axis_config={"decimal_number_config": {"num_decimal_places": 1}}).shift(0.6 * UP)
        xl = Tex("spacetime dimensions $D$", font_size=26, color=GREY_).next_to(ax.x_axis, DOWN, buff=0.3)
        yl = MathTex(r"\Delta", font_size=36, color=GREY_).next_to(ax.y_axis, UP, buff=0.1)
        low = [p for p in pts if p[0] <= 12.0]
        curve = VMobject(color=BLUE_, stroke_width=4).set_points_as_corners([ax.c2p(*p) for p in low])
        d4 = Dot(ax.c2p(4.0, min(pts, key=lambda p: abs(p[0] - 4.0))[1]), color=GOLD_, radius=0.07)
        l4 = Tex(r"$D = 4$", font_size=26, color=GOLD_).next_to(d4, RIGHT, buff=0.15)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), run_time=1.0)
        self.say(r"The echo period depends on the number of spacetime dimensions $D$.", hold=0.5)
        self.play(Create(curve), run_time=2.0)
        self.play(FadeIn(d4, scale=2), FadeIn(l4), run_time=0.6)
        self.wait(1.2)
        q = Tex("?", font_size=120, color=GOLD_).move_to(ax.c2p(150, 2.6))
        arrow = Arrow(ax.c2p(20, 2.05), ax.c2p(120, 1.85), color=GOLD_, stroke_width=4, buff=0)
        self.play(GrowArrow(arrow), FadeIn(q, scale=0.6), run_time=1.0)
        self.say(r"The quest: follow $\Delta(D)$ as far into large $D$ as the solver can be made to go.")
        self.say(r"Every wall on the way taught the solver something. This is the solver, in the order it was shaped.")
        self.play(FadeOut(VGroup(ax, xl, yl, curve, d4, l4, q, arrow, self.cap)), run_time=0.7)


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


def x_line(y=-1.9):
    """The x axis from the centre to the horizon, with the five nodes."""
    line = NumberLine(x_range=[0, 1, 0.1], length=10.0, color=GREY_, include_ticks=False).shift(y * UP)
    ends = VGroup(Tex(r"the centre, $x = 0$", font_size=24, color=BLUE_).next_to(line.n2p(0), DOWN, buff=0.15),
                  Tex(r"the horizon, $x = 1$", font_size=24, color=ORANGE_).next_to(line.n2p(1), DOWN, buff=0.15))
    ticks = VGroup(*[Line(line.n2p(v) + 0.1 * DOWN, line.n2p(v) + 0.1 * UP, color=WHITE, stroke_width=2) for v in CUTS[1:6]])
    nl = Tex("the five nodes", font_size=22, color=GREY_).next_to(line.n2p(0.5), DOWN, buff=0.15)
    return line, VGroup(ends, ticks, nl)


def low_axes():
    """Delta against D at low D, where single points can be told apart."""
    ax = Axes(x_range=[2, 14, 2], y_range=[2.2, 3.6, 0.2], x_length=9.6, y_length=4.2, tips=False,
              axis_config={"color": GREY_, "stroke_width": 2, "include_numbers": True, "font_size": 22},
              y_axis_config={"decimal_number_config": {"num_decimal_places": 1}}).shift(0.5 * UP)
    xl = Tex("spacetime dimensions $D$", font_size=26, color=GREY_).next_to(ax.x_axis, DOWN, buff=0.3)
    yl = MathTex(r"\Delta", font_size=36, color=GREY_).next_to(ax.y_axis, UP, buff=0.1)
    return ax, VGroup(xl, yl)


class S03a_Dot(Film):
    stage = r"3 \quad One dot, one solve"

    def construct(self):
        pts = delta_points()
        ax, lab = low_axes()
        at = lambda D: min(pts, key=lambda q: abs(q[0] - D))
        curve = VMobject(color=BLUE_, stroke_width=3, stroke_opacity=0.35).set_points_as_corners(
            [ax.c2p(*q) for q in pts if 3.9 <= q[0] <= 13.0])
        dots = VGroup(*[Dot(ax.c2p(*at(D)), color=BLUE_, radius=0.07) for D in range(4, 14)])
        self.play(Create(ax), FadeIn(lab), Create(curve), run_time=1.2)
        self.play(LaggedStart(*[FadeIn(d, scale=2.5) for d in dots], lag_ratio=0.18), run_time=2.0)
        self.say(r"Every dot on this curve is one computation: at that $D$, find the echoing solution, and read off its period $\Delta$.")
        D6 = at(6.0)
        pick = Dot(ax.c2p(*D6), color=GOLD_, radius=0.11)
        guide = VGroup(DashedLine(ax.c2p(D6[0], 2.2), ax.c2p(*D6), color=GOLD_, stroke_width=2),
                       DashedLine(ax.c2p(2, D6[1]), ax.c2p(*D6), color=GOLD_, stroke_width=2))
        tag = Tex(f"$D = 6$: \\quad $\\Delta = {D6[1]:.4f}$", font_size=30, color=GOLD_).next_to(pick, UR, buff=0.2)
        self.play(FadeIn(pick, scale=2), Create(guide), FadeIn(tag), run_time=0.9)
        self.wait(1.4)
        self.say(r"The quest to large $D$ is a long line of such solves, each harder than the last. Here is what one solve is.")
        self.play(FadeOut(VGroup(ax, lab, curve, dots, pick, guide, tag, self.cap)), run_time=0.7)


class S03b_Cone(Film):
    stage = r"4 \quad The centre, the light cone and $x$"

    def construct(self):
        C, T = [-2.0, 2.6, 0.0], 4.3
        P = lambda r, t: [C[0] + r, C[1] - T + t, 0.0]          # distance r from the centre, time t from the foot
        centre = Line(P(0, 0), C, color=BLUE_, stroke_width=4)
        base = Line(P(0, 0), P(8.4, 0), color=GREY_, stroke_width=2)
        tl = Tex(r"time $\uparrow$", font_size=24, color=GREY_).next_to(centre, LEFT, buff=0.15)
        rl = Tex(r"distance from the centre $\rightarrow$", font_size=24, color=GREY_).next_to(base, DOWN, buff=0.45).align_to(base, RIGHT)
        cl = Tex("the centre", font_size=24, color=BLUE_).next_to(P(0, 0), DOWN, buff=0.12)
        sk = Tex("sketch, drawn as if space were flat", font_size=20, color=GREY_).to_corner(UR, buff=0.35)
        self.play(Create(centre), Create(base), FadeIn(tl), FadeIn(rl), FadeIn(cl), FadeIn(sk), run_time=1.2)
        self.say(r"To see what the solver solves, draw a map of the collapse. Distance from the centre runs to the right, and time runs upward.")
        star = Dot(C, color=RED_, radius=0.1)
        sl = Tex("the collapse", font_size=26, color=RED_).next_to(star, UP, buff=0.12)
        self.play(FadeIn(star, scale=3), FadeIn(sl), run_time=0.7)
        self.say(r"The collapse is one place at one moment: the centre, at the top of the map.")
        # light moves on diagonals
        f1 = Line(P(2.6, 0), P(0, 2.6), color=GOLD_, stroke_width=3)
        d1 = Dot(P(2.6, 0), color=GOLD_, radius=0.09)
        self.say(r"Nothing travels faster than light. On this map a flash of light heading inward follows a diagonal.", hold=0.3)
        self.play(Create(f1), MoveAlongPath(d1, f1), run_time=2.2, rate_func=linear)
        self.play(FadeOut(d1), f1.animate.set_stroke(opacity=0.3), run_time=0.4)
        edge = Line(P(T, 0), C, color=ORANGE_, stroke_width=5)
        d2 = Dot(P(T, 0), color=ORANGE_, radius=0.1)
        self.say(r"One flash is timed just right: it arrives at the centre exactly at the moment of collapse.", hold=0.3)
        self.play(Create(edge), MoveAlongPath(d2, edge), run_time=3.0, rate_func=linear)
        self.play(Flash(star, color=ORANGE_), FadeOut(d2), run_time=0.6)
        mirror = DashedLine(P(-T, 0), C, color=ORANGE_, stroke_width=3)
        rim = Ellipse(width=2 * T, height=0.55, color=ORANGE_, stroke_width=2).move_to(P(0, 0))
        self.play(Create(mirror), Create(rim), run_time=1.2)
        self.say(r"Flashes like it close in from every side. Together they form a cone with its tip at the collapse: the light cone.")
        self.play(FadeOut(mirror), FadeOut(rim), run_time=0.5)
        # outside the cone: too late
        when = DashedLine(P(0, T), P(8.4, T), color=GREY_, stroke_width=2)
        wl = Tex("the moment of collapse", font_size=22, color=GREY_).next_to(when, UP, buff=0.08).align_to(when, RIGHT)
        E = Dot(P(6.2, 0.7), color=PURPLE_, radius=0.09)
        el = Tex("something out here", font_size=22, color=PURPLE_).next_to(E, DOWN, buff=0.12)
        late = Line(P(6.2, 0.7), P(6.2 - (T - 0.7), T), color=PURPLE_, stroke_width=3)
        d3 = Dot(P(6.2, 0.7), color=PURPLE_, radius=0.09)
        self.play(Create(when), FadeIn(wl), FadeIn(E, scale=2), FadeIn(el), run_time=0.8)
        self.say(r"Now take anything outside the cone. Even its light is still on the way when the collapse happens.", hold=0.3)
        self.play(Create(late), MoveAlongPath(d3, late), run_time=2.4, rate_func=linear)
        x3 = Cross(Dot(P(6.2 - (T - 0.7), T), radius=0.14), stroke_color=PURPLE_, stroke_width=4)
        tooL = Tex("too late", font_size=24, color=PURPLE_).next_to(x3, DOWN, buff=0.12).shift(0.45 * RIGHT)
        self.play(FadeIn(x3), FadeIn(tooL), FadeOut(d3), run_time=0.5)
        self.wait(1.8)
        inside = Polygon(P(0, 0), P(T, 0), C, stroke_width=0, fill_color=ORANGE_, fill_opacity=0.14)
        hz = Tex("the horizon", font_size=28, color=ORANGE_).next_to(edge.get_center(), RIGHT, buff=0.3)
        self.play(FadeOut(VGroup(E, el, late, x3, tooL)), FadeIn(inside), FadeIn(hz), run_time=0.8)
        self.say(r"So nothing outside can affect the collapse. All that shapes it lies inside the cone. The cone's edge is called the horizon.")
        note = Tex("(not the horizon of a black hole)", font_size=22, color=GREY_).next_to(hz, DOWN, buff=0.1).align_to(hz, LEFT)
        self.play(FadeIn(note), run_time=0.4)
        self.say(r"The solver therefore works only inside: from the centre out to the horizon.")
        self.play(FadeOut(VGroup(when, wl, f1, note)), run_time=0.5)
        # x: a ruler that shrinks with the time left
        rays = VGroup(*[Line(P(v * T, 0), C, color=GREY_, stroke_width=1.5) for v in CUTS[1:6]])
        def ruler(t):
            left = T - t
            g = VGroup(Line(P(0, t), P(left, t), color=WHITE, stroke_width=3))
            for v in [0.0] + CUTS[1:6] + [1.0]:
                g.add(Line(P(v * left, t - 0.09), P(v * left, t + 0.09), color=WHITE, stroke_width=2.5))
            return g
        times = [0.0, T * (1 - 1 / 2.2), T * (1 - 1 / 4.84)]
        r0 = ruler(times[0])
        self.say(r"Inside, each echo is the last one, shrunk. So measure position with a ruler that shrinks at the same rate.", hold=0.3)
        self.play(Create(r0), run_time=0.8)
        self.play(LaggedStart(*[Create(r) for r in rays], lag_ratio=0.12), run_time=1.2)
        kept = VGroup(r0.copy().set_stroke(opacity=0.35))
        self.add(kept)
        for t in times[1:]:
            self.play(Transform(r0, ruler(t)), run_time=1.5)
            kept.add(r0.copy().set_stroke(opacity=0.35))
            self.add(kept)
            self.wait(0.5)
        c0 = Tex(r"$x = 0$", font_size=28, color=BLUE_).next_to(cl, DOWN, buff=0.06)
        c1 = Tex(r"$x = 1$", font_size=28, color=ORANGE_).next_to(P(T, 0), DOWN, buff=0.12)
        ch = Tex(r"$x = 0.5$", font_size=24, color=GREY_).next_to(P(0.5 * T, 0), DOWN, buff=0.12)
        self.play(FadeIn(c0), FadeIn(c1), FadeIn(ch), run_time=0.6)
        self.say(r"That ruler is $x$: the distance from the centre, divided by the time left. The centre is $x = 0$, and the horizon is $x = 1$.")
        self.say(r"In $x$, every echo looks the same. Keep the horizon in mind: at large $D$, nearly every wall ahead begins at that edge.")
        self.play(FadeOut(VGroup(centre, base, tl, rl, cl, sk, star, sl, edge, inside, hz, rays, r0, kept, c0, c1, ch, self.cap)), run_time=0.7)


class S03c_Numbers(Film):
    stage = r"5 \quad The solution, written as numbers"

    def construct(self):
        # where things are: rays of constant x fanning down from the collapse
        C, T, Wd = [-6.0, 2.5, 0.0], 4.3, 3.7
        foot = lambda v: [C[0] + v * Wd, C[1] - T, 0.0]
        centre = Line(foot(0), C, color=BLUE_, stroke_width=4)
        cone = Line(foot(1), C, color=ORANGE_, stroke_width=4)
        rays = VGroup(*[Line(foot(v), C, color=GREY_, stroke_width=1.5) for v in CUTS[1:6]])
        star = Dot(C, color=RED_, radius=0.09)
        sl = Tex("the collapse", font_size=24, color=RED_).next_to(star, RIGHT, buff=0.15)
        tl = Tex(r"time $\uparrow$", font_size=22, color=GREY_).next_to(centre, LEFT, buff=0.12)
        rl = Tex(r"distance from the centre $\rightarrow$", font_size=22, color=GREY_).move_to([C[0] + 2.3, C[1] - T - 0.75, 0])
        c0 = Tex(r"$x = 0$", font_size=26, color=BLUE_).next_to(foot(0), DOWN, buff=0.12)
        c1 = Tex(r"$x = 1$", font_size=26, color=ORANGE_).next_to(foot(1), DOWN, buff=0.12)
        sk = Tex("sketch", font_size=20, color=GREY_).move_to([C[0] + 3.3, C[1] - 0.1, 0])
        hz = Tex("the horizon", font_size=26, color=ORANGE_).next_to(cone.get_center(), RIGHT, buff=0.25)
        self.play(FadeIn(VGroup(centre, cone, rays, star, sl, tl, rl, sk, c0, c1, hz)), run_time=0.9)

        # what lives there: waves that repeat with every echo (the real wave at the centre, D = 38)
        D38, co = harmonics(38.0)
        top = 1.25 * max(abs(wave_of(co, 2 * math.pi * k / 200)) for k in range(200))
        wax = Axes(x_range=[0, 4 * math.pi, 2 * math.pi], y_range=[-top, top, top], x_length=5.4, y_length=1.9, tips=False,
                   axis_config={"color": GREY_, "stroke_width": 1.5, "include_ticks": False}).move_to([3.5, 1.85, 0])
        wave = wax.plot(lambda t: wave_of(co, t), x_range=[0, 4 * math.pi, 0.02], color=GOLD_, stroke_width=3.5)
        wl = Tex(f"the scalar wave at the centre over two echoes, at $D = {D38:.0f}$", font_size=22, color=GREY_).next_to(wax, UP, buff=0.08)
        br = BraceBetweenPoints(wax.c2p(0, -top), wax.c2p(2 * math.pi, -top), direction=DOWN, color=GREY_)
        bl = Tex(r"one echo, $\Delta$", font_size=22, color=GREY_).next_to(br, DOWN, buff=0.05)
        self.play(Create(wax), FadeIn(wl), Create(wave), FadeIn(br), FadeIn(bl), run_time=1.6)
        self.say(r"Back on the map: at every $x$ between the centre and the horizon, the solution repeats with each echo. Each field there is a wave with period $\Delta$.")
        chips = VGroup(Tex(r"$p,\ m$: the scalar wave, heading out and in along light rays", font_size=24, color=BLUE_),
                       Tex(r"$a,\ c$: the curved spacetime it travels through", font_size=24, color=PURPLE_))
        chips.arrange(DOWN, buff=0.15, aligned_edge=LEFT).move_to([3.4, -0.55, 0])
        self.play(FadeIn(chips, shift=0.15 * UP), run_time=0.6)
        self.say(r"There are four fields: two for the scalar wave, and two for the spacetime it bends.")
        # a wave as a few numbers: its harmonics, each on its own scale
        self.play(FadeOut(chips), FadeOut(br), FadeOut(bl), run_time=0.4)
        minis = VGroup()
        for k in (1, 3, 5):
            size = math.hypot(*co[k])
            m = Axes(x_range=[0, 2 * math.pi, math.pi], y_range=[-1.3, 1.3, 1], x_length=1.6, y_length=1.0, tips=False,
                     axis_config={"color": GREY_, "stroke_width": 1, "include_ticks": False})
            c = m.plot(lambda t, k=k, size=size: wave_of(co, t, ks=(k,)) / size, x_range=[0, 2 * math.pi, 0.02],
                       color=GOLD_, stroke_width=2.5)
            minis.add(VGroup(m, c, Tex(f"harmonic {k}\\\\size {size:.2g}", font_size=20, color=GREY_).next_to(m, DOWN, buff=0.05)))
        minis.arrange(RIGHT, buff=0.3).move_to([3.5, -0.85, 0])
        plus = Tex(r"$=$ \quad a sum of pure tones, each drawn on its own scale:", font_size=22, color=GREY_).next_to(minis, UP, buff=0.1).align_to(minis, LEFT)
        self.play(FadeIn(plus), LaggedStart(*[FadeIn(m, shift=0.2 * DOWN) for m in minis], lag_ratio=0.3), run_time=1.6)
        self.say(r"A repeating wave is a sum of pure tones, its harmonics. Harmonic $k$ fits $k$ times into one echo, and each is about ten times smaller than the last.")
        self.say(r"So a wave need not be stored point by point. Its harmonics are enough: 32 numbers per wave. The smallest of them will be the first wall.")
        self.play(FadeOut(VGroup(centre, cone, rays, star, sl, tl, rl, c0, c1, sk, hz, wax, wave, wl, minis, plus)), run_time=0.6)

        # the list z, to scale: 64 centre numbers, five nodes of 128, 64 horizon numbers, Delta
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
        mid = Tex(r"five nodes $\times$ four fields $\times$ 32 numbers", font_size=24).next_to(blocks[3], DOWN, buff=0.15)
        lc = Tex(r"centre data:\\the numbers that start\\the solution at $x = 0$", font_size=22, color=BLUE_).next_to(blocks[0], UP, buff=0.2).align_to(blocks[0], LEFT)
        lh = Tex(r"horizon data:\\the numbers that start\\it at $x = 1$", font_size=22, color=ORANGE_).next_to(blocks[6], UP, buff=0.2).align_to(blocks[6], RIGHT)
        ld = Tex(r"$\Delta$ itself", font_size=24, color=GOLD_).next_to(blocks[7], DOWN, buff=0.15).align_to(blocks[7], RIGHT)
        self.play(LaggedStart(*[FadeIn(b, shift=0.2 * DOWN) for b in blocks[1:6]], lag_ratio=0.12), FadeIn(mid), run_time=1.6)
        self.say(r"Store those 32 numbers for each of the four fields, on five rays between the centre and the horizon: the nodes.")
        self.play(FadeIn(blocks[0]), FadeIn(blocks[6]), FadeIn(blocks[7]), FadeIn(lc), FadeIn(lh), FadeIn(ld), run_time=0.9)
        self.say(r"Add a few numbers for each end, and the period $\Delta$ itself.")
        tot = Tex("769 numbers", font_size=40, color=GOLD_).move_to([0, -1.2, 0])
        self.play(FadeIn(zl), FadeIn(tot, shift=0.2 * UP), run_time=0.7)
        self.say(r"That list of 769 numbers is $z$. One $z$ is one candidate: a whole echoing solution, and its period.")
        self.play(FadeOut(VGroup(blocks, zl, mid, lc, lh, ld, tot, self.cap)), run_time=0.7)


class S03d_Join(Film):
    stage = r"6 \quad Do the pieces join?"

    def construct(self):
        line, deco = x_line()
        eq = MathTex(r"x\,(x - c)\,\frac{\partial p}{\partial x} \;=\; \cdots", font_size=44).move_to([0, 2.55, 0])
        self.play(Create(line), FadeIn(deco), run_time=0.9)
        self.say(r"Is a given $z$ a solution? Einstein's equations, with the wave equation, say how every field changes from one $x$ to the next.", hold=0.4)
        self.play(Write(eq), run_time=1.2)
        self.wait(3.0)
        P = [piece(line, i, 1.0) for i in range(6)]
        tl = Tex("Taylor steps", font_size=24, color=GREEN_).next_to(line.n2p(0.5), UP, buff=0.18)
        self.say(r"So the numbers at one node predict the numbers at the next. That prediction is a Taylor step:", hold=0.3)
        self.play(LaggedStart(*[Create(P[i]) for i in (1, 2, 3, 4)], lag_ratio=0.6), FadeIn(tl), run_time=2.6)
        self.say(r"a polynomial of high order that follows the equations across the gap between two nodes.")
        self.say(r"At the two ends this fails: the factor in front becomes zero, and the equation would divide by it.", hold=0.4)
        self.play(eq[0][0].animate.set_color(BLUE_), eq[0][1:6].animate.set_color(ORANGE_), run_time=0.6)
        self.play(Indicate(eq[0][0], color=BLUE_), Indicate(eq[0][1:6], color=ORANGE_), run_time=1.2)
        self.wait(2.2)
        self.say(r"$x = 0$ is the centre. And $c$ is the speed of light on our map, so $x - c = 0$ is the horizon: light there runs along the edge, not across it.")
        cl = Tex("series from the centre", font_size=24, color=BLUE_).next_to(line.n2p(0.15), UP, buff=0.18)
        hl = Tex("series from the horizon", font_size=24, color=ORANGE_).next_to(line.n2p(0.85), UP, buff=0.18)
        self.say(r"So near each end the solution is written as a power series that is smooth there, started by that end's own numbers in $z$: the centre data and the horizon data.", hold=0.4)
        self.play(Create(P[0]), FadeIn(cl), run_time=1.3)
        self.play(Create(P[5]), FadeIn(hl), run_time=1.3)
        self.wait(2.6)
        g = gaps(line, 1.0)
        self.say(r"Where two pieces meet, they should agree. For a guessed $z$ they do not.", hold=0.4)
        self.play(LaggedStart(*[GrowFromCenter(s) for s in g], lag_ratio=0.2), run_time=1.4)
        self.wait(1.6)
        F = MathTex(r"F(z)", font_size=44, color=RED_)
        Ft = Tex("= all the gaps, one for each number in $z$", font_size=28, color=RED_).next_to(F, RIGHT, buff=0.25)
        VGroup(F, Ft).move_to([0, 2.55, 0])
        self.play(FadeOut(eq), run_time=0.4)
        self.play(FadeIn(F), FadeIn(Ft), run_time=0.6)
        self.say(r"Those gaps are the residual $F(z)$. Only the true fields, with the true $\Delta$, close every one of them.")
        root = MathTex(r"F(z) = 0", font_size=56, color=GOLD_).move_to([0, 2.55, 0])
        self.play(FadeOut(F), FadeOut(Ft), FadeIn(root, scale=1.2), run_time=0.7)
        self.say(r"That is what a solve is: find the $z$ with $F(z) = 0$. A solution is a root.")
        self.play(FadeOut(VGroup(line, deco, tl, cl, hl, g, root, self.cap, *P)), run_time=0.7)


class S03e_Newton(Film):
    stage = r"7 \quad Newton's method"

    def construct(self):
        line, deco = x_line()
        e = ValueTracker(1.0)
        shot = always_redraw(lambda: VGroup(*[piece(line, i, e.get_value()) for i in range(6)]))
        gp = always_redraw(lambda: gaps(line, e.get_value()))
        eq = MathTex(r"J\,dz = -F(z)", font_size=52, color=GOLD_).move_to([-2.2, 2.55, 0])
        res = Tex(r"largest gap: $10^{-2}$", font_size=34, color=RED_).move_to([3.4, 2.55, 0])
        self.add(line, deco, shot, gp)
        self.wait(0.4)
        self.say(r"Newton's method closes the gaps. It measures how each gap responds to each number in $z$: a table $J$, the Jacobian.", hold=0.4)
        self.play(Write(eq), FadeIn(res), run_time=1.2)
        self.wait(3.0)
        self.say(r"One linear solve with $J$ gives the change $dz$ that should cancel every gap at once. Apply it, and repeat.", hold=0.4)
        for scale, expo in [(0.35, 4), (0.06, 8), (0.0, 16), (0.0, 32)]:
            new = Tex(r"largest gap: $10^{-" + str(expo) + r"}$", font_size=34, color=RED_ if expo < 16 else GREEN_).move_to(res)
            self.play(e.animate.set_value(scale), Transform(res, new), run_time=1.0)
            self.wait(0.5)
        self.say(r"Each round roughly squares the error. But only if the first guess was already close: remember that.")
        self.play(FadeOut(VGroup(line, deco, shot, gp, eq, res)), run_time=0.5)
        pts = delta_points()
        ax, lab = low_axes()
        at = lambda D: min(pts, key=lambda q: abs(q[0] - D))
        curve = VMobject(color=BLUE_, stroke_width=3, stroke_opacity=0.35).set_points_as_corners(
            [ax.c2p(*q) for q in pts if 3.9 <= q[0] <= 13.0])
        dots = VGroup(*[Dot(ax.c2p(*at(D)), color=BLUE_, radius=0.07) for D in (4, 5)])
        new = Dot(ax.c2p(*at(6.0)), color=GOLD_, radius=0.11)
        self.play(FadeIn(VGroup(ax, lab, curve, dots)), run_time=0.6)
        self.play(FadeIn(new, scale=3), Flash(new, color=GOLD_), run_time=0.8)
        self.say(r"The gaps are closed. That is one dot on the curve: its $\Delta$ is the last number in $z$. Now for the next $D$.")
        self.play(FadeOut(VGroup(ax, lab, curve, dots, new, self.cap)), run_time=0.7)


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
        self.say(r"A solved point is the best first guess for its neighbour.", hold=0.3)
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
        self.say(r"In double-double arithmetic, about 32 digits, the walk climbed for a day. At $D = 60.96$ it stopped.")
        self.play(FadeOut(VGroup(ax, xl, yl, dot, path, full, Dl, Dn, wall, self.cap)), run_time=0.7)


class S05_FirstWall(Film):
    stage = r"9 \quad The first wall: weak harmonics"

    def construct(self):
        pts = frontier_points()
        ax, lab = self.frontier_axes()
        self.play(Create(ax), FadeIn(lab), run_time=0.8)
        c1 = self.front(ax, pts, 0.0, 4.0)
        self.play(Create(c1), run_time=1.6)
        w = Tex("60.96", font_size=26, color=RED_).next_to(ax.c2p(3.0, 61), UP, buff=0.1)
        self.play(FadeIn(w), run_time=0.4)
        self.say(r"Why here? The answer is in the harmonics: the pure tones that add up to the wave over one echo.")
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
        self.say(r"Pinned, the walk moved again, to 67.8, 71 and 73.3. Quad-double arithmetic and new node layouts took it to 78.1.")
        self.play(FadeOut(VGroup(ax, lab, c1, c2, w, fix, self.cap)), run_time=0.7)


class S06_Amplifier(Film):
    stage = r"10 \quad The amplifier"

    def construct(self):
        f = MathTex(r"G \approx 0.7^{-(D-3)}", font_size=60, color=ORANGE_).move_to([-3.3, 2.2, 0])
        self.say(r"One mode of the equations amplifies any error in the horizon data, the numbers that start the horizon's series, by a factor $G$.", hold=0.4)
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
        base, per = [-5.6, -0.9, 0], 11.2 / 130.0
        axis = Line(base, [base[0] + 130 * per, base[1], 0], color=GREY_, stroke_width=2)
        types = [("double", 16, GREY_), ("double-double", 32, BLUE_), ("quad-double", 64, GREEN_), ("octo-double", 127, PURPLE_)]
        marks = VGroup()
        for i, (name, dig, col) in enumerate(types):
            x = base[0] + dig * per
            tick = Line([x, base[1] - 0.12, 0], [x, base[1] + 1.05, 0], color=col, stroke_width=2.5)
            t = Tex(f"{name}\\\\{dig} digits", font_size=22, color=col).next_to(tick, UP, buff=0.08)
            marks.add(VGroup(tick, t))
        al = Tex("decimal digits", font_size=22, color=GREY_).next_to(axis, DOWN, buff=0.12).align_to(axis, RIGHT)
        eaten = always_redraw(lambda: Rectangle(width=max(0.01, log10G(D.get_value()) * per), height=0.42, stroke_width=0,
                                                 fill_color=ORANGE_, fill_opacity=0.8).move_to(
            [base[0] + log10G(D.get_value()) * per / 2, base[1] + 0.3, 0]))
        el = Tex("digits lost to $G$", font_size=22, color=ORANGE_).next_to(axis, UP, buff=0.75).align_to(axis, LEFT)
        self.play(FadeIn(VGroup(Dl, Dn, Gl, Gn)), Create(axis), FadeIn(al), run_time=0.8)
        self.play(LaggedStart(*[FadeIn(m, shift=0.15 * UP) for m in marks], lag_ratio=0.2), FadeIn(eaten), FadeIn(el), run_time=1.6)
        self.say(r"A number is stored as a sum of 1, 2, 4 or 8 ordinary doubles. More parts, more digits.")
        def flag(Dv, text, col, drop):
            x = base[0] + log10G(Dv) * per
            stem = Line([x, base[1] - 0.05, 0], [x, base[1] - drop, 0], color=col, stroke_width=2)
            return VGroup(stem, Tex(text, font_size=22, color=col).next_to(stem, DOWN, buff=0.06).align_to(stem, LEFT))

        self.say(r"$G$ grows exponentially with $D$, and it eats those digits.", hold=0.3)
        self.play(D.animate.set_value(80), run_time=2.6, rate_func=linear)
        f1 = flag(80, r"$D \approx 80$: double-double gives out", BLUE_, 1.25)
        self.play(FadeIn(f1, shift=0.1 * DOWN), run_time=0.5)
        self.wait(1.0)
        self.play(D.animate.set_value(131), run_time=1.8, rate_func=linear)
        self.say(r"At $D = 131$ it is already $10^{20}$. Every stage of the solver needs longer numbers as $D$ climbs.", hold=2.2)
        self.play(D.animate.set_value(194), run_time=2.2, rate_func=linear)
        f2 = flag(194, r"$D \approx 194$: quad-double gives out", GREEN_, 0.75)
        self.play(FadeIn(f2, shift=0.1 * DOWN), run_time=0.5)
        self.wait(1.0)
        self.play(D.animate.set_value(260), run_time=2.2, rate_func=linear)
        self.wait(0.6)
        Dn.clear_updaters()
        Gn.clear_updaters()
        self.say(r"Double-double carried the walk to about 80, quad-double to about 194, octo-double beyond.")
        self.play(FadeOut(VGroup(f, Dl, Dn, Gl, Gn, axis, marks, al, eaten, el, f1, f2, self.cap)), run_time=0.7)


class S07_Basin(Film):
    stage = r"11 \quad A week at 78"

    def construct(self):
        pts = frontier_points()
        ax, lab = self.frontier_axes()
        c1 = self.front(ax, pts, 0.0, 8.9)
        self.add(ax, lab, c1)
        c2 = self.front(ax, pts, 8.9, 16.0)
        self.play(Create(c2), run_time=2.4, rate_func=linear)
        self.say(r"Then, for a week, nothing passed $D = 78$.")
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
        self.say(r"None landed past 79.3. The wall was not the step size. It was the basin: how close a guess Newton needs.")
        self.say(r"A longer stride could not help. The prediction itself had to become far more accurate.")
        self.play(FadeOut(VGroup(ax, lab, c1, c2, q, nl, nlab, back, drawn, lim, ll, self.cap)), run_time=0.7)


class S08_Taylor(Film):
    stage = r"12 \quad Taylor continuation in $D$"

    def construct(self):
        f = MathTex(r"z(D_0 + h) = z_0 + z_1 h + z_2 h^2 + \cdots + z_{24} h^{24}", font_size=44).move_to([0, 2.55, 0])
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
        self.say(r"At a solved base $D_0$, all 24 coefficients come from one factorization of the Jacobian.", hold=0.3)
        for n in (2, 4, 8, 16, 24):
            self.play(Transform(cur, partial(n)), Transform(ol, Tex(f"order {n}", font_size=28, color=GOLD_).move_to(ol)), run_time=0.75)
            self.wait(0.25)
        a = 0.95
        step = DashedLine(ax.c2p(a, 0.2), ax.c2p(a, 1.0 / (1.0 + 0.8 * a)), color=GREEN_, stroke_width=3)
        sl = Tex(r"the step: where the last term has\\fallen to $\varepsilon$ of the first", font_size=22, color=GREEN_, tex_environment="flushleft").next_to(ax.c2p(a, 0.285), RIGHT, buff=0.12)
        land = Dot(ax.c2p(a, 1.0 / (1.0 + 0.8 * a)), color=GREEN_, radius=0.08)
        self.play(Create(step), FadeIn(sl), FadeIn(land, scale=2), run_time=0.8)
        self.say(r"The series itself says how far it can be trusted. Step there, and the guess is good to many digits.")
        self.say(r"Then recompute the horizon's data at every trial point, so their errors start as small as the arithmetic allows. Newton finishes.")
        self.play(FadeOut(VGroup(f, ax, xl, xr, sk, true, base, cur, ol, step, sl, land)), run_time=0.6)
        pts = frontier_points()
        fx, lab = self.frontier_axes()
        c1 = self.front(fx, pts, 0.0, 16.0)
        self.play(FadeIn(VGroup(fx, lab, c1)), run_time=0.6)
        c2 = self.front(fx, pts, 16.0, 18.05)
        self.play(Create(c2), run_time=2.6)
        self.say(r"The wall at 78 fell. In two days the chain passed 86, 96, 127 and 146.")
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
        pts = frontier_points()
        ax, lab = self.frontier_axes()
        c1 = self.front(ax, pts, 0.0, 18.9)
        self.add(ax, lab, c1)
        q = self.quote("Keep going to 400")
        self.play(FadeIn(q, shift=0.2 * DOWN), run_time=0.5)
        roles = VGroup(Tex("harmonics 1, 3: solved", font_size=24, color=BLUE_),
                       Tex("harmonics 5 to 13: held on a fit of", font_size=24, color=GREEN_),
                       Tex("138 full solutions, $D$ 38 to 73", font_size=24, color=GREEN_),
                       Tex("harmonics 15 and up: zero", font_size=24, color=GREY_))
        roles.arrange(DOWN, buff=0.12, aligned_edge=LEFT).move_to(ax.c2p(8.6, 190))
        self.play(FadeIn(roles), run_time=0.6)
        self.say(r"The solutions form a smooth family in $D$. A fit of the full solutions carries the weak harmonics upward.", hold=0.3)
        c2 = self.front(ax, pts, 18.9, 28.3)
        self.play(Create(c2), run_time=4.0, rate_func=linear)
        self.say(r"215. Then 242.6. Each flat stretch a wall of its own, each passed in days.")
        self.play(FadeOut(roles), run_time=0.4)
        # the fold: a branch that turns back
        fold_ax = Axes(x_range=[0, 1, 1], y_range=[0, 1, 1], x_length=3.4, y_length=2.2, tips=False,
                       axis_config={"color": GREY_, "stroke_width": 1.5, "include_ticks": False}).move_to(ax.c2p(9.5, 215))
        P = [fold_ax.c2p(0.85 - 0.75 * (2 * t - 1) ** 2, 0.08 + 0.84 * t) for t in [k / 60 for k in range(61)]]
        br = VMobject(color=RED_, stroke_width=4).set_points_as_corners(P)
        fl = Tex("sketch: a fold. The branch turns back.", font_size=20, color=GREY_).next_to(fold_ax, DOWN, buff=0.08)
        fx = Tex("$D$", font_size=20, color=GREY_).next_to(fold_ax.x_axis, RIGHT, buff=0.08)
        self.play(Create(fold_ax), FadeIn(fx), Create(br), FadeIn(fl), run_time=1.6)
        self.say(r"At 259.9 the chain met a fold: its own discrete equations turned back. It sat there for four days.")
        c3 = self.front(ax, pts, 28.3, 34.2)
        self.play(Create(c3), run_time=3.0, rate_func=linear)
        self.say(r"Tying the number of terms in the horizon's series to $D$ removed the fold. The chain reached 263.6.")
        self.play(FadeOut(VGroup(fold_ax, br, fl, fx, q)), run_time=0.4)
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
        self.say(r"And the amplifier $G$, the reason for ever longer numbers? A search of the literature found what it is.")
        m1 = Tex("the mass mode", font_size=34, color=ORANGE_).move_to([0, 1.2, 0])
        m2 = Tex("= a violation of the momentum constraint,", font_size=34).next_to(m1, DOWN, buff=0.25)
        m3 = Tex("which our equations never imposed", font_size=34).next_to(m2, DOWN, buff=0.2)
        m4 = Tex(r"the momentum constraint: one of Einstein's equations. It ties how the\\metric function $a$ changes in time to the wave's flow of energy.",
                 font_size=24, color=GREY_).next_to(m3, DOWN, buff=0.45)
        self.play(FadeIn(m1), run_time=0.5)
        self.play(FadeIn(m2), FadeIn(m3), FadeIn(m4), run_time=0.7)
        self.say(r"Our equations left that constraint free, and the solver spent its digits holding the free mode down.")
        self.play(FadeOut(VGroup(m1, m2, m3, m4)), run_time=0.4)
        # the unknowns lose a field
        names, cols = ["p", "m", "a", "c"], [BLUE_, GREEN_, PURPLE_, ORANGE_]
        cells = VGroup()
        for n, c in zip(names, cols):
            r = Rectangle(width=1.5, height=0.9, stroke_color=WHITE, stroke_width=1.5, fill_color=c, fill_opacity=0.4)
            r.add(MathTex(n, font_size=40).move_to(r))
            cells.add(r)
        cells.arrange(RIGHT, buff=0.0).move_to([0, 0.45, 0])
        cl = Tex("the fields at each node", font_size=24, color=GREY_).next_to(cells, UP, buff=0.15)
        self.play(FadeIn(cells), FadeIn(cl), run_time=0.6)
        self.say(r"Gundlach's 1997 construction of the critical solution sets the metric function $a$ from that constraint.", hold=0.5)
        cross = Cross(cells[2], stroke_color=RED_, stroke_width=6)
        self.play(Create(cross), run_time=0.5)
        self.play(FadeOut(cells[2]), FadeOut(cross), cells[3].animate.shift(1.5 * LEFT), run_time=0.8)
        self.play(g.animate.set_opacity(0.25), run_time=0.6)
        gone = Cross(g, stroke_color=GREEN_, stroke_width=5)
        self.play(Create(gone), run_time=0.5)
        self.say(r"With $a$ no longer an unknown, there is no mass mode, and nothing for $G$ to amplify.")
        res = Tex(r"a prototype in this form converged at $D = 100$ in double-double arithmetic", font_size=30, color=GREEN_)
        res.move_to([0, -1.2, 0])
        self.play(FadeIn(res, shift=0.2 * UP), run_time=0.6)
        self.say(r"Whether it stays this easy as $D$ grows is being measured now.")
        self.play(FadeOut(VGroup(g, gone, cells[0], cells[1], cells[3], cl, res, self.cap)), run_time=0.7)


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
                     Tex("animated with Manim", font_size=22, color=GREY_))
        end.arrange(DOWN, buff=0.35)
        self.play(FadeOut(VGroup(ax, xl, yl, curve, n, arrow, goal, self.cap)), run_time=0.7)
        self.play(FadeIn(end, shift=0.2 * UP), run_time=0.8)
        self.wait(4.0)
        self.play(FadeOut(end), run_time=0.8)
