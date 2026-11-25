#!/usr/bin/env python3
"""The quest to large D: a short silent film of how each wall reshaped the solver for Choptuik's echo period.

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
SCENES = ["S00_Title", "S01_Echo", "S02_Question", "S03_Root", "S04_Walk", "S05_FirstWall", "S06_Amplifier",
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
        sub = Tex(r"how each wall reshaped a solver for Choptuik's echo period", font_size=36, color=GREY_)
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


class S03_Root(Film):
    stage = r"3 \quad A solution is a root"

    def construct(self):
        # the unknown vector, to scale: 64 centre numbers, five nodes of 128, 64 horizon numbers, Delta
        W, x, blocks, labels = 11.0, 0.0, VGroup(), VGroup()
        unit = W / 769.0
        parts = [("centre", 64, BLUE_)] + [("node", 128, GREY_)] * 5 + [("horizon", 64, ORANGE_), ("Delta", 6, GOLD_)]
        for name, n, col in parts:
            r = Rectangle(width=n * unit, height=0.55, stroke_color=WHITE, stroke_width=1.5, fill_color=col,
                          fill_opacity=0.55 if name != "node" else 0.18)
            r.move_to([x + n * unit / 2 - W / 2, 2.35, 0])
            if name == "node":
                for j, c in enumerate([BLUE_, GREEN_, PURPLE_, ORANGE_]):
                    sub = Rectangle(width=n * unit / 4, height=0.55, stroke_width=0.8, stroke_color=WHITE, fill_color=c,
                                    fill_opacity=0.35)
                    sub.move_to(r.get_left() + RIGHT * (j + 0.5) * n * unit / 4)
                    r.add(sub)
            blocks.add(r)
            x += n * unit
        labels.add(Tex("centre", font_size=24, color=BLUE_).next_to(blocks[0], DOWN, buff=0.12),
                   Tex("five nodes, four fields each: $p,\\ m,\\ a,\\ c$", font_size=24).next_to(blocks[3], DOWN, buff=0.12),
                   Tex("horizon", font_size=24, color=ORANGE_).next_to(blocks[6], DOWN, buff=0.12),
                   MathTex(r"\Delta", font_size=30, color=GOLD_).next_to(blocks[7], UP, buff=0.1))
        zl = MathTex(r"z", font_size=44).next_to(blocks, LEFT, buff=0.25)
        nz = Tex("769 numbers", font_size=26, color=GREY_).next_to(blocks, UP, buff=0.12).shift(3.5 * LEFT)
        self.say(r"Pack the fields over one echo, and $\Delta$ itself, into a single vector $z$.", hold=0.4)
        self.play(LaggedStart(*[FadeIn(b, shift=0.2 * DOWN) for b in blocks], lag_ratio=0.12), FadeIn(zl), run_time=1.8)
        self.play(FadeIn(labels), FadeIn(nz), run_time=0.8)
        self.wait(1.8)

        # the residual along x: series from both ends, Taylor steps between the nodes, mismatches at the nodes
        line = NumberLine(x_range=[0, 1, 0.1], length=10.0, color=GREY_, include_ticks=False).shift(1.55 * DOWN)
        ends = VGroup(Tex("centre", font_size=24, color=GREY_).next_to(line.n2p(0), DOWN, buff=0.15),
                      Tex("past light cone", font_size=24, color=GREY_).next_to(line.n2p(1), DOWN, buff=0.15))
        nodes = [0.3, 0.4, 0.5, 0.6, 0.7]
        ticks = VGroup(*[Line(line.n2p(v) + 0.1 * DOWN, line.n2p(v) + 0.1 * UP, color=WHITE, stroke_width=2) for v in nodes])
        e = ValueTracker(1.0)
        cuts = [0.0] + nodes + [1.0]
        offs = [0.26, -0.2, 0.22, -0.17, 0.15, -0.24]

        def g(v):
            return 0.55 * math.sin(2.6 * v + 0.4) + 0.25 * math.cos(7.0 * v)

        def pieces():
            out = VGroup()
            for i in range(6):
                a, b = cuts[i], cuts[i + 1]
                P = [line.n2p(a + (b - a) * k / 40) + UP * (0.95 + g(a + (b - a) * k / 40) + e.get_value() * offs[i])
                     for k in range(41)]
                out.add(VMobject(color=[BLUE_, GREEN_, GREEN_, GREEN_, GREEN_, ORANGE_][i], stroke_width=4).set_points_as_corners(P))
            return out

        shot = always_redraw(pieces)
        how = VGroup(Tex(r"series from the centre $\rightarrow$", font_size=22, color=BLUE_).next_to(line.n2p(0.15), UP, buff=0.16),
                     Tex(r"Taylor steps $\rightarrow$", font_size=22, color=GREEN_).next_to(line.n2p(0.5), UP, buff=0.16),
                     Tex(r"$\leftarrow$ series from the horizon", font_size=22, color=ORANGE_).next_to(line.n2p(0.85), UP, buff=0.16))
        self.say(r"Integrate the field equations in pieces: out from the centre, node to node, and in from the horizon.", hold=0.4)
        self.play(Create(line), FadeIn(ends), FadeIn(ticks), run_time=0.8)
        self.play(FadeIn(shot), FadeIn(how), run_time=1.2)
        self.wait(2.4)
        eqs = VGroup(MathTex(r"F(z) = 0", font_size=44), MathTex(r"J\,dz = -F(z)", font_size=44, color=GOLD_))
        eqs.arrange(RIGHT, buff=1.0).move_to([0, 0.75, 0])
        self.say(r"Where the pieces fail to join is the residual $F(z)$. A solution is a root: $F(z) = 0$.", hold=0.4)
        self.play(Write(eqs[0]), run_time=1.0)
        self.wait(2.2)
        res = Tex(r"$|F| = 10^{-2}$", font_size=34, color=RED_).next_to(eqs, RIGHT, buff=0.9)
        self.say(r"Newton's method steps toward the root. Each step solves one linear system with the Jacobian $J$.", hold=0.4)
        self.play(Write(eqs[1]), FadeIn(res), run_time=1.0)
        for scale, expo in [(0.35, 4), (0.06, 8), (0.0, 16), (0.0, 32)]:
            new = Tex(r"$|F| = 10^{-" + str(expo) + r"}$", font_size=34, color=RED_ if expo < 16 else GREEN_).move_to(res)
            self.play(e.animate.set_value(scale), Transform(res, new), run_time=0.9)
            self.wait(0.35)
        self.say(r"Near the root the error squares at every step, as long as the first guess is close enough.")
        self.play(FadeOut(VGroup(blocks, labels, zl, nz, line, ends, ticks, shot, how, eqs, res, self.cap)), run_time=0.7)


class S04_Walk(Film):
    stage = r"4 \quad Walking in $D$"

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
    stage = r"5 \quad The first wall: weak harmonics"

    def construct(self):
        pts = frontier_points()
        ax, lab = self.frontier_axes()
        self.play(Create(ax), FadeIn(lab), run_time=0.8)
        c1 = self.front(ax, pts, 0.0, 4.0)
        self.play(Create(c1), run_time=1.6)
        w = Tex("60.96", font_size=26, color=RED_).next_to(ax.c2p(3.0, 61), UP, buff=0.1)
        self.play(FadeIn(w), run_time=0.4)
        self.say(r"Why here? The solution is a wave over one echo: a sum of odd harmonics, each far smaller than the last.")
        D0, sig = grip(79.77)
        hs = sorted(sig)
        panel = VGroup()
        for i, h in enumerate(hs):
            dec = 30.0 + math.log10(sig[h])
            bar = Rectangle(width=dec * 0.25, height=0.34, stroke_width=0, fill_color=RED_, fill_opacity=0.85 - 0.14 * i)
            name = Tex(f"harmonic {h}", font_size=24)
            val = MathTex(r"10^{" + f"{math.log10(sig[h]):.0f}" + "}", font_size=28, color=RED_)
            row = VGroup(name, bar, val).arrange(RIGHT, buff=0.2)
            panel.add(row)
        panel.arrange(DOWN, buff=0.14, aligned_edge=LEFT)
        for row in panel:
            row[1].align_to(panel[0][1], LEFT)
            row[2].next_to(row[1], RIGHT, buff=0.15)
        head = Tex(f"how firmly the equations hold each harmonic, at $D = {D0:.1f}$", font_size=26, color=GREY_)
        box = VGroup(head, panel).arrange(DOWN, buff=0.25).move_to(ax.c2p(22, 150))
        back = SurroundingRectangle(box, color=GREY_, stroke_width=1, fill_color="#0d1117", fill_opacity=0.92, buff=0.25)
        self.play(FadeIn(back), FadeIn(head), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(r, shift=0.2 * RIGHT) for r in panel], lag_ratio=0.25), run_time=2.0)
        self.say(r"The equations barely pin the higher harmonics down. Their grip falls by decades per harmonic, and with $D$.")
        self.say(f"At $D = {D0:.1f}$ a full Newton step would move harmonic 13 by 0.44. Its true size is 0.00000037.")
        self.say(r"Roundoff times the condition number reached 1 just before the stall. Newton was stepping on noise.")
        self.play(FadeOut(VGroup(back, box)), run_time=0.5)
        fix = Tex(r"hold the weak harmonics on their smooth trend in $D$, solve for everything else", font_size=28, color=GREEN_)
        fix.move_to(ax.c2p(19, 235))
        self.play(FadeIn(fix, shift=0.2 * UP), run_time=0.6)
        self.say(r"The fix: stop asking the equations for what they cannot give. Hold those harmonics, and solve the rest.", hold=0.5)
        c2 = self.front(ax, pts, 4.0, 8.9)
        self.play(Create(c2), run_time=2.2)
        self.say(r"Pinned, the walk moved again, to 67.8, 71 and 73.3. Quad-double arithmetic and new node layouts took it to 78.1.")
        self.play(FadeOut(VGroup(ax, lab, c1, c2, w, fix, self.cap)), run_time=0.7)


class S06_Amplifier(Film):
    stage = r"6 \quad The amplifier"

    def construct(self):
        f = MathTex(r"G \approx 0.7^{-(D-3)}", font_size=60, color=ORANGE_).move_to([-3.3, 2.2, 0])
        self.say(r"One mode of the equations amplifies any error in the horizon's data by a factor $G$.", hold=0.4)
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
    stage = r"7 \quad A week at 78"

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
    stage = r"8 \quad Taylor continuation in $D$"

    def construct(self):
        f = MathTex(r"z(D_0 + h) = z_0 + z_1 h + z_2 h^2 + \cdots + z_{24} h^{24}", font_size=44).move_to([0, 2.55, 0])
        self.say(r"Found in the literature, the asymptotic numerical method (Cochelin, 1994): carry the whole solution as a Taylor series in $D$.", hold=0.4)
        self.play(Write(f), run_time=1.8)
        self.wait(1.0)
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
    stage = r"9 \quad Five walls in one day"

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
    stage = r"10 \quad The slow climb"

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
        self.say(r"Tying the order of the horizon's series to $D$ removed the fold. The chain reached 263.6.")
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
    stage = r"11 \quad Back to basics"

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
        self.play(FadeIn(m1), run_time=0.5)
        self.play(FadeIn(m2), FadeIn(m3), run_time=0.7)
        self.say(r"Our equations left that constraint free, and the solver spent its digits holding the free mode down.")
        self.play(FadeOut(VGroup(m1, m2, m3)), run_time=0.4)
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
