"""Convert trivial inline math $`...`$ in README.md to plain Unicode text, to cut the number of expressions GitHub
must render. Only expressions that become plain text exactly (no remaining TeX) are converted."""
import re, sys
GREEK = {r"\Delta": "Δ", r"\omega": "ω", r"\gamma": "γ", r"\lambda": "λ", r"\varepsilon": "ε", r"\epsilon": "ε",
         r"\theta": "θ", r"\pi": "π", r"\mu": "μ", r"\nu": "ν", r"\xi": "ξ", r"\sigma": "σ", r"\chi": "χ",
         r"\Phi": "Φ", r"\Lambda": "Λ", r"\alpha": "α", r"\beta": "β", r"\kappa": "κ", r"\zeta": "ζ", r"\eta": "η",
         r"\phi": "φ", r"\tau": "τ", r"\Gamma": "Γ", r"\Pi": "Π", r"\varphi": "φ", r"\delta": "δ", r"\rho": "ρ"}
OPS = {r"\infty": "∞", r"\le": "≤", r"\ge": "≥", r"\approx": "≈", r"\pm": "±", r"\times": "×", r"\to": "→",
       r"\sim": "~", r"\ne": "≠", r"\cdot": "·", r"\ll": "≪", r"\gg": "≫", r"\ldots": "…", r"\cdots": "⋯",
       r"\,": " ", r"\;": " ", r"\ ": " ", r"\%": "%"}
SUP = str.maketrans("0123456789-+=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺⁼⁽⁾ⁿ")
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def plain(t):
    u = t
    # macros, longest first
    for k in sorted(list(GREEK) + list(OPS), key=len, reverse=True):
        v = GREEK.get(k, OPS.get(k))
        u = re.sub(re.escape(k) + r"(?![a-zA-Z])", v, u)
    # superscripts with digits / minus only
    u = re.sub(r"\^\{([0-9\-+]+)\}", lambda m: m.group(1).translate(SUP), u)
    u = re.sub(r"\^([0-9])", lambda m: m.group(1).translate(SUP), u)
    # subscript infinity and digits
    u = re.sub(r"_\{?∞\}?", "∞", u)
    u = re.sub(r"_\{([0-9]+)\}", lambda m: m.group(1).translate(SUB), u)
    u = re.sub(r"_([0-9])", lambda m: m.group(1).translate(SUB), u)
    u = u.replace("-", "−") if re.search(r"(^|[\s=(])-[0-9]", u) else u
    if re.search(r"[\\{}^_*|`]", u) or re.search(r"<[A-Za-z/!]", u):
        return None
    return u


def main(path, write):
    s = open(path).read()
    lines = s.split("\n"); inb = False; out = []; n_conv = n_keep = 0; ex = []
    pat = re.compile(r"\$`([^`]*)`\$")
    for l in lines:
        if l.strip().startswith("```"):
            inb = not inb; out.append(l); continue
        if inb:
            out.append(l); continue
        def rep(m):
            nonlocal n_conv, n_keep
            p = plain(m.group(1))
            if p is None:
                n_keep += 1; return m.group(0)
            n_conv += 1
            if len(ex) < 60 and m.group(1) != p: ex.append((m.group(1), p))
            return p
        out.append(pat.sub(rep, l))
    print(f"converted {n_conv}, kept {n_keep}")
    for a, b in ex[::2]:
        print(f"   {a!r:45s} -> {b}")
    if write:
        open(path, "w").write("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1], len(sys.argv) > 2 and sys.argv[2] == "write")
