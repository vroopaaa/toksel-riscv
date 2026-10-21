#!/usr/bin/env python3
"""
Paper figure: (a) where each data structure of the bucket-based token selection lives
(RISC-V core with the BLK and BKT units, main memory), (b) the two passes of the
algorithm, with each custom instruction coloured by the unit it uses.

Writes toksel_hw.svg next to this script, then toksel_hw.pdf / .png via rsvg-convert.
Style follows the paper's pipeline figure: rounded boxes with a 2px black outline,
dotted tinted containers, bold coloured container titles.
"""

import os
import subprocess

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "toksel_hw")
W, H = 1040, 600

SANS = "Liberation Sans, Arial, Helvetica, sans-serif"
MONO = "Liberation Mono, Courier New, monospace"
INK = "#1d2430"
MUTED = "#4a5260"
# unit colours: (container fill, container stroke, title, box fill)
CORE = ("#f2f6fb", "#6f95c2", "#2c5d8f", "#ffffff")
BLK = ("#e6eef8", "#4a76a8", "#2c5d8f", "#d4e3f3")
BKT = ("#fcf1e2", "#c98a2b", "#8a5a12", "#f6dfbf")
MEM = ("#f5f5f4", "#8d8d8a", "#4f4f4c", "#e7e7e5")
STEP1 = ("#fdf7ee", "#d9a95a", "#8a5a12")
STEP2 = ("#eff4fa", "#6f95c2", "#2c5d8f")

parts = []


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=14, weight="normal", fill=INK, anchor="middle", family=SANS, style="normal"):
    """y is the visual centre of the line."""
    parts.append(f'<text x="{x}" y="{y + 0.35 * size:.1f}" font-family="{family}" font-size="{size}" '
                 f'font-weight="{weight}" font-style="{style}" fill="{fill}" text-anchor="{anchor}">{esc(s)}</text>')


def container(x, y, w, h, colours, title, title_x=None, anchor="middle", size=19):
    fill, stroke, tcol = colours[:3]
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="2.2" stroke-dasharray="2 4" stroke-linecap="round"/>')
    tx = title_x if title_x is not None else x + w / 2
    text(tx, y + 22, title, size=size, weight="bold", fill=tcol, anchor=anchor)


def box(x, y, w, h, fill, lines):
    """lines: list of (string, size, weight, family). Centred block of lines."""
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{INK}" stroke-width="2"/>')
    gaps = [ln[1] * 1.25 for ln in lines]
    total = sum(gaps)
    cy = y + h / 2 - total / 2
    for (s, size, weight, family), g in zip(lines, gaps):
        text(x + w / 2, cy + g / 2, s, size=size, weight=weight, family=family)
        cy += g


def pill(x, y, s, colours, size=14):
    """Instruction name in mono on its unit's colour; x = left edge, y = centre. Returns right edge."""
    w = 0.6 * size * len(s) + 10
    parts.append(f'<rect x="{x}" y="{y - size * 0.75:.1f}" width="{w:.1f}" height="{size * 1.5:.1f}" rx="5" '
                 f'fill="{colours[3]}" stroke="{colours[1]}" stroke-width="1.4"/>')
    text(x + 5, y, s, size=size, family=MONO, anchor="start", weight="bold")
    return x + w


def flow(x, y, w, h, runs):
    """White algorithm step box. runs: list of lines; a line is a list of (kind, string) with
    kind 'mono', 'sans', 'blk', 'bkt'. Lines are left-aligned with 12px padding."""
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="#ffffff" stroke="{INK}" stroke-width="2"/>')
    n = len(runs)
    for li, line in enumerate(runs):
        cy = y + h / 2 + (li - (n - 1) / 2) * 23
        cx = x + 12
        for kind, s in line:
            if kind in ("blk", "bkt"):
                cx = pill(cx, cy, s, BLK if kind == "blk" else BKT) + 6
            elif kind == "mono":
                text(cx, cy, s, size=14, family=MONO, anchor="start")
                cx += 0.6 * 14 * len(s)
            else:
                text(cx, cy, s, size=14, anchor="start", fill=MUTED)
                cx += 7 * len(s)   # sans runs only end lines, so an estimate is enough


def arrow(d, both=False, colour="#3a4150", width=2):
    start = ' marker-start="url(#ah)"' if both else ""
    parts.append(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{width}"{start} marker-end="url(#ah)"/>')


# ---------------- (a) hardware structures ----------------
container(16, 16, 504, 360, CORE, "RISC-V core (RV32IM)")
box(34, 132, 112, 138, "#ffffff", [("Register", 19, "bold", SANS), ("file", 19, "bold", SANS),
                                   ("x0–x31", 14, "normal", SANS)])

container(228, 54, 278, 150, BLK, "BLK unit", title_x=242, anchor="start", size=16)
box(240, 90, 126, 102, BLK[3], [("Valid map", 16, "bold", SANS), ("64 × 64-bit", 13, "normal", SANS),
                                ("registers", 13, "normal", SANS), ("1 = free", 13, "normal", SANS)])
box(376, 90, 118, 102, BLK[3], [("3×3 clear", 16, "bold", SANS), ("one-hot", 13, "normal", SANS),
                                ("decode", 13, "normal", SANS), ("+ widen", 13, "normal", SANS)])

container(228, 214, 278, 150, BKT, "BKT unit", title_x=242, anchor="start", size=16)
box(240, 250, 126, 102, BKT[3], [("Bucket", 16, "bold", SANS), ("heads", 16, "bold", SANS),
                                 ("1024 × 12-bit", 13, "normal", SANS), ("SRAM", 13, "normal", SANS)])
box(376, 250, 118, 102, BKT[3], [("Valid bits", 16, "bold", SANS), ("1024-bit", 13, "normal", SANS),
                                 ("register", 13, "normal", SANS), ("0 → reads −1", 13, "normal", SANS)])

# register file <-> units
arrow("M 148 168 L 226 168", both=True)
text(187, 136, "blk", size=13, family=MONO, weight="bold", fill=BLK[2])
text(187, 152, "blk.clr", size=13, family=MONO, weight="bold", fill=BLK[2])
arrow("M 148 238 L 226 238", both=True)
text(187, 258, "bkt.ins", size=13, family=MONO, weight="bold", fill=BKT[2])
text(187, 274, "bkt.head", size=13, family=MONO, weight="bold", fill=BKT[2])
text(187, 290, "bkt.clr", size=13, family=MONO, weight="bold", fill=BKT[2])

container(16, 392, 504, 192, MEM, "Main memory")
box(32, 430, 152, 138, MEM[3], [("Score map", 16, "bold", SANS), ("score[i]", 14, "normal", MONO),
                                ("4096 × 10-bit", 13, "normal", SANS)])
box(192, 430, 152, 138, MEM[3], [("Successors", 16, "bold", SANS), ("next[i]", 14, "normal", MONO),
                                 ("4096 × 16-bit", 13, "normal", SANS)])
box(352, 430, 152, 138, MEM[3], [("Selected", 16, "bold", SANS), ("tokens", 16, "bold", SANS),
                                 ("picks[n]", 14, "normal", MONO), ("K entries", 13, "normal", SANS)])

# register file <-> memory
arrow("M 90 272 L 90 390", both=True)
text(100, 330, "lhu / lh / sh", size=13, family=MONO, anchor="start", fill=MUTED)

# ---------------- (b) algorithm ----------------
container(538, 16, 486, 278, STEP1, "STEP 1 : Linking pass")
FX, FW = 572, 238
flow(FX, 52, FW, 44, [[("bkt", "bkt.clr"), ("sans", "empty all buckets")]])
flow(FX, 116, FW, 44, [[("mono", "for i = N−1 down to 0")]])
flow(FX, 180, FW, 72, [[("mono", "next[i] ←")], [("bkt", "bkt.ins(i, score[i])")]])
arrow(f"M {FX + FW / 2} 96 L {FX + FW / 2} 114")
arrow(f"M {FX + FW / 2} 160 L {FX + FW / 2} 178")
arrow(f"M {FX} 216 L {FX - 16} 216 L {FX - 16} 138 L {FX - 2} 138")   # loop

# example: insert cell 0 with score 5
EX = 832


def chain(x, y, head, nodes):
    parts.append(f'<rect x="{x}" y="{y - 12}" width="50" height="24" rx="5" fill="{BKT[3]}" stroke="{INK}" stroke-width="1.6"/>')
    text(x + 25, y, head, size=12, family=MONO)
    cx = x + 50
    for nd in nodes:
        arrow(f"M {cx + 1} {y} L {cx + 11} {y}", width=1.6)
        cx += 12
        if nd == "−1":
            text(cx + 2, y, "−1", size=12, family=MONO, anchor="start", fill=MUTED)
            cx += 20
        else:
            parts.append(f'<rect x="{cx}" y="{y - 12}" width="22" height="24" rx="5" fill="{MEM[3]}" stroke="{INK}" stroke-width="1.6"/>')
            text(cx + 11, y, nd, size=12, family=MONO)
            cx += 22


text(EX, 60, "Insert cell 0 (score 5)", size=14, weight="bold", anchor="start", fill=STEP1[2])
text(EX, 88, "before", size=12, anchor="start", fill=MUTED)
chain(EX, 110, "head[5]", ["2", "3", "−1"])
text(EX, 144, "after", size=12, anchor="start", fill=MUTED)
chain(EX, 166, "head[5]", ["0", "2", "3", "−1"])
text(EX, 202, "bkt.ins returns the old", size=12, anchor="start", fill=MUTED)
text(EX, 218, "head (2) into next[0] and", size=12, anchor="start", fill=MUTED)
text(EX, 234, "makes 0 the new head.", size=12, anchor="start", fill=MUTED)
text(EX, 258, "Lists end up in index order.", size=12, anchor="start", fill=MUTED, style="italic")

container(538, 306, 486, 278, STEP2, "STEP 2 : Walk")
flow(FX, 340, FW, 40, [[("blk", "blk.clr"), ("sans", "all cells free")]])
flow(FX, 396, FW, 40, [[("mono", "for v = 1023 down to 0")]])
flow(FX, 452, FW, 56, [[("mono", "c = "), ("bkt", "bkt.head(v)")], [("mono", "then c = next[c] to −1")]])
flow(FX, 524, FW, 48, [[("mono", "if "), ("blk", "blk(c)"), ("mono", ": pick c")]])
arrow(f"M {FX + FW / 2} 380 L {FX + FW / 2} 394")
arrow(f"M {FX + FW / 2} 436 L {FX + FW / 2} 450")
arrow(f"M {FX + FW / 2} 508 L {FX + FW / 2} 522")
arrow(f"M {FX} 548 L {FX - 10} 548 L {FX - 10} 480 L {FX - 2} 480")   # next cell in bucket
arrow(f"M {FX - 10} 548 L {FX - 22} 548 L {FX - 22} 416 L {FX - 2} 416")  # next bucket

# example: 5x5 grid around a pick
GX, GY, CS = 846, 342, 24
picked, blocked = (2, 1), {(r, c) for r in (1, 2, 3) for c in (0, 1, 2)}
for r in range(5):
    for c in range(5):
        fill = "#e9b25c" if (r, c) == picked else ("#c4c8cf" if (r, c) in blocked else "#ffffff")
        parts.append(f'<rect x="{GX + c * (CS + 3)}" y="{GY + r * (CS + 3)}" width="{CS}" height="{CS}" rx="4" '
                     f'fill="{fill}" stroke="{INK}" stroke-width="1.4"/>')
text(GX + 1 * (CS + 3) + CS / 2, GY + 2 * (CS + 3) + CS / 2, "c", size=13, family=MONO, weight="bold")
text(EX, 498, "blk(c) = 1: c is picked and", size=12, anchor="start", fill=MUTED)
text(EX, 514, "its 3×3 neighbourhood is", size=12, anchor="start", fill=MUTED)
text(EX, 530, "blocked in one cycle.", size=12, anchor="start", fill=MUTED)
text(EX, 554, "blk(c) = 0: c already", size=12, anchor="start", fill=MUTED)
text(EX, 570, "blocked, skip it.", size=12, anchor="start", fill=MUTED)

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
       '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5.5" markerHeight="5.5" '
       'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#3a4150"/></marker></defs>'
       f'<rect width="{W}" height="{H}" fill="#ffffff"/>' + "".join(parts) + "</svg>")
with open(OUT + ".svg", "w") as f:
    f.write(svg)
subprocess.run(["rsvg-convert", "-f", "pdf", "-o", OUT + ".pdf", OUT + ".svg"], check=True)
subprocess.run(["rsvg-convert", "-f", "png", "-z", "2", "-o", OUT + ".png", OUT + ".svg"], check=True)
print("wrote", OUT + ".{svg,pdf,png}")
