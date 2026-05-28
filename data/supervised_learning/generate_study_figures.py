from __future__ import annotations

import math
from pathlib import Path


OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)


COLORS = {
    "ink": "#1f2937",
    "muted": "#64748b",
    "grid": "#e5e7eb",
    "blue": "#2563eb",
    "blue_light": "#dbeafe",
    "red": "#dc2626",
    "red_light": "#fee2e2",
    "green": "#16a34a",
    "green_light": "#dcfce7",
    "orange": "#ea580c",
    "orange_light": "#ffedd5",
    "purple": "#7c3aed",
    "purple_light": "#ede9fe",
    "teal": "#0f766e",
    "teal_light": "#ccfbf1",
    "gray": "#f8fafc",
    "white": "#ffffff",
}


def esc(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def svg(width: int, height: int, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
<style>
text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: {COLORS["ink"]}; }}
.title {{ font-size: 24px; font-weight: 700; }}
.subtitle {{ font-size: 14px; fill: {COLORS["muted"]}; }}
.label {{ font-size: 13px; fill: {COLORS["muted"]}; }}
.small {{ font-size: 12px; fill: {COLORS["muted"]}; }}
.node {{ font-size: 13px; font-weight: 650; }}
</style>
<rect x="0" y="0" width="{width}" height="{height}" fill="{COLORS["white"]}"/>
{body}
</svg>
"""


def write(name: str, width: int, height: int, body: str) -> None:
    (OUT / name).write_text(svg(width, height, body), encoding="utf-8")


def text(x: float, y: float, value: str, cls: str = "label", anchor: str = "start") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{esc(value)}</text>'


def line(x1: float, y1: float, x2: float, y2: float, stroke: str, width: float = 2, dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "#cbd5e1", rx: float = 8, width: float = 1.5) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'


def circle(x: float, y: float, r: float, fill: str, stroke: str = COLORS["white"], width: float = 1.5) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'


def polyline(points: list[tuple[float, float]], stroke: str, width: float = 3, fill: str = "none", dash: str | None = None) -> str:
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"{dash_attr}/>'


def arrow_marker() -> str:
    return """<defs>
<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
<path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/>
</marker>
</defs>"""


def arrow(x1: float, y1: float, x2: float, y2: float) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{COLORS["muted"]}" stroke-width="2" marker-end="url(#arrow)"/>'


def sigmoid_curve() -> None:
    w, h = 720, 420
    left, top, plot_w, plot_h = 80, 58, 570, 295

    def mx(z: float) -> float:
        return left + (z + 6) / 12 * plot_w

    def my(p: float) -> float:
        return top + (1 - p) * plot_h

    body = [text(36, 34, "Sigmoid: from score to probability", "title")]
    body.append(text(36, 55, "Any linear score z is squeezed into the probability range [0, 1].", "subtitle"))
    for p in [0, 0.25, 0.5, 0.75, 1]:
        body.append(line(left, my(p), left + plot_w, my(p), COLORS["grid"], 1))
        body.append(text(left - 12, my(p) + 4, f"{p:.2g}", "small", "end"))
    for z in [-6, -3, 0, 3, 6]:
        body.append(line(mx(z), top, mx(z), top + plot_h, COLORS["grid"], 1))
        body.append(text(mx(z), top + plot_h + 24, str(z), "small", "middle"))
    body.append(line(left, top + plot_h, left + plot_w, top + plot_h, COLORS["ink"], 2))
    body.append(line(left, top, left, top + plot_h, COLORS["ink"], 2))
    body.append(line(left, my(0.5), left + plot_w, my(0.5), COLORS["orange"], 2, "6 6"))
    body.append(line(mx(0), top, mx(0), top + plot_h, COLORS["orange"], 2, "6 6"))
    pts = []
    for i in range(241):
        z = -6 + 12 * i / 240
        p = 1 / (1 + math.exp(-z))
        pts.append((mx(z), my(p)))
    body.append(polyline(pts, COLORS["blue"], 4))
    body.append(circle(mx(0), my(0.5), 5, COLORS["orange"]))
    body.append(text(mx(0) + 12, my(0.5) - 12, "z = 0, P = 0.5", "label"))
    body.append(text(left + plot_w / 2, h - 25, "linear score z", "label", "middle"))
    body.append(text(25, top + plot_h / 2, "probability", "label", "middle"))
    write("01_sigmoid_curve.svg", w, h, "\n".join(body))


def logistic_decision_boundary() -> None:
    w, h = 720, 430
    left, top, plot_w, plot_h = 70, 70, 580, 285

    def mx(x: float) -> float:
        return left + x / 10 * plot_w

    def my(y: float) -> float:
        return top + (10 - y) / 10 * plot_h

    body = [text(36, 36, "Logistic regression decision boundary", "title")]
    body.append(text(36, 57, "The model predicts class 1 on one side of the line and class 0 on the other.", "subtitle"))
    body.append(f'<polygon points="{mx(0):.1f},{my(1):.1f} {mx(9):.1f},{my(10):.1f} {mx(10):.1f},{my(10):.1f} {mx(10):.1f},{my(0):.1f} {mx(0):.1f},{my(0):.1f}" fill="{COLORS["red_light"]}" opacity="0.8"/>')
    body.append(f'<polygon points="{mx(0):.1f},{my(10):.1f} {mx(0):.1f},{my(1):.1f} {mx(9):.1f},{my(10):.1f}" fill="{COLORS["blue_light"]}" opacity="0.8"/>')
    for v in range(0, 11, 2):
        body.append(line(mx(v), top, mx(v), top + plot_h, COLORS["grid"], 1))
        body.append(line(left, my(v), left + plot_w, my(v), COLORS["grid"], 1))
    body.append(line(left, top + plot_h, left + plot_w, top + plot_h, COLORS["ink"], 2))
    body.append(line(left, top, left, top + plot_h, COLORS["ink"], 2))
    body.append(line(mx(0), my(1), mx(9), my(10), COLORS["orange"], 4))
    body.append(text(mx(4.9), my(5.9) - 12, "P = 0.5 boundary", "label", "middle"))
    class0 = [(1.0, 8.6), (1.7, 7.2), (2.6, 7.8), (3.2, 8.8), (4.4, 9.0), (2.2, 5.8), (4.9, 7.5)]
    class1 = [(6.0, 2.8), (7.1, 3.1), (8.0, 1.7), (8.6, 4.2), (6.9, 5.1), (5.3, 3.9), (9.1, 2.6)]
    for x, y in class0:
        body.append(circle(mx(x), my(y), 7, COLORS["blue"]))
    for x, y in class1:
        body.append(circle(mx(x), my(y), 7, COLORS["red"]))
    body.append(text(112, 95, "Predicted 0", "label"))
    body.append(text(545, 330, "Predicted 1", "label"))
    body.append(text(left + plot_w / 2, h - 32, "feature x1", "label", "middle"))
    body.append(text(22, top + plot_h / 2, "feature x2", "label", "middle"))
    write("02_logistic_decision_boundary.svg", w, h, "\n".join(body))


def confusion_matrix() -> None:
    w, h = 720, 430
    body = [text(36, 38, "Confusion matrix", "title")]
    body.append(text(36, 60, "The four outcomes of a binary classifier.", "subtitle"))
    x0, y0, cw, ch = 190, 115, 210, 100
    body.append(text(x0 + cw, 95, "Predicted class", "label", "middle"))
    body.append(text(92, y0 + ch, "Actual class", "label", "middle"))
    body.append(text(x0 + cw / 2, y0 - 15, "Positive", "label", "middle"))
    body.append(text(x0 + 1.5 * cw, y0 - 15, "Negative", "label", "middle"))
    body.append(text(x0 - 20, y0 + ch / 2, "Positive", "label", "end"))
    body.append(text(x0 - 20, y0 + 1.5 * ch, "Negative", "label", "end"))
    cells = [
        (0, 0, "TP", "True Positive", COLORS["green_light"], COLORS["green"]),
        (1, 0, "FN", "False Negative", COLORS["orange_light"], COLORS["orange"]),
        (0, 1, "FP", "False Positive", COLORS["red_light"], COLORS["red"]),
        (1, 1, "TN", "True Negative", COLORS["green_light"], COLORS["green"]),
    ]
    for col, row, big, small, fill, stroke in cells:
        x, y = x0 + col * cw, y0 + row * ch
        body.append(rect(x, y, cw, ch, fill, stroke, 10, 2))
        body.append(text(x + cw / 2, y + 43, big, "title", "middle"))
        body.append(text(x + cw / 2, y + 69, small, "label", "middle"))
    body.append(text(x0 + cw + 80, y0 + 2 * ch + 42, "FP = Type I error", "small", "middle"))
    body.append(text(x0 + cw + 80, y0 + 2 * ch + 62, "FN = Type II error", "small", "middle"))
    write("03_confusion_matrix.svg", w, h, "\n".join(body))


def metric_denominators() -> None:
    w, h = 760, 470
    body = [text(36, 38, "Precision, recall, and specificity look in different directions", "title")]
    body.append(text(36, 60, "The metric changes depending on which group is used as the denominator.", "subtitle"))
    x0, y0, cw, ch = 185, 110, 185, 88
    body.append(text(x0 + cw / 2, 94, "Predicted +", "label", "middle"))
    body.append(text(x0 + 1.5 * cw, 94, "Predicted -", "label", "middle"))
    body.append(text(x0 - 18, y0 + ch / 2, "Actual +", "label", "end"))
    body.append(text(x0 - 18, y0 + 1.5 * ch, "Actual -", "label", "end"))
    for col, row, big, fill in [
        (0, 0, "TP", COLORS["green_light"]),
        (1, 0, "FN", COLORS["orange_light"]),
        (0, 1, "FP", COLORS["red_light"]),
        (1, 1, "TN", COLORS["green_light"]),
    ]:
        x, y = x0 + col * cw, y0 + row * ch
        body.append(rect(x, y, cw, ch, fill, "#cbd5e1", 8, 1.5))
        body.append(text(x + cw / 2, y + 53, big, "title", "middle"))
    # Highlights.
    body.append(rect(x0 - 6, y0 - 6, cw + 12, 2 * ch + 12, "none", COLORS["blue"], 8, 4))
    body.append(text(x0 + cw / 2, y0 + 2 * ch + 36, "Precision", "node", "middle"))
    body.append(text(x0 + cw / 2, y0 + 2 * ch + 56, "TP / predicted positives", "small", "middle"))
    body.append(rect(x0 + 2 * cw + 24, y0 - 6, 112, ch + 12, "none", COLORS["purple"], 8, 4))
    body.append(rect(x0 - 6, y0 - 6, 2 * cw + 12, ch + 12, "none", COLORS["purple"], 8, 4))
    body.append(text(x0 + 2 * cw + 112, y0 + 34, "Recall", "node", "middle"))
    body.append(text(x0 + 2 * cw + 112, y0 + 56, "TP / actual positives", "small", "middle"))
    body.append(rect(x0 - 6, y0 + ch - 6, 2 * cw + 12, ch + 12, "none", COLORS["teal"], 8, 4))
    body.append(text(x0 + cw, y0 + 2 * ch + 98, "Specificity = TN / actual negatives", "node", "middle"))
    body.append(text(x0 + cw, y0 + 2 * ch + 120, "Precision uses predictions; recall and specificity use reality.", "small", "middle"))
    write("04_metric_denominators.svg", w, h, "\n".join(body))


def threshold_tradeoff() -> None:
    w, h = 720, 420
    left, top, plot_w, plot_h = 78, 70, 570, 275

    def mx(t: float) -> float:
        return left + (t - 0.1) / 0.8 * plot_w

    def my(v: float) -> float:
        return top + (1 - v) * plot_h

    body = [text(36, 36, "Threshold trade-off", "title")]
    body.append(text(36, 58, "Raising the threshold usually increases precision and decreases recall.", "subtitle"))
    for v in [0, 0.25, 0.5, 0.75, 1]:
        body.append(line(left, my(v), left + plot_w, my(v), COLORS["grid"], 1))
        body.append(text(left - 10, my(v) + 4, f"{v:.2g}", "small", "end"))
    for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
        body.append(line(mx(t), top, mx(t), top + plot_h, COLORS["grid"], 1))
        body.append(text(mx(t), top + plot_h + 24, f"{t:.1f}", "small", "middle"))
    body.append(line(left, top + plot_h, left + plot_w, top + plot_h, COLORS["ink"], 2))
    body.append(line(left, top, left, top + plot_h, COLORS["ink"], 2))
    precision = []
    recall = []
    for i in range(81):
        t = 0.1 + 0.8 * i / 80
        p = 0.38 + 0.57 / (1 + math.exp(-7 * (t - 0.52)))
        r = 0.95 - 0.78 / (1 + math.exp(-8 * (t - 0.50)))
        precision.append((mx(t), my(p)))
        recall.append((mx(t), my(r)))
    body.append(polyline(precision, COLORS["blue"], 4))
    body.append(polyline(recall, COLORS["red"], 4))
    body.append(text(505, 140, "Precision", "node"))
    body.append(text(180, 130, "Recall", "node"))
    body.append(text(left + plot_w / 2, h - 30, "decision threshold", "label", "middle"))
    write("05_threshold_tradeoff.svg", w, h, "\n".join(body))


def roc_curve() -> None:
    w, h = 720, 420
    left, top, plot_w, plot_h = 82, 70, 560, 275

    def mx(x: float) -> float:
        return left + x * plot_w

    def my(y: float) -> float:
        return top + (1 - y) * plot_h

    body = [text(36, 36, "ROC curve", "title")]
    body.append(text(36, 58, "Shows the recall gained for each amount of false alarm risk.", "subtitle"))
    for v in [0, 0.25, 0.5, 0.75, 1]:
        body.append(line(left, my(v), left + plot_w, my(v), COLORS["grid"], 1))
        body.append(line(mx(v), top, mx(v), top + plot_h, COLORS["grid"], 1))
        body.append(text(left - 10, my(v) + 4, f"{v:.2g}", "small", "end"))
        body.append(text(mx(v), top + plot_h + 24, f"{v:.2g}", "small", "middle"))
    body.append(line(left, top + plot_h, left + plot_w, top + plot_h, COLORS["ink"], 2))
    body.append(line(left, top, left, top + plot_h, COLORS["ink"], 2))
    body.append(line(mx(0), my(0), mx(1), my(1), COLORS["muted"], 2, "7 7"))
    pts = [(0, 0), (0.02, 0.22), (0.05, 0.48), (0.10, 0.70), (0.18, 0.84), (0.32, 0.93), (0.55, 0.98), (1, 1)]
    body.append(polyline([(mx(x), my(y)) for x, y in pts], COLORS["blue"], 4))
    body.append(text(mx(0.58), my(0.47), "Random baseline", "small"))
    body.append(text(mx(0.23), my(0.88), "Better curve bends top-left", "node"))
    body.append(text(left + plot_w / 2, h - 30, "False Positive Rate = 1 - specificity", "label", "middle"))
    body.append(text(25, top + plot_h / 2, "True Positive Rate = recall", "label", "middle"))
    write("06_roc_curve.svg", w, h, "\n".join(body))


def pr_curve() -> None:
    w, h = 720, 420
    left, top, plot_w, plot_h = 82, 70, 560, 275

    def mx(x: float) -> float:
        return left + x * plot_w

    def my(y: float) -> float:
        return top + (1 - y) * plot_h

    body = [text(36, 36, "Precision-recall curve", "title")]
    body.append(text(36, 58, "Best for rare positives: can precision stay high as recall grows?", "subtitle"))
    for v in [0, 0.25, 0.5, 0.75, 1]:
        body.append(line(left, my(v), left + plot_w, my(v), COLORS["grid"], 1))
        body.append(line(mx(v), top, mx(v), top + plot_h, COLORS["grid"], 1))
        body.append(text(left - 10, my(v) + 4, f"{v:.2g}", "small", "end"))
        body.append(text(mx(v), top + plot_h + 24, f"{v:.2g}", "small", "middle"))
    body.append(line(left, top + plot_h, left + plot_w, top + plot_h, COLORS["ink"], 2))
    body.append(line(left, top, left, top + plot_h, COLORS["ink"], 2))
    body.append(line(mx(0), my(0.1), mx(1), my(0.1), COLORS["muted"], 2, "7 7"))
    pts = [(0.02, 1.0), (0.15, 0.96), (0.35, 0.90), (0.55, 0.76), (0.68, 0.55), (0.82, 0.30), (1.0, 0.12)]
    body.append(polyline([(mx(x), my(y)) for x, y in pts], COLORS["purple"], 4))
    body.append(text(mx(0.60), my(0.12) - 9, "Rare-class baseline", "small"))
    body.append(text(mx(0.62), my(0.62), "The cliff", "node"))
    body.append(text(left + plot_w / 2, h - 30, "Recall", "label", "middle"))
    body.append(text(25, top + plot_h / 2, "Precision", "label", "middle"))
    write("07_pr_curve.svg", w, h, "\n".join(body))


def decision_tree_structure() -> None:
    w, h = 760, 460
    body = [arrow_marker(), text(36, 36, "Decision tree structure", "title")]
    body.append(text(36, 58, "A prediction is made by following rules from the root to a leaf.", "subtitle"))
    nodes = {
        "root": (380, 100, 190, 58, "Age <= 30?", COLORS["blue_light"], COLORS["blue"]),
        "left": (240, 205, 190, 58, "Income > 50k?", COLORS["purple_light"], COLORS["purple"]),
        "right": (520, 205, 190, 58, "Class = 0", COLORS["green_light"], COLORS["green"]),
        "ll": (135, 325, 150, 58, "Class = 1", COLORS["green_light"], COLORS["green"]),
        "lr": (345, 325, 150, 58, "Class = 0", COLORS["green_light"], COLORS["green"]),
    }
    body.append(arrow(380, 129, 240, 205))
    body.append(arrow(380, 129, 520, 205))
    body.append(arrow(240, 234, 135, 325))
    body.append(arrow(240, 234, 345, 325))
    body.append(text(292, 176, "yes", "small", "middle"))
    body.append(text(468, 176, "no", "small", "middle"))
    body.append(text(184, 296, "yes", "small", "middle"))
    body.append(text(300, 296, "no", "small", "middle"))
    for x, y, ww, hh, label, fill, stroke in nodes.values():
        body.append(rect(x - ww / 2, y - hh / 2, ww, hh, fill, stroke, 10, 2))
        body.append(text(x, y + 5, label, "node", "middle"))
    body.append(text(380, 155, "root node", "small", "middle"))
    body.append(text(240, 260, "internal node", "small", "middle"))
    body.append(text(520, 260, "leaf node", "small", "middle"))
    write("08_decision_tree_structure.svg", w, h, "\n".join(body))


def gini_split() -> None:
    w, h = 780, 450
    body = [text(36, 36, "Gini split: bad vs good", "title")]
    body.append(text(36, 58, "A good split creates purer child nodes, so the weighted Gini is lower.", "subtitle"))

    def draw_panel(x: int, title: str, subtitle: str, left_counts: tuple[int, int], right_counts: tuple[int, int], gini: str) -> None:
        body.append(rect(x, 95, 320, 285, COLORS["gray"], "#cbd5e1", 12, 1.5))
        body.append(text(x + 160, 125, title, "node", "middle"))
        body.append(text(x + 160, 146, subtitle, "small", "middle"))
        body.append(text(x + 88, 178, "Left child", "label", "middle"))
        body.append(text(x + 232, 178, "Right child", "label", "middle"))
        body.append(rect(x + 35, 195, 110, 120, COLORS["white"], "#cbd5e1", 8, 1.5))
        body.append(rect(x + 175, 195, 110, 120, COLORS["white"], "#cbd5e1", 8, 1.5))

        def dots(cx: int, cy: int, counts: tuple[int, int]) -> None:
            coords = [(cx - 24, cy - 20), (cx, cy - 20), (cx + 24, cy - 20), (cx - 12, cy + 8), (cx + 12, cy + 8), (cx - 24, cy + 36), (cx, cy + 36), (cx + 24, cy + 36)]
            colors = [COLORS["blue"]] * counts[0] + [COLORS["red"]] * counts[1]
            for (dx, dy), col in zip(coords, colors):
                body.append(circle(dx, dy, 7, col))

        dots(x + 90, 235, left_counts)
        dots(x + 230, 235, right_counts)
        body.append(text(x + 160, 350, f"Weighted Gini = {gini}", "node", "middle"))

    draw_panel(55, "Poor split", "Both children stay mixed", (3, 2), (2, 3), "0.48")
    draw_panel(405, "Better split", "One child is pure", (3, 0), (2, 5), "0.286")
    write("09_gini_split_good_bad.svg", w, h, "\n".join(body))


def tree_axis_boundary() -> None:
    w, h = 720, 430
    left, top, plot_w, plot_h = 75, 72, 570, 285

    def mx(x: float) -> float:
        return left + x / 10 * plot_w

    def my(y: float) -> float:
        return top + (10 - y) / 10 * plot_h

    body = [text(36, 36, "Decision tree decision surface", "title")]
    body.append(text(36, 58, "Trees split one feature at a time, creating box-like regions.", "subtitle"))
    body.append(rect(left, top, plot_w, plot_h, COLORS["blue_light"], "#cbd5e1", 0, 1))
    body.append(f'<path d="M {mx(4):.1f} {my(10):.1f} L {mx(4):.1f} {my(6):.1f} L {mx(6.2):.1f} {my(6):.1f} L {mx(6.2):.1f} {my(3.5):.1f} L {mx(8):.1f} {my(3.5):.1f} L {mx(8):.1f} {my(0):.1f} L {mx(10):.1f} {my(0):.1f} L {mx(10):.1f} {my(10):.1f} Z" fill="{COLORS["red_light"]}" opacity="0.82"/>')
    for v in range(0, 11, 2):
        body.append(line(mx(v), top, mx(v), top + plot_h, COLORS["grid"], 1))
        body.append(line(left, my(v), left + plot_w, my(v), COLORS["grid"], 1))
    body.append(line(left, top + plot_h, left + plot_w, top + plot_h, COLORS["ink"], 2))
    body.append(line(left, top, left, top + plot_h, COLORS["ink"], 2))
    boundary = [(4, 10), (4, 6), (6.2, 6), (6.2, 3.5), (8, 3.5), (8, 0)]
    body.append(polyline([(mx(x), my(y)) for x, y in boundary], COLORS["orange"], 5))
    class0 = [(1, 8), (2, 6.6), (3.2, 7.2), (2.4, 4.7), (5, 7.5), (5.5, 6.8)]
    class1 = [(6.8, 5.0), (7.6, 4.1), (8.4, 2.8), (9.0, 1.7), (7.0, 2.4), (8.8, 5.2)]
    for x, y in class0:
        body.append(circle(mx(x), my(y), 7, COLORS["blue"]))
    for x, y in class1:
        body.append(circle(mx(x), my(y), 7, COLORS["red"]))
    body.append(text(left + plot_w / 2, h - 30, "feature x1", "label", "middle"))
    body.append(text(24, top + plot_h / 2, "feature x2", "label", "middle"))
    write("10_tree_axis_aligned_boundary.svg", w, h, "\n".join(body))


def bagging_diagram() -> None:
    w, h = 850, 460
    body = [arrow_marker(), text(36, 36, "Random forest = bagging + random features + voting", "title")]
    body.append(text(36, 58, "Bootstrap samples train diverse trees; their predictions are aggregated.", "subtitle"))
    body.append(rect(35, 150, 120, 145, COLORS["gray"], "#cbd5e1", 10, 1.5))
    body.append(text(95, 135, "Original data", "node", "middle"))
    for i in range(24):
        x = 58 + (i % 4) * 23
        y = 170 + (i // 4) * 19
        body.append(circle(x, y, 5, COLORS["blue"] if i % 3 else COLORS["red"]))
    body.append(arrow(160, 220, 245, 135))
    body.append(arrow(160, 220, 245, 220))
    body.append(arrow(160, 220, 245, 305))
    for idx, y in enumerate([95, 180, 265], start=1):
        body.append(rect(250, y, 155, 65, COLORS["blue_light"], COLORS["blue"], 9, 1.5))
        body.append(text(327, y + 29, f"Bootstrap sample {idx}", "node", "middle"))
        body.append(text(327, y + 49, "sample with replacement", "small", "middle"))
        body.append(arrow(410, y + 32, 500, y + 32))
        # Small tree.
        cx = 555
        body.append(circle(cx, y + 20, 12, COLORS["purple"]))
        body.append(line(cx, y + 32, cx - 30, y + 60, COLORS["muted"], 2))
        body.append(line(cx, y + 32, cx + 30, y + 60, COLORS["muted"], 2))
        body.append(circle(cx - 30, y + 60, 10, COLORS["green"]))
        body.append(circle(cx + 30, y + 60, 10, COLORS["green"]))
        body.append(text(cx, y + 89, f"Tree {idx}", "small", "middle"))
        body.append(arrow(610, y + 32, 695, 220))
    body.append(rect(700, 175, 115, 90, COLORS["green_light"], COLORS["green"], 10, 2))
    body.append(text(757, 210, "Majority", "node", "middle"))
    body.append(text(757, 232, "vote", "node", "middle"))
    body.append(text(757, 290, "classification", "small", "middle"))
    body.append(text(757, 310, "or average for regression", "small", "middle"))
    write("11_bagging_random_forest.svg", w, h, "\n".join(body))


def random_feature_selection() -> None:
    w, h = 820, 420
    body = [arrow_marker(), text(36, 36, "Random feature selection at each split", "title")]
    body.append(text(36, 58, "Each node sees only a subset of features, which decorrelates trees.", "subtitle"))
    features = ["age", "income", "city", "device", "visits", "price", "gender", "score"]
    x0, y0 = 55, 110
    body.append(text(55, 95, "All features", "node"))
    for i, f in enumerate(features):
        body.append(rect(x0 + i * 86, y0, 72, 38, COLORS["gray"], "#cbd5e1", 7, 1.2))
        body.append(text(x0 + i * 86 + 36, y0 + 24, f, "small", "middle"))
    body.append(arrow(385, 158, 215, 225))
    body.append(arrow(385, 158, 405, 225))
    body.append(arrow(385, 158, 595, 225))
    subsets = [
        (90, 225, ["age", "city", "score"], "Node A chooses best split"),
        (310, 225, ["income", "visits", "price"], "Node B sees another subset"),
        (530, 225, ["device", "price", "gender"], "Node C sees another subset"),
    ]
    fills = [COLORS["blue_light"], COLORS["purple_light"], COLORS["teal_light"]]
    strokes = [COLORS["blue"], COLORS["purple"], COLORS["teal"]]
    for idx, (x, y, subset, caption) in enumerate(subsets):
        body.append(rect(x, y, 180, 115, fills[idx], strokes[idx], 10, 2))
        body.append(text(x + 90, y + 24, f"Random subset {idx + 1}", "node", "middle"))
        for j, f in enumerate(subset):
            body.append(rect(x + 20 + j * 52, y + 45, 45, 32, COLORS["white"], "#cbd5e1", 6, 1.1))
            body.append(text(x + 42 + j * 52, y + 66, f, "small", "middle"))
        body.append(text(x + 90, y + 97, caption, "small", "middle"))
    body.append(text(410, 385, "Less correlation between trees = better averaging", "node", "middle"))
    write("12_random_feature_selection.svg", w, h, "\n".join(body))


def variance_reduction() -> None:
    w, h = 760, 430
    left, top, plot_w, plot_h = 75, 72, 590, 285

    def mx(x: float) -> float:
        return left + x / 10 * plot_w

    def my(y: float) -> float:
        return top + (10 - y) / 10 * plot_h

    body = [text(36, 36, "Averaging reduces variance", "title")]
    body.append(text(36, 58, "Individual trees can be noisy; the forest average is more stable.", "subtitle"))
    for v in range(0, 11, 2):
        body.append(line(mx(v), top, mx(v), top + plot_h, COLORS["grid"], 1))
        body.append(line(left, my(v), left + plot_w, my(v), COLORS["grid"], 1))
    body.append(line(left, top + plot_h, left + plot_w, top + plot_h, COLORS["ink"], 2))
    body.append(line(left, top, left, top + plot_h, COLORS["ink"], 2))
    tree_colors = ["#93c5fd", "#fca5a5", "#c4b5fd", "#5eead4", "#fdba74"]
    all_ys: list[list[float]] = []
    xs = [i * 0.5 for i in range(21)]
    for t in range(5):
        ys = []
        pts = []
        for x in xs:
            base = 2.4 + 0.55 * x
            noise = math.sin(1.4 * x + t * 1.2) * (1.25 - t * 0.08)
            y = max(0.4, min(9.5, base + noise))
            ys.append(y)
            pts.append((mx(x), my(y)))
        all_ys.append(ys)
        body.append(polyline(pts, tree_colors[t], 2.4))
    avg = []
    for i, x in enumerate(xs):
        avg_y = sum(series[i] for series in all_ys) / len(all_ys)
        avg.append((mx(x), my(avg_y)))
    body.append(polyline(avg, COLORS["blue"], 5))
    body.append(text(475, 135, "Single trees: high variance", "small"))
    body.append(text(470, 215, "Forest average: smoother", "node"))
    body.append(text(left + plot_w / 2, h - 30, "feature value", "label", "middle"))
    body.append(text(26, top + plot_h / 2, "prediction", "label", "middle"))
    write("13_variance_reduction.svg", w, h, "\n".join(body))


def main() -> None:
    sigmoid_curve()
    logistic_decision_boundary()
    confusion_matrix()
    metric_denominators()
    threshold_tradeoff()
    roc_curve()
    pr_curve()
    decision_tree_structure()
    gini_split()
    tree_axis_boundary()
    bagging_diagram()
    random_feature_selection()
    variance_reduction()


if __name__ == "__main__":
    main()
