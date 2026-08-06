"""Pure-Pillow PNG chart renderer for the FiVI PBS+DBBU gap history (no
matplotlib available in this environment). Draws two stacked line charts:
gap (log-scale y, since it spans ~1.6 -> ~0.08) and vl/vu bounds."""
import json
import math
import os

from PIL import Image, ImageDraw, ImageFont

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
with open(os.path.join(RES, 'fivi_pbs_dbbu_history.json')) as f:
    DATA = json.load(f)

W, H = 1000, 900
PAD = {'l': 70, 'r': 30, 't': 50, 'b': 50}
BG = (255, 255, 255)
INK = (11, 11, 11)
MUTED = (137, 135, 129)
GRID = (225, 224, 217)
BLUE = (42, 120, 214)
RED = (227, 73, 72)
GREEN = (27, 175, 122)

try:
    KFONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
    FONT = ImageFont.truetype(KFONT, 15)
    FONT_B = ImageFont.truetype(KFONT, 17)
    FONT_S = ImageFont.truetype(KFONT, 12)
except Exception:
    FONT = FONT_B = FONT_S = ImageFont.load_default()


def draw_line_chart(draw, x0, y0, w, h, series, title, y_is_log=False, y_ticks=None, legend_at='bottom'):
    xs_all = [d['iter'] for d in DATA]
    xmin, xmax = min(xs_all), max(xs_all)
    if y_is_log:
        all_y = [d[s['key']] for s in series for d in DATA if d[s['key']] > 0]
        ymin_l, ymax_l = math.log10(min(all_y)), math.log10(max(all_y))
        pad = (ymax_l - ymin_l) * 0.08 or 0.1
        ymin_l -= pad
        ymax_l += pad
        def Y(v):
            return y0 + h - (math.log10(v) - ymin_l) / (ymax_l - ymin_l) * h
    else:
        all_y = [d[s['key']] for s in series for d in DATA]
        ymin, ymax = min(all_y), max(all_y)
        pad = (ymax - ymin) * 0.1 or 0.1
        ymin -= pad
        ymax += pad
        def Y(v):
            return y0 + h - (v - ymin) / (ymax - ymin) * h

    def X(v):
        return x0 + (v - xmin) / (xmax - xmin) * w

    draw.text((x0, y0 - 26), title, fill=INK, font=FONT_B)

    if y_ticks:
        for v in y_ticks:
            y = Y(v)
            draw.line([(x0, y), (x0 + w, y)], fill=GRID, width=1)
            draw.text((x0 - 8, y - 7), f"{v:g}", fill=MUTED, font=FONT_S, anchor="ra")
    draw.line([(x0, y0 + h), (x0 + w, y0 + h)], fill=(195, 194, 183), width=1)

    xtick_step = max(1, (xmax - xmin) // 10)
    xt = xmin
    while xt <= xmax:
        x = X(xt)
        draw.line([(x, y0 + h), (x, y0 + h + 5)], fill=(195, 194, 183), width=1)
        draw.text((x, y0 + h + 8), str(xt), fill=MUTED, font=FONT_S, anchor="ma")
        xt += xtick_step

    for s in series:
        pts = [(X(d['iter']), Y(d[s['key']])) for d in DATA]
        draw.line(pts, fill=s['color'], width=2)
        for (px, py) in pts:
            draw.ellipse([px - 2.5, py - 2.5, px + 2.5, py + 2.5], fill=s['color'])

    lx = x0 + w - 220
    ly = (y0 + 6) if legend_at == 'top' else (y0 + h - len(series) * 24 - 6)
    for i, s in enumerate(series):
        yy = ly + i * 24
        draw.rectangle([lx, yy, lx + 12, yy + 12], fill=(255, 255, 255))
        draw.rectangle([lx, yy, lx + 12, yy + 12], fill=s['color'])
        draw.text((lx + 20, yy - 2), s['label'], fill=INK, font=FONT_S)


img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)

d.text((30, 16), "FiVI + PBS(Algorithm 4) + DBBU(Algorithm 5, simplified) gap 수렴 추이",
       fill=INK, font=FONT_B)
d.text((30, 38), f"age 40-85 (h=46), {len(DATA)} iterations, dbbu_interval=5",
       fill=MUTED, font=FONT_S)

chart_w = W - PAD['l'] - PAD['r']
chart_h1 = 330
chart_h2 = 330

draw_line_chart(d, PAD['l'], 90, chart_w, chart_h1,
                 [{'key': 'gap', 'color': BLUE, 'label': 'gap = v_u - v_l'}],
                 "gap (log scale y — iter 1의 큰 값부터 수렴까지 한 눈에)",
                 y_is_log=True, y_ticks=[0.05, 0.1, 0.2, 0.4, 0.8, 1.6], legend_at='top')

draw_line_chart(d, PAD['l'], 90 + chart_h1 + 70, chart_w, chart_h2,
                 [{'key': 'vu', 'color': RED, 'label': 'upper bound (v_u)'},
                  {'key': 'vl', 'color': GREEN, 'label': 'lower bound (v_l)'}],
                 "v_u / v_l (선형 스케일)")

out_path = os.path.join(RES, 'fivi_gap_convergence.png')
img.save(out_path)
print("saved", out_path)
