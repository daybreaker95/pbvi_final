"""Renders the controlled PBS on/off ablation (results/pbs_ablation.json)
as a 2-panel PNG: gap trajectory (both variants) and v_l trajectory (both
variants, with the shared/frozen v_u as a reference line)."""
import json
import os

from PIL import Image, ImageDraw, ImageFont

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
with open(os.path.join(RES, 'pbs_ablation.json')) as f:
    D = json.load(f)
NO_PBS = D['no_pbs']
PBS = D['pbs']

W = 1000
BG = (255, 255, 255)
INK = (11, 11, 11)
MUTED = (137, 135, 129)
GRID = (225, 224, 217)
BLUE = (42, 120, 214)
RED = (227, 73, 72)
GREEN = (27, 175, 122)
VIOLET = (74, 58, 167)

KFONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
FONT = ImageFont.truetype(KFONT, 15)
FONT_B = ImageFont.truetype(KFONT, 19)
FONT_S = ImageFont.truetype(KFONT, 12)


def panel(draw, x0, y0, w, h, series, title, note=None, hline=None, legend_at='bottom',
          skip_iters=0):
    xs_all = [r['iter'] for r in NO_PBS]
    xmin, xmax = min(xs_all), max(xs_all)

    def vals_of(key, data):
        return [(r['iter'], r[key]) for r in data]

    # y-range is fit to iterations AFTER skip_iters -- the PBS iter-1/2
    # cold-start transient (random improve-only backup on an empty vector
    # set) is a known, expected outlier that would otherwise squash the
    # converged-region detail; it is still PLOTTED, just outside the
    # fitted range (clipped at the top, annotated).
    all_y = []
    for s in series:
        all_y += [v for x, v in vals_of(s['key'], s['data']) if x > skip_iters]
    if hline is not None:
        all_y.append(hline)
    ymin, ymax = min(all_y), max(all_y)
    pad = (ymax - ymin) * 0.12 or 0.01
    ymin -= pad
    ymax += pad

    draw.text((x0, y0 - 26), title, font=FONT_B, fill=INK)

    plot_h = h - 30
    plot_y0 = y0 + 10

    def xm(x):
        return x0 + (x - xmin) / (xmax - xmin) * w

    def ym(y):
        return plot_y0 + plot_h - (y - ymin) / (ymax - ymin) * plot_h

    for i in range(5):
        gy = plot_y0 + plot_h * i / 4
        draw.line([(x0, gy), (x0 + w, gy)], fill=GRID, width=1)
        yv = ymax - (ymax - ymin) * i / 4
        draw.text((x0 - 8, gy - 6), f"{yv:.4f}", font=FONT_S, fill=MUTED, anchor='ra')

    if hline is not None:
        hy = ym(hline)
        for xseg in range(x0, x0 + w, 10):
            draw.line([(xseg, hy), (xseg + 5, hy)], fill=MUTED, width=1)

    top_bound = plot_y0 - 4
    for s in series:
        pts = vals_of(s['key'], s['data'])
        coords = [(xm(x), max(top_bound, min(plot_y0 + plot_h, ym(y)))) for x, y in pts]
        draw.line(coords, fill=s['color'], width=2)
        for (x, yv), (cx, cy) in zip(pts, coords):
            clipped = ym(yv) < top_bound
            r = 5 if clipped else 2
            draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                         outline=s['color'] if clipped else None,
                         width=2 if clipped else 0,
                         fill=None if clipped else s['color'])
            if clipped:
                lbl_row = 0 if x <= 1 else 1
                draw.text((cx + 8, top_bound - 20 + lbl_row * 15),
                          f"iter{x}={yv:.2f}", font=FONT_S, fill=s['color'])

    for i in range(0, len(xs_all), 5):
        xt = xs_all[i]
        gx = xm(xt)
        draw.line([(gx, plot_y0 + plot_h), (gx, plot_y0 + plot_h + 4)], fill=MUTED, width=1)
        draw.text((gx, plot_y0 + plot_h + 6), f"{xt}", font=FONT_S, fill=MUTED, anchor='ma')

    if note:
        draw.text((x0, plot_y0 + plot_h + 24), note, font=FONT_S, fill=MUTED)

    if legend_at:
        lx = x0
        ly = y0 - 26 + (0 if legend_at == 'top' else 0)
        lx = x0 + w - 10
        for i, s in enumerate(series):
            tw = draw.textlength(s['label'], font=FONT_S)
            lx -= tw + 22
        for s in series:
            draw.ellipse([lx, y0 - 22, lx + 8, y0 - 14], fill=s['color'])
            draw.text((lx + 12, y0 - 24), s['label'], font=FONT_S, fill=INK)
            lx += draw.textlength(s['label'], font=FONT_S) + 22


H = 1000
img = Image.new('RGB', (W, H), BG)
d = ImageDraw.Draw(img)

d.text((30, 24), "PBS on/off 통제 비교 (iteration 수·stochastic trajectory·wtp 전부 고정, PBS만 토글)",
       font=FONT_B, fill=INK)
d.text((30, 50), "x축 = FiVI iteration (1~39)", font=FONT_S, fill=MUTED)

x0, w = 90, W - 130
h1 = 380

y = 90
panel(d, x0, y, w, h1, [
    {'label': 'gap (PBS 없음)', 'key': 'gap', 'data': NO_PBS, 'color': BLUE},
    {'label': 'gap (PBS)', 'key': 'gap', 'data': PBS, 'color': RED},
], "1. Gap (v_u - v_l) - PBS 콜드스타트(빈 원, iter1~2)는 범위 밖",
   note="PBS는 iter1=3.399, iter2=1.859로 시작(랜덤 improve-only backup의 예상된 초기 misfit) - y축은 왜곡 방지 위해 iter>2 기준",
   skip_iters=2)
y += h1 + 90

panel(d, x0, y, w, h1, [
    {'label': 'v_l (PBS 없음)', 'key': 'vl', 'data': NO_PBS, 'color': BLUE},
    {'label': 'v_l (PBS)', 'key': 'vl', 'data': PBS, 'color': RED},
], "2. v_l (lower bound) - v_u는 두 버전 모두 39회 내내 완전히 동일(점선)",
   note="점선 = v_u (PBS 유무 무관, 완전히 고정) = 21.10936  |  PBS iter1=17.71, iter2=19.25 (범위 밖)",
   hline=21.109356545560825, skip_iters=2)

out_path = os.path.join(RES, 'pbs_ablation.png')
img.save(out_path)
print('saved', out_path)
