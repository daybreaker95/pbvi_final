"""Pure-Pillow PNG dashboard: belief-point count, gap(upper-lower), value
bounds (v_l/v_u, i.e. 'reward'), and Monte-Carlo clinical metrics, all
tracked over FiVI training iterations -- to see whether each stabilises."""
import json
import math
import os

from PIL import Image, ImageDraw, ImageFont

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
with open(os.path.join(RES, 'fivi_full_trajectory.json')) as f:
    OUT = json.load(f)
DATA = OUT['trajectory']
BASE = OUT['no_screen_baseline']
CLIN = [d for d in DATA if 'clinical' in d]
XKEY = 'train_time'   # wall-clock FiVI solve time (sec, excludes MC eval time) instead of iter count

W = 1040
BG = (255, 255, 255)
INK = (11, 11, 11)
MUTED = (137, 135, 129)
GRID = (225, 224, 217)
BLUE = (42, 120, 214)
RED = (227, 73, 72)
GREEN = (27, 175, 122)
ORANGE = (235, 104, 52)
VIOLET = (74, 58, 167)

KFONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
FONT = ImageFont.truetype(KFONT, 15)
FONT_B = ImageFont.truetype(KFONT, 18)
FONT_S = ImageFont.truetype(KFONT, 12)


def panel(draw, x0, y0, w, h, series, title, note=None, y_is_log=False, legend_at='bottom', hline=None, no_legend=False):
    xs_all = [d[XKEY] for d in DATA]
    xmin, xmax = min(xs_all), max(xs_all)

    def vals_of(s):
        return [(d[XKEY], s['get'](d)) for d in DATA if s['get'](d) is not None]

    all_pts = {s['label']: vals_of(s) for s in series}
    all_y = [v for pts in all_pts.values() for _, v in pts]
    if hline is not None:
        all_y = all_y + [hline]
    if y_is_log:
        pos_y = [v for v in all_y if v > 0]
        ymin_l, ymax_l = math.log10(min(pos_y)), math.log10(max(pos_y))
        pad = (ymax_l - ymin_l) * 0.08 or 0.15
        ymin_l -= pad
        ymax_l += pad
        def Y(v):
            return y0 + h - (math.log10(v) - ymin_l) / (ymax_l - ymin_l) * h
    else:
        ymin, ymax = min(all_y), max(all_y)
        pad = (ymax - ymin) * 0.12 or 1
        ymin -= pad
        ymax += pad
        def Y(v):
            return y0 + h - (v - ymin) / (ymax - ymin) * h

    def X(v):
        return x0 + (v - xmin) / (xmax - xmin) * w

    draw.text((x0, y0 - 24), title, fill=INK, font=FONT_B)
    if note:
        draw.text((x0, y0 - 4), note, fill=MUTED, font=FONT_S)

    n_grid = 5
    for i in range(n_grid + 1):
        if y_is_log:
            v = 10 ** (ymin_l + (ymax_l - ymin_l) * i / n_grid)
        else:
            v = ymin + (ymax - ymin) * i / n_grid
        y = Y(v)
        draw.line([(x0, y), (x0 + w, y)], fill=GRID, width=1)
        if y_is_log:
            lbl = f"{v:.2g}"
        else:
            span = ymax - ymin
            decimals = max(0, 2 - math.floor(math.log10(span))) if span > 0 else 2
            lbl = f"{v:.{decimals}f}"
        draw.text((x0 - 8, y - 7), lbl, fill=MUTED, font=FONT_S, anchor="ra")
    draw.line([(x0, y0 + h), (x0 + w, y0 + h)], fill=(195, 194, 183), width=1)

    if hline is not None:
        yy = Y(hline)
        for xx in range(int(x0), int(x0 + w), 10):
            draw.line([(xx, yy), (xx + 5, yy)], fill=MUTED, width=1)

    xtick_step = (xmax - xmin) / 10 or 1
    for i in range(11):
        xt = xmin + xtick_step * i
        x = X(xt)
        draw.line([(x, y0 + h), (x, y0 + h + 5)], fill=(195, 194, 183), width=1)
        lbl = f"{xt:.0f}" if XKEY == 'iter' else f"{xt:.1f}"
        draw.text((x, y0 + h + 8), lbl, fill=MUTED, font=FONT_S, anchor="ma")

    for s in series:
        pts = [(X(it), Y(v)) for it, v in all_pts[s['label']]]
        if len(pts) > 1:
            draw.line(pts, fill=s['color'], width=2)
        for (px, py) in pts:
            r = 4 if len(pts) < 25 else 2.4
            draw.ellipse([px - r, py - r, px + r, py + r], fill=s['color'])

    if not no_legend:
        lx = x0 + w - 230
        ly = (y0 + 6) if legend_at == 'top' else (y0 + h - len(series) * 22 - 6)
        for i, s in enumerate(series):
            yy = ly + i * 22
            draw.rectangle([lx, yy, lx + 12, yy + 12], fill=s['color'])
            draw.text((lx + 20, yy - 1), s['label'], fill=INK, font=FONT_S)


H_TOTAL = 2470
img = Image.new("RGB", (W, H_TOTAL), BG)
d = ImageDraw.Draw(img)

d.text((30, 16), "FiVI 순수 알고리즘(PBS/stochastic 없음) 학습 안정화 추이 — x축: 학습 소요시간(초)",
       fill=INK, font=FONT_B)
d.text((30, 40), f"age 40-85 (h=46), {len(DATA)} iterations(iter당 궤적 1개, 논문 그대로), 체크포인트 {len(CLIN)}개"
                 f"(MC N=10,000,000, CRN 적용 -- 매 checkpoint 동일 코호트로 평가), x축은 MC평가 시간 제외한 순수 FiVI backup 시간",
       fill=MUTED, font=FONT_S)

x0, w = 70, W - 70 - 30
h1 = 240
y = 90

panel(d, x0, y, w, h1, [
    {'label': 'belief point 개수', 'get': lambda r: r['n_belief'], 'color': ORANGE},
], "1. Belief point 개수 (n_belief) — 포화 안 함, 계속 증가", legend_at='top')
y += h1 + 70

panel(d, x0, y, w, h1, [
    {'label': 'gap = v_u - v_l', 'get': lambda r: r['gap'], 'color': BLUE},
], "2. Gap (upper - lower bound), log scale — t~2s 이후 고정 (~0.09)", y_is_log=True, legend_at='top')
y += h1 + 70

panel(d, x0, y, w, h1, [
    {'label': 'reward (정책 실제 시뮬레이션, MC 할인보상 평균)', 'get': lambda r: r.get('reward'), 'color': VIOLET},
], "3. Reward — Perseus/FiVI 논문이 쓰는 그 R (v_l과는 별개, 실측값)",
     note=f"no-screen 기준선(점선) = {BASE['reward']:.3f}", hline=BASE['reward'], legend_at='bottom')
y += h1 + 70

panel(d, x0, y, w, h1, [
    {'label': 'v_u (upper bound)', 'get': lambda r: r['vu'], 'color': RED},
    {'label': 'v_l (lower bound, 분석적 값함수)', 'get': lambda r: r['vl'], 'color': GREEN},
], "4. Value bounds (v_u / v_l) — 분석적 추정치, reward와는 다른 것", no_legend=True)
d.text((x0 + w - 280, y - 4), "v_u", fill=RED, font=FONT_S)
d.text((x0 + w - 250, y - 4), "/", fill=INK, font=FONT_S)
d.text((x0 + w - 244, y - 4), "v_l", fill=GREEN, font=FONT_S)
y += h1 + 70

panel(d, x0, y, w, h1, [
    {'label': 'cum_qaly (비할인 평생 누적 QALY, CRN)', 'get': lambda r: r.get('cum_qaly'), 'color': VIOLET},
], "5. 비할인 누적 QALY — ssrn-3802759 style 'total QALYs' 헤드라인 지표",
     note=f"no-screen 기준선(점선) = {BASE['cum_qaly']:.3f}  |  최종 Δqaly_vs_noscreen = "
          f"{[d for d in DATA if 'delta_qaly_vs_noscreen' in d][-1]['delta_qaly_vs_noscreen']:+.4f}",
     hline=BASE['cum_qaly'], legend_at='bottom')
y += h1 + 70

panel(d, x0, y, w, h1, [
    {'label': 'life_years (비할인 평생 생존연수, CRN)', 'get': lambda r: r.get('life_years'), 'color': GREEN},
], "6. LYG(Life-Years Gained) — Zaika 2024와 같은 지표(life-years, QALY 아님)",
     note=f"no-screen 기준선(점선) = {BASE['life_years']:.3f}년  |  최종 ΔLYG_vs_noscreen = "
          f"{[d for d in DATA if 'delta_lyg_vs_noscreen' in d][-1]['delta_lyg_vs_noscreen']:+.4f}년",
     hline=BASE['life_years'], legend_at='bottom')
y += h1 + 70

panel(d, x0, y, w, h1, [
    {'label': 'policy: CRC 사망/10만', 'get': lambda r: r.get('clinical', {}).get('crc_100k'), 'color': ORANGE},
], "7. 임상지표 — CRC 사망/10만 (MC 평가, 체크포인트별)",
     note=f"점선 = no-screen 기준선({BASE['crc_100k']:.0f})", hline=BASE['crc_100k'], legend_at='top')

out_path = os.path.join(RES, 'fivi_training_dashboard.png')
img.save(out_path)
print("saved", out_path)
