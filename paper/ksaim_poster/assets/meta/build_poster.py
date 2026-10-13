#!/usr/bin/env python3
"""Fill the composed paper2poster template with this paper's content.

Usage (idempotent; always starts from the pristine composed template):
    python assets/meta/build_poster.py            # template -> poster.html
The pristine template is assets/meta/poster_template.html, produced by
    python <skill>/references/compose_poster.py --orientation portrait \
        --layout full --style solid --header pv2 --theme teal --out <that path>

Every number below is drawn verbatim from paper/manuscript.md (Tables 1-10)
and the submitted KoSAIM abstract. Do not round or invent.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent
TEMPLATE = HERE / "poster_template.html"
TARGET = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "poster.html"

html = TEMPLATE.read_text(encoding="utf-8")


def must_replace(doc: str, old: str, new: str, what: str) -> str:
    if old not in doc:
        sys.exit(f"anchor not found for {what}: {old[:80]!r}")
    return doc.replace(old, new, 1)


# ── 1. structural edits ────────────────────────────────────────────────────
# 1a. Motivation has no figure: drop the teaser block (and its comment).
html = re.sub(
    r"\s*<!-- OPTIONAL: half-column Motivation figure\..*?-->\s*"
    r'<figure><img src="\{\{TEASER_FIGURE\}\}" alt=""><figcaption>\{\{TEASER_CAPTION\}\}</figcaption></figure>',
    "", html, count=1, flags=re.S)
assert "{{TEASER_FIGURE}}" not in html, "teaser figure still present"

# 1b. Motivation: one evidence bullet + a key-stat for the risk gradient.
html = must_replace(html,
    "          <li>{{MOTIVATION_2}}</li>\n        </ul>",
    "        </ul>\n"
    '        <div class="p-key-stat">\n'
    '          <div class="num">23×</div>\n'
    '          <div class="label">higher lifetime CRC incidence in the top 2 % of latent risk than in the bottom half: a gradient no fixed schedule can express, but one the policy infers from findings alone.</div>\n'
    '        </div>',
    "motivation key-stat")

# 1c. Problem: research-question pull-quote first, then the paragraph (a plain
#     text bottom keeps the painted callout out of the card's padding zone).
html = must_replace(html,
    "        <p>{{PROBLEM}}</p>",
    '        <div class="p-callout-bar">Can adapting each person\'s next interval to their own findings beat the best fixed schedule at equal or lower colonoscopy volume?</div>\n'
    "        <p>{{PROBLEM}}</p>",
    "problem callout")

# 1d. Method: the key equation (objective) is carried by the schematic's crimson
#     objective strip; a separate .p-eq beside the figure made the bullets column
#     taller than the figure and starved the figure below the 90% fill gate.
#     The three bullets render as numbered steps (P8) so the card carries a widget.
html = must_replace(html,
    "          <ul>\n            <li>{{METHOD_1}}</li>\n            <li>{{METHOD_2}}</li>\n            <li>{{METHOD_3}}</li>\n          </ul>",
    '          <div class="p-steps">\n'
    '            <div class="step">{{METHOD_1}}</div>\n'
    '            <div class="step">{{METHOD_2}}</div>\n'
    '            <div class="step">{{METHOD_3}}</div>\n'
    '          </div>',
    "method steps")

# 1e. Key Results: engine figure + 4-column table drawn from Table 4.
KEY_TABLE = '''        <table class="p-table results">
          <tr><th>Programme (engine, 1,000,000 per arm)</th><th>Colonoscopies per person</th><th>CRC deaths per 100,000</th><th>Deaths averted vs no screening per 1,000 colonoscopies</th></tr>
          <tr><td class="method">Fixed 10-yearly (50/60/70)</td><td>2.58</td><td>1032.1</td><td>3.32</td></tr>
          <tr><td class="method">Best fixed, no surveillance</td><td>2.44</td><td>980.9</td><td>3.71</td></tr>
          <tr class="best"><td class="method">Adaptive policy (ours)</td><td>2.29</td><td>899.8</td><td>4.32</td></tr>
          <tr><td class="method">10-yearly + surveillance</td><td>2.95</td><td>879.8</td><td>3.42</td></tr>
          <tr><td class="method">Adaptive, matched volume</td><td>2.87</td><td>818.1</td><td>3.72</td></tr>
        </table>
        <figure><img src="assets/figures/dp_frontier_c6b.png" alt=""><figcaption>Efficiency frontiers in the engine: CRC deaths (left) and diagnoses (right).</figcaption></figure>'''
html = re.sub(r'        <table class="results">.*?</table>', lambda m: KEY_TABLE, html, count=1, flags=re.S)
assert "{{BASELINE}}" not in html, "results table not replaced"

# 1f. Ablation Study card becomes "Ablation & Robustness" with the adherence figure.
ABL = '''      <div class="section grow" data-section="ablation-study">
        <h2>Robustness <button class="listen-btn" data-section="ablation-study">Listen</button></h2>
        <figure><img src="assets/figures/dp_adherence_c6b.png" alt=""><figcaption>Mortality and colonoscopy use vs adherence; n = 200,000.</figcaption></figure>
        <div class="p-key-stat">
          <div class="num">18%</div>
          <div class="label">{{ABLATION_1}}</div>
        </div>
        <p>{{ABLATION_2}}</p>
      </div>'''
html = re.sub(r'      <div class="section grow" data-section="ablation-study">.*?<p class="conclusion">\{\{ABLATION_CONCLUSION\}\}</p>\n      </div>',
              lambda m: ABL, html, count=1, flags=re.S)
# Lean render: no "So what" line and no delta pill under the Key Results table
# (the Key numbers card carries the +30 % hero; the pill would duplicate it).
html = must_replace(html, '        <p class="conclusion">{{KEY_RESULT_CONCLUSION}}</p>\n', "", "key-result conclusion drop")
html = must_replace(html, '        <div class="callout">{{HEADLINE_DELTA}}</div>\n', "", "key-result callout drop")
assert "<h2>Robustness " in html, "ablation card not replaced"

# 1g. New band between the results band and the Takeaway.
BAND35 = '''  <!-- BAND 3.5 — What the policy does | Risk information vs adaptivity -->
  <div class="columns">
    <div class="col">
      <div class="section grow" data-section="what-the-policy-does">
        <h2>What the policy does <button class="listen-btn" data-section="what-the-policy-does">Listen</button></h2>
        <table class="p-table">
          <tr><th>Colonoscopy ages along each observation path (<span style="text-transform:none">λ</span> = 0.001561)</th><th>Men</th><th>Women</th></tr>
          <tr><td>all normal</td><td>49, 69, 78</td><td>58, 73</td></tr>
          <tr><td>1–2 adenomas, then normal</td><td>49, 58, 67, 75</td><td>58, 63, 73, 80</td></tr>
          <tr><td>≥3 adenomas, then normal</td><td>49, 50, 57, 64, 75</td><td>58, 59, 63, 69, 74, 79</td></tr>
          <tr><td>advanced adenoma, then normal</td><td>49, 52, 64, 75</td><td>58, 63, 73, 80</td></tr>
          <tr><td>advanced adenoma twice</td><td>49, 52, 53, 61, 67, 75</td><td>58, 63, 67, 71, 76, 80</td></tr>
        </table>
        <p><strong>Graded, immediate response:</strong> 1–2 adenomas bring the next exam in 9 years (men) or 5 (women); an advanced adenoma in 3 or 5.</p>
        <p class="conclusion">A clean first exam earns 15–20 years; ≥3 adenomas pull the next exam to within 1–4 years (1 here); the sexes are offset by ~9 years (49 vs 58).</p>
      </div>
    </div>
    <div class="col">
      <div class="section grow" data-section="risk-information-vs-adaptivity">
        <h2>Risk information vs adaptivity <button class="listen-btn" data-section="risk-information-vs-adaptivity">Listen</button></h2>
        <p>If the risk class were <strong>observable at baseline</strong>, the policy would target it explicitly: <span class="hi">6.34 deaths averted per 1,000 colonoscopies,</span> none for the lowest-risk half and <span class="num">7.49</span> for the highest-risk 5% (mortality 5,535 under 10-yearly to 1,956 per 100,000).</p>
        <table class="p-table">
          <tr><th>CRC deaths / 100,000 (in-model)</th><th>No score</th><th>Score (AUC 0.60)</th><th>Score buys</th></tr>
          <tr><td>Best fixed schedule</td><td>972</td><td>942</td><td>−30</td></tr>
          <tr class="best"><td>Adaptive policy</td><td>892</td><td>867</td><td>−24</td></tr>
          <tr><td>Adaptivity buys</td><td>−80</td><td>−74</td><td></td></tr>
        </table>
        <div class="p-key-stat">
          <div class="num">3×</div>
          <div class="label">more benefit from adaptivity than from a realistic baseline risk score; the two gains are nearly additive.</div>
        </div>
        <p>In the engine (n = 200,000) the two cannot be told apart (+6.5 ± 25.1 per 100,000).</p>
      </div>
    </div>
  </div>

'''
html = must_replace(html, "  <!-- BAND 4 — full-width closing statement. -->", BAND35 + "  <!-- BAND 4 — full-width closing statement. -->", "band 3.5 insertion")

# 1g2. Key numbers: shorter heading (fits one line in the narrow track); stacked hero
#      (no .hero-main wrapper) over a 2x3 grid of verbatim tiles, closed by a soft callout.
html = must_replace(html, "<h2>Headline Numbers</h2>", "<h2>Key numbers</h2>", "headline heading")
HERO = '''        <div class="headline-hero">
          <div class="hero-val">{{HERO_VAL}}</div>
          <div class="hero-label">{{HERO_LABEL}}</div>
          <div class="supporting">
            <div class="stat-mini"><div class="val">{{STAT_2_VAL}}</div><div class="lbl">{{STAT_2_LBL}}</div></div>
            <div class="stat-mini"><div class="val">−201</div><div class="lbl">CRC diagnoses per 100,000 vs 10-yearly</div></div>
            <div class="stat-mini"><div class="val">{{STAT_3_VAL}}</div><div class="lbl">{{STAT_3_LBL}}</div></div>
            <div class="stat-mini"><div class="val">{{STAT_4_VAL}}</div><div class="lbl">{{STAT_4_LBL}}</div></div>
            <div class="stat-mini"><div class="val">−81</div><div class="lbl">CRC deaths per 100,000 vs best fixed, no surveillance</div></div>
            <div class="stat-mini"><div class="val">+18%</div><div class="lbl">deaths averted per colonoscopy vs 5-yearly screening</div></div>
          </div>
        </div>'''
html = re.sub(r'        <div class="headline-hero">.*?\n        </div>\n(?=      </div>)', lambda m: HERO + "\n", html, count=1, flags=re.S)
assert "−201" in html, "headline hero not replaced"

# 1h. Takeaway: mic-drop callout + plain closing paragraph (ends with text so the painted widget seats inside the card).
html = must_replace(html,
    "      <p>{{TAKEAWAY}}</p>",
    '      <div class="p-callout-primary">{{TAKEAWAY}}</div>\n'
    "      <p>The solved policy is a legible three-tier surveillance rule with a sex offset: a 15–20-year interval after a clean first exam, 3–9 years after 1–2 or advanced adenomas, and re-examination within 1–4 years after ≥3 adenomas. Limitations: all results are internal to CMOST and inherit its calibration, and non-attendance is modelled as random. Life-years did not differ significantly between adaptive and fixed schedules, an honest null reported alongside the positive findings.</p>",
    "takeaway")

# 1i. Narration playlist for the sections actually rendered.
html = re.sub(r'const PLAYLIST = \[[^\]]*\];',
              'const PLAYLIST = ["title", "problem", "motivation", "method", "key-result", "ablation-study", "what-the-policy-does", "risk-information-vs-adaptivity", "takeaway"];',
              html, count=1)

# 1j. Header: collapse the pv2 masthead to the title block; the venue + QR strip and the
#     institution emblem flank the byline (markup moved in 1j2).
OVERRIDES = '''<style id="ksaim-overrides">
  /* KoSAIM 2026: one byline row under the title. The venue badge + QR sit at its left edge and
     the institution emblem at its right edge, both level with authors + affiliations + contact. */
  header.titlebar.pv2 { grid-template-columns: minmax(0, 1fr); grid-template-areas: "tt"; }
  header.titlebar.pv2 .byline-row .meta-rail { grid-area: auto; grid-column: 1; grid-row: 1; justify-self: start; align-self: center; margin: 0 0 0 6pt; padding: 0; border-right: 0; }
  header.titlebar .byline-row { display: grid; grid-template-columns: 1fr auto 1fr; align-items: stretch; column-gap: 12pt; }
  header.titlebar .byline { grid-column: 2; text-align: center; }
  header.titlebar .inst-logo { grid-column: 3; justify-self: end; align-self: stretch; height: 100%; max-height: 150pt; width: auto; object-fit: contain; background: none; box-shadow: none; padding: 0; margin-right: 6pt; }
  header.titlebar.pv2 .title-block h1 { font-size: 52pt; }
  header.titlebar.pv2 .authors { font-size: 30pt; }
  /* A0 body text: 28pt reads comfortably at 2 m and leaves room for five bands. */
  .poster { font-size: 28pt; gap: 18pt; }
  /* Print poster: the hidden Listen buttons must not reserve heading width
     (a 5.5em pad wraps every two-word heading in the narrow result columns). */
  .poster .columns .section h2, .poster .band-full .section h2 { padding-right: 0.3em; }
  /* Results band: compact table, and cap the two figures just above their 90%-width floors. */
  table.results { font-size: 0.78em; }
  table.results th, table.results td { padding: 4px 8px; }
  table.results td.method { white-space: nowrap; }
  .p-table { font-size: 0.85em; }
  /* Key numbers: six supporting tiles in a 2-column grid under the stacked hero. */
  .headline-hero .hero-val { font-size: 100pt; }
  .headline-hero .supporting { display: grid; grid-template-columns: 1fr 1fr; gap: 0.4em 0.8em; justify-items: center; margin-top: 0.5em; }
  .headline-hero .stat-mini .val { font-size: 42pt; }
  .headline-hero .stat-mini .lbl { font-size: 21pt; text-align: center; }
  .section[data-section="what-the-policy-does"] .p-table { font-size: 0.83em; }
  .section[data-section="what-the-policy-does"] .p-table th, .section[data-section="what-the-policy-does"] .p-table td { padding: 0.14em 0.5em; }
  /* Problem card sits 3 px under the FULL gate: open the leading a touch instead of adding a line. */
  .section[data-section="problem"] > p { line-height: 1.62; }
  .section[data-section="problem"] .p-callout-bar { line-height: 1.5; }
  .section[data-section="what-the-policy-does"] .p-table th:first-child { width: 42%; }
  .section[data-section="what-the-policy-does"] .p-table td:nth-child(n+2) { white-space: nowrap; }
  .section[data-section="risk-information-vs-adaptivity"] .p-table { font-size: 0.8em; }
  /* Guideline: affiliations must be clearly visible at the top. */
  header.titlebar .author-legend { font-size: 28pt; opacity: 1; }
  /* Show the corresponding author's contact line (hidden by the template default). */
  header.titlebar .contact { display: block; font-size: 24pt; opacity: 0.9; margin-top: 0.05em; }
  /* Key-stat as a flex row (number | label) instead of a float: identical look,
     but the label box no longer underlaps the number in the PPTX export. */
  .p-key-stat { display: flex; align-items: flex-start; gap: 0.35em; }
  .p-key-stat .num { float: none; flex: 0 0 auto; margin: 0; }
  .p-key-stat .label { display: block; flex: 1 1 auto; min-width: 0; }
  .p-key-stat::after { display: none; }
  /* Method hero: widen the figure so it fills the row height set by the bullets. */
  .method-hero .section[data-section="method"] > .method-body { grid-template-columns: 1.6fr 1fr; }
  .method-hero .section[data-section="method"] { padding-top: 10pt; padding-bottom: 10pt; }
  .method-hero .section[data-section="method"] figcaption { font-size: 0.7em; margin-top: 0.25em; }
  .p-table.results { font-size: 0.70em; margin-top: 0; }
  .p-table.results th { font-size: 0.68em; }
  .p-table.results th, .p-table.results td { padding: 1px 6px; }
  /* Reclaim vertical room: tighter card padding, tighter header, smaller QR. */
  .poster .section { padding: 14pt 18pt; }
  header.titlebar { padding-top: 14pt; padding-bottom: 12pt; }
  .titlebar.pv2 .qr-block .qr-img { width: 84pt; height: 84pt; }
  .section[data-section="key-result"] > figure { max-height: 520px; }
  /* The results band carries the two figures: give it a larger share of free height. */
  .poster .columns.columns--3 { flex-grow: 2.2; min-height: 1036px; }
  .poster .columns:has(.section[data-section="what-the-policy-does"]) { min-height: 800px; }
  .band-full .section p:last-child { margin-bottom: 0; }
  .p-table td.method { white-space: nowrap; }
  .section[data-section="ablation-study"] > figure { max-height: 305px; }
  /* Takeaway frame: mirror the 6px left accent bar on the right and the 4px top stripe at the bottom. */
  .section[data-section="takeaway"] { border-right: 6px solid var(--accent); }
  .section[data-section="takeaway"]::after {
    content: ""; position: absolute; bottom: 0; left: 0; right: 0; height: 4px; pointer-events: none;
    background: linear-gradient(90deg, var(--section-accent, var(--accent)),
      color-mix(in srgb, var(--section-accent, var(--accent)) 70%, white));
  }
</style>
</head>'''
html = must_replace(html, "</head>", OVERRIDES, "header overrides")

# 1j2. Header: put the venue badge + QR at the left edge and the institution emblem (transparent
#      PNG, no chip) at the right edge of the byline rows (authors / affiliations / contact), and
#      drop the bottom logo strip.
NL = "\n"
m_mr = re.search(r'\n(  <div class="meta-rail" data-section="header-meta">\n.*?\n  </div>)(?=\n  <div class="title-block">)',
                 html, flags=re.S)
assert m_mr, "meta rail not found"
html = html[:m_mr.start()] + html[m_mr.end():]
META_RAIL = NL.join("    " + line for line in m_mr.group(1).split(NL))
OLD_BYLINE = NL.join([
    '    <div class="authors">{{AUTHORS}}</div>',
    '    <div class="author-legend">{{AUTHOR_LEGEND}}</div>',
    '    <div class="contact">{{CONTACT}}</div>',
    '  </div>',
])
NEW_BYLINE = NL.join([
    '    <div class="byline-row">',
    META_RAIL,
    '      <div class="byline">',
    '        <div class="authors">{{AUTHORS}}</div>',
    '        <div class="author-legend">{{AUTHOR_LEGEND}}</div>',
    '        <div class="contact">{{CONTACT}}</div>',
    '      </div>',
    '      <img class="inst-logo" src="assets/logos/university-of-ulsan.png" alt="University of Ulsan">',
    '    </div>',
    '  </div>',
])
html = must_replace(html, OLD_BYLINE, NEW_BYLINE, "byline row")
html, n_lb = re.subn(r'\s*<div class="logo-block institution-rail"[^>]*>.*?</div>', "", html, count=1, flags=re.S)
assert n_lb == 1, "logo strip not removed"

# 1k. Drop the commented-out optional-section recipes (they carry {{CONTRIBUTION_*}} /
#     {{DATASET_*}} tokens inside an HTML comment and would trip the leftover gate).
html, n_opt = re.subn(r"\s*<!-- ═+\s*OPTIONAL SECTIONS.*?═+ -->", "", html, count=1, flags=re.S)
assert n_opt == 1, "optional-sections comment block not found"

# ── 2. placeholder substitutions ───────────────────────────────────────────
SUBS = {
    "{{TITLE}}": "Adaptive Colonoscopy Screening From a Mixed-Observability Markov Decision Process Built on the CMOST Microsimulation: More Cancers and Deaths Prevented per Colonoscopy Than Any Fixed Schedule",
    "{{AUTHORS}}": "Taehyung Lee<sup>1</sup>, Inseo Choi<sup>2</sup>, Jaehoon Shim<sup>3</sup>",
    "{{AUTHORS_PLAIN}}": "Taehyung Lee, Inseo Choi, Jaehoon Shim",
    "{{AUTHOR_LEGEND}}": '<div class="institutes-line"><sup>1</sup> University of Ulsan College of Medicine &nbsp;&nbsp; <sup>2</sup> Chosun University College of Medicine &nbsp;&nbsp; <sup>3</sup> Seoul National University</div>',
    "{{VENUE_NAME}}": "KoSAIM",
    "{{VENUE_YEAR}}": "2026",
    "{{VENUE_TAG}}": "POSTER",
    "{{VENUE_LINK}}": "https://www.kosaim.org/",
    "{{CONTACT}}": "Email: daybreaker9502@gmail.com &nbsp;·&nbsp; Phone: 010-8023-3200",
    "{{LOGO_1}}": "assets/logos/university-of-ulsan.png", "{{LOGO_2}}": "", "{{LOGO_3}}": "", "{{LOGO_4}}": "",
    "{{QR_PAPER}}": "",
    "{{QR_CODE}}": "assets/qr/code.png",

    "{{PROBLEM}}": "Colonoscopy prevents colorectal cancer (CRC) but is <strong>scarce and invasive</strong>: <span class=\"hi\">judge it by what it prevents per exam.</span> Optimised schedules (Zaika et al. 2024) are <strong>fixed</strong> and ignore what a person's <strong>colonoscopies found</strong>, the one signal screening itself generates.",

    "{{MOTIVATION_1}}": "<strong>A colonoscopy is evidence.</strong> ≥3 adenomas lift the probability of the high-risk classes from <span class=\"num\">0.05</span> to <span class=\"num\">0.93</span> in men (0.98 in women); a normal exam lowers it to 0.007 in men and 0.017 in women.",

    "{{METHOD_1}}": "<strong>State.</strong> Observed: age, sex, years since last colonoscopy (τ), last finding (normal / 1–2 / ≥3 / advanced adenomas). Hidden belief: <span class=\"num\">6</span> risk classes × <span class=\"num\">11</span> clinical states. WAIT or SCREEN yearly, ages 40–80.",
    "{{METHOD_2}}": "<strong>Kernels from the engine.</strong> Maximum-likelihood estimates from 4,000,000 CMOST lives (<span class=\"num\">4,520,648</span> colonoscopies), conditioned on (τ, last finding).",
    "{{METHOD_3}}": "<strong>Solve, then deploy.</strong> Point-based value iteration; the shadow price λ sweeps the frontier. Every policy is re-run inside CMOST on its simulated findings against 10-yearly and 5-yearly screening, the best of 2,112 fixed schedules and 10-yearly + surveillance (headline arms: <span class=\"num\">1,000,000</span> persons).",
    "{{METHOD_FIGURE}}": "assets/figures/momdp_pipeline.png",
    "{{METHOD_CAPTION}}": "CMOST supplies the kernels; point-based value iteration solves the MOMDP; policies run back inside the engine.",

    "{{HEADLINE_DELTA}}": "+30 % deaths and +25 % diagnoses averted per 1,000 colonoscopies, with 11 % fewer colonoscopies",
    "{{KEY_RESULT_CONCLUSION}}": "Adaptivity, not better calendar ages, drives the gain: re-optimised fixed ages recover only about 39 % of it.",

    "{{ABLATION_1}}": "mortality cut by fixed 10-yearly screening (missed slot lost) at 30% attendance, down from 46%; the adaptive policy, not re-solved, holds 44–53%.",
    "{{ABLATION_2}}": "<strong>Model ablation.</strong> Policies from three simpler models still beat 10-yearly screening by 121–172 deaths per 100,000 (n = 200,000).",
    "{{ABLATION_CONCLUSION}}": "The gain rests on responding to findings at all, not on the precise model.",

    "{{HERO_VAL}}": "+30%",
    "{{HERO_LABEL}}": "CRC deaths averted per 1,000 colonoscopies vs 10-yearly screening",
    "{{STAT_2_VAL}}": "−132", "{{STAT_2_LBL}}": "CRC deaths per 100,000 vs 10-yearly",
    "{{STAT_3_VAL}}": "−11%", "{{STAT_3_LBL}}": "colonoscopies per person vs 10-yearly",
    "{{STAT_4_VAL}}": "44–53%", "{{STAT_4_LBL}}": "mortality cut at attendance 0.3–1.0 (n = 200,000)",

    "{{TAKEAWAY}}": "Individualising colonoscopy timing on realised findings prevents more cancers and deaths per colonoscopy than any fixed schedule, including schedules optimised for the same simulator, and the gain survives imperfect adherence.",
}

for token, value in SUBS.items():
    html = html.replace(token, value)

# House style: no space before "%" (a trailing " %" can wrap alone -> polish ORPHAN warning).
html = html.replace(" %", "%")

leftover = sorted(set(re.findall(r"\{\{[A-Z0-9_]+\}\}", html)))
if leftover:
    sys.exit(f"unreplaced placeholders remain: {leftover}")

TARGET.write_text(html, encoding="utf-8")
print(f"wrote {TARGET} ({len(html)} bytes)")
