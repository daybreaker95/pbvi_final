#!/usr/bin/env python3
"""Build the paper2assets-compatible bundle for the KoSAIM 2026 poster.

The upstream `paper2assets` skill is not installed on this machine, so this
script produces the same contract by hand, honestly annotated from the
manuscript (paper/manuscript.md) and its figure table:

  assets/meta/figures.json    figure records with the figure_semantics.v2 contract
  assets/meta/sections.json   section index (normalized JSON)
  assets/meta/narration.json  Listen-button scripts parsed from paper_spec.md
  assets/meta/metadata.json   venue / links / QR manifest
  manifest.json               commit markers (sha256) that select_figures.py +
                              check_poster.py preflight verify

Run from anywhere:  python assets/meta/build_bundle.py
Re-run whenever figures.json content or paper_spec.md audio scripts change.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from PIL import Image

OUT = Path(__file__).resolve().parent.parent.parent          # .../paper/ksaim_poster
META = OUT / "assets" / "meta"
FIG_DIR = OUT / "assets" / "figures"
SPEC = META / "paper_spec.md"

CODE_URL = "https://github.com/daybreaker95/pbvi_final"

SCHEMA = "figure_semantics.v2"


def rec(file: str, fid: str, label: str, page: int, caption: str,
        roles: list[tuple[str, float, list[str]]],
        relevance: list[tuple[str, float, list[str]]],
        source: str) -> dict:
    """One figures.json record satisfying utils/figure_semantics.semantics_available."""
    path = FIG_DIR / Path(file).name
    width = height = None
    if path.is_file():
        with Image.open(path) as im:
            width, height = im.size
    role_map = {r: c for r, c, _ in roles}
    primary = min(role_map, key=lambda r: (-role_map[r], r))
    exclusions = []
    for r, c, ev in roles:
        if r in ("method", "result", "ablation") and c >= 0.55:
            exclusions.append({"section_id": "motivation", "role": r,
                               "confidence": c, "evidence": ev})
    return {
        "file": file,
        "path": file,
        "page": page,
        "figure_id": fid,
        "original_label": label,
        "caption_label": label,
        "caption": caption,
        "width": width,
        "height": height,
        "column": "full",
        "caption_provenance": {
            "selected_source": source,
            "quality": {"status": "trusted", "reasons": []},
        },
        "semantic_schema_version": SCHEMA,
        "semantic_eligible": True,
        "semantic_status": "classified",
        "semantic_role": primary,
        "semantic_confidence": role_map[primary],
        "semantic_roles": [{"role": r, "confidence": c, "evidence": ev} for r, c, ev in roles],
        "section_relevance": [{"section_id": s, "confidence": c, "evidence": ev} for s, c, ev in relevance],
        "section_exclusions": exclusions,
    }


FIGURES = [
    rec("assets/figures/momdp_pipeline.png", "fig-method-pipeline", "Method schematic", 0,
        "Pipeline: the instrumented CMOST engine supplies maximum-likelihood kernels for a "
        "mixed-observability MDP, solved by point-based value iteration and deployed back "
        "inside the engine on population-paired arms.",
        roles=[("method", 0.95, ["poster schematic of Sections 2.1-2.6: engine -> kernels -> MOMDP -> PBVI -> deployment"])],
        relevance=[("method", 0.95, ["Methods 2.1-2.6"])],
        source="poster_schematic_from_methods"),
    rec("assets/figures/dp_frontier_c6b.png", "fig-1", "Figure 1", 1,
        "Efficiency frontiers in the engine: CRC deaths (left) and CRC diagnoses (right) per "
        "100,000 versus colonoscopies per person for the adaptive policy family, the guideline "
        "schedules, the best fixed schedules and the surveillance-augmented programmes.",
        roles=[("result", 0.95, ["Figure 1 caption: efficiency frontiers in the engine; Table 4 headline arms"])],
        relevance=[("key-result", 0.95, ["Section 3.3"])],
        source="manuscript_figure_table"),
    rec("assets/figures/dp_validation_c6b.png", "fig-2", "Figure 2", 2,
        "Model versus engine decision-time prevalence of early adenoma, advanced adenoma and "
        "undetected cancer by age in men, under 10-yearly screening (left) and no screening (right).",
        roles=[("dataset", 0.70, ["Figure 2 caption: model reproduces the engine's prevalence trajectories (Section 3.1 validation)"]),
               ("method", 0.50, ["validates the reduced model of Section 2"]),
               ("result", 0.40, ["Table 1 model-vs-engine agreement"])],
        relevance=[("dataset-benchmark", 0.70, ["Section 3.1"]), ("method", 0.50, ["Section 2"])],
        source="manuscript_figure_table"),
    rec("assets/figures/dp_policy_c6b_lam0.001561.png", "fig-3", "Figure 3", 3,
        "Screening ages of the headline policy along canonical observation paths and by known "
        "risk class, by sex.",
        roles=[("qualitative", 0.80, ["Figure 3 caption: what the policy does (Section 3.4)"]),
               ("result", 0.60, ["Table 6 screening ages"]),
               ("method", 0.40, ["illustrates the solved policy"])],
        relevance=[("what-the-policy-does", 0.85, ["Section 3.4"]), ("key-result", 0.60, ["Table 6"])],
        source="manuscript_figure_table"),
    rec("assets/figures/dp_riskscore_c6b.png", "fig-4", "Figure 4", 4,
        "Value of a baseline risk score against its discrimination at matched colonoscopy "
        "volume (left) and the engine arms on the efficiency plane (right).",
        roles=[("result", 0.80, ["Figure 4 caption: Section 3.6 score ladder and engine arms"]),
               ("ablation", 0.45, ["information vs adaptivity 2x2 (Table 9)"])],
        relevance=[("risk-information-vs-adaptivity", 0.85, ["Section 3.6"]), ("key-result", 0.70, ["Tables 8-9"])],
        source="manuscript_figure_table"),
    rec("assets/figures/dp_adherence_c6b.png", "fig-5", "Figure 5", 5,
        "CRC mortality and colonoscopy use versus attendance probability for the fixed "
        "programme (with and without recall) and the adaptive policy.",
        roles=[("result", 0.85, ["Figure 5 caption: Section 3.7 adherence scenarios (Table 10)"]),
               ("ablation", 0.60, ["robustness analysis"])],
        relevance=[("robustness", 0.90, ["Section 3.7"]), ("ablation-study", 0.60, ["Table 10"]), ("key-result", 0.65, ["Table 10"])],
        source="manuscript_figure_table"),
    rec("assets/figures/dp_per_colonoscopy_c6b.png", "fig-6", "Figure 6", 6,
        "Deaths and diagnoses averted per 1000 colonoscopies, all headline arms and the "
        "surveillance-augmented programmes.",
        roles=[("result", 0.75, ["Figure 6 caption: per-colonoscopy efficiency, Table 4 columns"])],
        relevance=[("key-result", 0.75, ["Table 4"])],
        source="manuscript_figure_table"),
]

SEMANTIC_COMMIT_FIELDS = (
    "file", "page", "caption", "caption_provenance", "figure_id",
    "semantic_schema_version", "semantic_eligible", "semantic_status",
    "semantic_role", "semantic_confidence", "semantic_roles",
    "section_exclusions", "section_relevance",
)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def kebab(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


SECTION_IDS = {
    "Problem": "problem", "Motivation": "motivation", "Contribution": "contribution",
    "Method": "method", "Dataset / Benchmark": "dataset-benchmark", "Key Result": "key-result",
    "Ablation Study": "ablation-study", "Headline Numbers": "headline-numbers", "Takeaway": "takeaway",
}


def parse_spec() -> tuple[list[dict], list[dict], str]:
    """Return (sections, narration_sections, title) parsed from paper_spec.md."""
    if not SPEC.is_file():
        return [], [], ""
    text = SPEC.read_text(encoding="utf-8")
    title = ""
    m = re.search(r"^#\s+(.+)$", text, re.M)
    if m:
        title = m.group(1).strip()
    sections, narration = [], []
    tm = re.search(r"\*\*title_audio_script:\*\*\s*(.+)", text)
    if tm:
        narration.append({"id": "title", "heading": "Title", "text": tm.group(1).strip()})
    for block in re.split(r"^## ", text, flags=re.M)[1:]:
        heading, _, body = block.partition("\n")
        heading = heading.strip()
        name = heading.replace("Custom: ", "").strip()
        sid = SECTION_IDS.get(name, kebab(name))
        sections.append({"id": sid, "heading": name, "custom": heading.startswith("Custom:")})
        am = re.search(r"\*\*Audio script:\*\*\s*(.+)", body)
        if am:
            narration.append({"id": sid, "heading": name, "text": am.group(1).strip()})
    return sections, narration, title


def main() -> None:
    META.mkdir(parents=True, exist_ok=True)
    sections, narration, title = parse_spec()

    figures_payload = FIGURES
    (META / "figures.json").write_text(
        json.dumps(figures_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    projection = [{k: r.get(k) for k in SEMANTIC_COMMIT_FIELDS} for r in figures_payload]
    fig_digest = sha(json.dumps(projection, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":")).encode("utf-8"))

    sections_payload = {"schema": "paper2assets.sections.v1", "title": title, "sections": sections}
    sections_bytes = (json.dumps(sections_payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    (META / "sections.json").write_bytes(sections_bytes)

    narration_payload = {"provider": "edge", "voice": None, "sections": narration}
    narration_bytes = (json.dumps(narration_payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    (META / "narration.json").write_bytes(narration_bytes)

    qr_path = "assets/qr/code.png"
    metadata = {
        "title": title,
        "venue": "KoSAIM",
        "year": "2026",
        "emails": [],
        "code_url": CODE_URL,
        "project_url": "",
        "paper_url": "",
        "arxiv_id": "",
        "doi": "",
        "qr": ([{"path": qr_path, "label": "Code", "url": CODE_URL}]
               if (OUT / qr_path).is_file() else []),
    }
    (META / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                                        encoding="utf-8")

    hashes = {"figures": fig_digest, "sections": sha(sections_bytes), "narration": sha(narration_bytes)}
    package_generation = sha("\n".join(f"{n}:{d}" for n, d in sorted(hashes.items())).encode("utf-8"))
    manifest = {
        "schema": "paper2assets.manifest.v2",
        "producer": "build_bundle.py (hand-built; paper2assets not installed)",
        "title": title,
        "source": "paper/manuscript.md",
        "figure_semantics_generation": fig_digest,
        "artifacts": {
            "figures": {"path": "assets/meta/figures.json", "hash_scope": "semantic-selection-fields", "sha256": fig_digest},
            "sections": {"path": "assets/meta/sections.json", "hash_scope": "normalized-json", "sha256": hashes["sections"]},
            "narration": {"path": "assets/meta/narration.json", "hash_scope": "normalized-json", "sha256": hashes["narration"]},
        },
        "package_generation": package_generation,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
    missing = [r["file"] for r in figures_payload if not (OUT / r["file"]).is_file()]
    print(f"bundle written to {OUT}")
    print(f"  figures: {len(figures_payload)} records; missing on disk: {missing or 'none'}")
    print(f"  sections: {len(sections)}; narration clips: {len(narration)}")


if __name__ == "__main__":
    main()
