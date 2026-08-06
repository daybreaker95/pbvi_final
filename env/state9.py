"""
state9.py -- NEW 9-state clinical discretisation, kept SEPARATE from
env/cmost_individual.py's clinical_state() (the currently-active 7-state
pipeline) so that pipeline is left untouched.

Axis: full cancer stage (I-IV), not early/advanced buckets, and NOT folded
into a single "Diagnosed" absorbing state. Instead, "detected" (has this
cancer been diagnosed yet) is returned SEPARATELY as a boolean -- it is meant
to be used as fully-observed CONTEXT (like age), not as part of the
belief-tracked state itself: the same Ca-I..IV states are shared by
undiagnosed and diagnosed patients, just under a different transition/reward
table selected by the `detected` context flag.

Polyps (Normal/Early Polyp/Advanced Polyp) never need a "detected" context:
a colonoscopy finding a polyp removes it immediately (no persistent
"detected polyp" state), unlike cancer, which persists (with ongoing CRC-
death risk) after diagnosis.

Once a cancer is diagnosed, CMOST's own engine freezes further natural-
history stage progression (Ca.Cancer bookkeeping stops; only the
already-drawn mortality-matrix survival clock matters), so the state
reported for a diagnosed patient is simply the stage AT DIAGNOSIS
(pt.det_stage[0]), not a moving stage.
"""
from __future__ import annotations

NORMAL = 0
EARLY_POLYP = 1
ADV_POLYP = 2
CA_I = 3
CA_II = 4
CA_III = 5
CA_IV = 6
CRC_DEATH = 7
OTHER_DEATH = 8
N_STATES9 = 9
STATE9_NAMES = [
    "Normal", "Early Polyp", "Advanced Polyp",
    "Cancer I", "Cancer II", "Cancer III", "Cancer IV",
    "CRC Death", "Other Death",
]

_CA_STAGE_TO_IDX = {7: CA_I, 8: CA_II, 9: CA_III, 10: CA_IV}


def clinical_state9(pt) -> tuple[int, bool]:
    """Return (state_index, detected) for a cmost_individual.Patient.

    detected is only meaningful (True) for the Ca-I..IV states; for
    Normal/Early Polyp/Advanced Polyp it is always False (no persistent
    detected-polyp state exists), and for the two death states it is
    irrelevant (both are terminal regardless).
    """
    if not pt.alive:
        return (CRC_DEATH, True) if pt.death_cause == 2 else (OTHER_DEATH, False)
    if pt.ever_clinical:
        # frozen at the stage recorded at diagnosis (engine stops advancing
        # Ca.Cancer bookkeeping once a cancer is detected)
        stage = int(pt.det_stage[0]) if pt.det_stage else 7
        return _CA_STAGE_TO_IDX.get(stage, CA_I), True
    if pt.ca_stage:
        stage = max(pt.ca_stage)
        return _CA_STAGE_TO_IDX.get(stage, CA_I), False
    mps = pt.max_polyp_stage()
    if mps >= 5:
        return ADV_POLYP, False
    if mps >= 1:
        return EARLY_POLYP, False
    return NORMAL, False
