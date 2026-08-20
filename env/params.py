"""
params.py -- load a CMOST settings file and build the exact parameter bundle
used by the individual-level engine.

The bundle is produced by calculate_sub.prepare_parameters(), guaranteeing that
the individual engine is parameter-identical to the validated population model.
"""

from __future__ import annotations

import os
import sys
import copy
import importlib.util

import numpy as np

_ENV_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.abspath(os.path.join(_ENV_DIR, '..', 'fast_engine_deps'))  # vendored
                            # calculate_sub.py + Evaluation.py + settings/CMOST13.py +
                            # NumberCrunching_100000_fastengine.py (renamed on import to
                            # avoid a sys.modules collision with cmost_engine's own,
                            # differently-versioned NumberCrunching_100000.py when both
                            # are imported in the same process -- see tests/*_4way_eval.py)
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)


def load_settings(name: str = 'CMOST13') -> dict:
    """Load settings dict from python/settings/<name>.py (deep-copied)."""
    settings_path = os.path.join(_PKG_DIR, 'settings', f'{name}.py')
    spec = importlib.util.spec_from_file_location(f'_settings_{name}', settings_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return copy.deepcopy(mod.settings)


def build_params(settings_name: str = 'CMOST13',
                 n_patients: int = 1,
                 seed: int | None = None) -> dict:
    """Return the CMOST parameter bundle for the individual engine.

    Parameters
    ----------
    settings_name : str
        Which settings file to load (default 'CMOST13').
    n_patients : int
        Sizes the individual_risk / gender pools inside prepare_parameters.
        The engine samples from these pools, so a modest pool (e.g. 500) is
        plenty; this does NOT limit how many patients you can simulate.
    seed : int, optional
        Seeds numpy's global RNG *for the parameter preparation only*
        (mortality-matrix shuffling, risk-pool sampling).
    """
    from calculate_sub import prepare_parameters

    if seed is not None:
        np.random.seed(seed)
    settings = load_settings(settings_name)
    settings['Number_patients'] = max(int(n_patients), 1)
    # Natural-history estimation must not have the model's own screening on.
    if 'Screening' in settings and isinstance(settings['Screening'], dict):
        settings['Screening']['Mode'] = 'off'
    settings['Polyp_Surveillance'] = 'off'
    settings['Cancer_Surveillance'] = 'off'
    return prepare_parameters({'Variables': settings})
