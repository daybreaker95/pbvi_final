"""Manifest (size + MD5) of the artefacts that are excluded from the
repository for size: the per-chunk engine output under results/dp/runs/ and
the solved alpha-vector sets under results/dp/policies/. Lets anyone who
regenerates them check that they reproduced the paper's runs exactly (the
engine is deterministic given the chunk seed, and so is the solver).

python -m dp.manifest            -> results/dp/manifest.json
python -m dp.manifest --check    -> verify the files on disk against the manifest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

from .common import RES

DIRS = ['runs', 'policies']


def md5(path, block=1 << 22):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build():
    out = {}
    t0 = time.time()
    for d in DIRS:
        root = os.path.join(RES, d)
        for dp, _, files in os.walk(root):
            for fn in sorted(files):
                if not fn.endswith('.npz'):
                    continue
                p = os.path.join(dp, fn)
                rel = os.path.relpath(p, RES).replace('\\', '/')
                out[rel] = dict(bytes=os.path.getsize(p), md5=md5(p))
    print(f'{len(out)} files hashed ({time.time() - t0:.0f}s)')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    path = os.path.join(RES, 'manifest.json')
    if a.check:
        man = json.load(open(path))['files']
        bad = []
        for rel, info in man.items():
            p = os.path.join(RES, rel)
            if not os.path.exists(p) or os.path.getsize(p) != info['bytes'] or md5(p) != info['md5']:
                bad.append(rel)
        print(f'{len(man) - len(bad)} of {len(man)} files match; mismatches: {bad[:20]}')
        return
    files = build()
    total = sum(v['bytes'] for v in files.values())
    with open(path, 'w') as f:
        json.dump(dict(generated=time.strftime('%Y-%m-%d'), n_files=len(files), total_bytes=total, files=files), f, indent=0)
    print('saved', path, f'({len(files)} files, {total / 1e9:.2f} GB)')


if __name__ == '__main__':
    main()
