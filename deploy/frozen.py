# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen pi_0 dirs (e.g. logs/frozen/pi0_rma_2026-11-08/) carry a SHA256SUMS manifest (``<sha256>  ./<relpath>``
per line). Supervisor 11-08: verify it at load, and load the RMA pi_0 only from such a dir. No Isaac Sim needed."""

from __future__ import annotations

import hashlib
import os

MANIFEST = "SHA256SUMS"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_manifest(path: str) -> str | None:
    """The SHA256SUMS of the nearest enclosing dir (the file's dir or its parent: params/env.yaml sits one down)."""
    d = os.path.dirname(os.path.abspath(path))
    for cand in (d, os.path.dirname(d)):
        m = os.path.join(cand, MANIFEST)
        if os.path.exists(m):
            return m
    return None


def verify_listed(path: str) -> bool:
    """If ``path`` lies in a dir with a SHA256SUMS: raise unless it is listed and its hash matches. Returns True when
    verified, False when there is no manifest."""
    m = find_manifest(path)
    if m is None:
        return False
    root = os.path.dirname(m)
    rel = "./" + os.path.relpath(os.path.abspath(path), root)
    sums = {}
    for line in open(m):
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            sums[parts[1].strip()] = parts[0]
    if rel not in sums:
        raise RuntimeError(f"{path} is in a frozen dir ({m}) but not listed in its {MANIFEST}")
    got = sha256(path)
    if got != sums[rel]:
        raise RuntimeError(f"{path}: sha256 {got[:12]}... does not match {MANIFEST} ({sums[rel][:12]}...)")
    print(f"[frozen] verified {rel} against {m} (sha256 {got[:12]}...)", flush=True)
    return True
