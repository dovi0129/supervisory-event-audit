from __future__ import annotations
import sys as _runtime_sys
from pathlib import Path as _RuntimePath
_runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[2]))
from audit_runtime import PACKAGE_ROOT, WORKSPACE, record_path, require_records
if __name__ == "__main__":
    require_records()
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = WORKSPACE
OUT = ROOT / 'artifacts/stage1_audit'
SEED = 7702
PERMUTATIONS = 200
BOOTSTRAPS = 2000

def load_common():
    meta = json.loads((ROOT / 'artifacts/intervention/data/metadata.json').read_text())
    m = pd.read_parquet(ROOT / str(record_path('artifacts_private_s68_minutes_parquet'))).set_index('time')
    states = pd.read_parquet(ROOT / str(record_path('artifacts_intervention_data_modes_private_parquet'))).reindex(m.index)
    events = pd.read_parquet(ROOT / 'artifacts/operator_policy/events.parquet', columns=['time', 'loop', 'kind', 'shape', 'size', 'dur_min', 'prev', 'new'], filters=[('class', '==', '운전자 후보'), ('kind', 'in', ['target', 'mode', 'op', 'dcs'])]).sort_values(['time', 'loop'])
    events['position'] = m.index.get_indexer(events.time)
    assert (events.position >= 0).all()
    roles = sorted(events.loc[events.kind.eq('target'), 'loop'].unique())
    target = np.zeros((len(m), len(roles)), bool)
    for j, role in enumerate(roles):
        target[events.loc[events.loop.eq(role) & events.kind.eq('target'), 'position'], j] = True
    windows = []
    lot = m.lot.fillna('__missing__').astype(str).to_numpy()
    run = np.cumsum(np.r_[True, lot[1:] != lot[:-1]])
    for loop, g in events[events.kind.eq('mode')].groupby('loop'):
        rows = list(g.sort_values('position').itertuples())
        for i, r in enumerate(rows):
            if r.shape != 'aut→man':
                continue
            returned = i + 1 < len(rows) and rows[i + 1].shape == 'man→aut'
            start = int(r.position)
            end = int(rows[i + 1].position) if returned else len(m)
            windows.append({'loop': loop, 'start': start, 'end': end, 'length': end - start, 'returned': bool(returned), 'quick': bool(end - start <= 2), 'lot': lot[start], 'region': str(m.region.iloc[start]), 'cross_lot': bool(run[start] != run[max(start, end - 1)]), 'known_lot': lot[start] != '__missing__', 'time': m.index[start]})
    return (m, states, events, target, roles, pd.DataFrame(windows), meta)

def scope_mask(times, smoke):
    return np.asarray(times < pd.Timestamp('2022-03-01')) if smoke else np.ones(len(times), bool)

def interval_any(flag, starts, ends):
    p = np.r_[0, np.cumsum(np.asarray(flag, bool))]
    starts = np.maximum(0, np.asarray(starts, int))
    ends = np.minimum(len(flag), np.asarray(ends, int))
    return p[ends] - p[starts] > 0

def lot_layout(lots):
    codes, _ = pd.factorize(pd.Series(lots).fillna('__missing__'), sort=True)
    order = np.argsort(codes, kind='stable')
    counts = np.bincount(codes)
    begin = np.r_[0, np.cumsum(counts)[:-1]]
    base = np.repeat(begin, counts)
    rank = np.arange(len(codes)) - base
    group = np.repeat(np.arange(len(counts)), counts)
    return (order, counts, base, rank, group)

def rotated_indices(layout, rng):
    order, counts, base, rank, group = layout
    offsets = np.array([rng.integers(0, n) for n in counts])
    source = order[base + (rank + offsets[group]) % counts[group]]
    result = np.empty(len(order), int)
    result[order] = source
    return result

def lot_bootstrap(values, lots, seed=SEED):
    d = pd.DataFrame({'value': np.asarray(values, float), 'lot': np.asarray(lots)}).dropna()
    g = d.groupby('lot').value.agg(['sum', 'count'])
    if len(g) < 2:
        return [None, None]
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(g), size=(BOOTSTRAPS, len(g)))
    v = g['sum'].to_numpy()[ix].sum(axis=1) / g['count'].to_numpy()[ix].sum(axis=1)
    return np.quantile(v, [0.025, 0.975]).tolist()

def stage_bin(elapsed):
    return np.searchsorted([5, 15, 30, 60, 120], elapsed, side='right')

def exchange_indices(frame, rng, ignore_stage=False):
    keys = ['pair'] if ignore_stage else ['pair', 'step', 'elapsed_bin']
    donor = np.arange(len(frame))
    changed = np.zeros(len(frame), bool)
    for _, indices in frame.groupby(keys, dropna=False, sort=False).indices.items():
        ix = np.asarray(indices)
        episodes = frame.iloc[ix].episode_id.to_numpy()
        groups = [ix[episodes == ep] for ep in np.unique(episodes)]
        if len(groups) < 2:
            continue
        order = rng.permutation(len(groups))
        offset = int(rng.integers(1, len(groups)))
        for pos, g in enumerate(order):
            recipient = groups[g]
            source = groups[order[(pos + offset) % len(order)]]
            donor[recipient] = rng.choice(source, size=len(recipient), replace=True)
            changed[recipient] = True
    assert np.all(frame.episode_id.to_numpy()[donor[changed]] != frame.episode_id.to_numpy()[changed])
    return (donor, changed)

def kaplan_meier(duration, event):
    duration = np.asarray(duration, float)
    event = np.asarray(event, bool)
    survival = 1.0
    rows = []
    median = None
    for t in np.unique(duration):
        risk = int((duration >= t).sum())
        d = int(((duration == t) & event).sum())
        c = int(((duration == t) & ~event).sum())
        if risk and d:
            survival *= 1 - d / risk
        rows.append({'time': float(t), 'at_risk': risk, 'events': d, 'censored': c, 'survival': survival})
        if median is None and survival <= 0.5:
            median = float(t)
    return (rows, median)
