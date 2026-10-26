from __future__ import annotations
import sys as _runtime_sys
from pathlib import Path as _RuntimePath
_runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[2]))
from audit_runtime import PACKAGE_ROOT, WORKSPACE, record_path, require_records
if __name__ == "__main__":
    require_records()
import argparse
import json
import numpy as np
import pandas as pd
from s77_common import OUT, SEED, PERMUTATIONS, load_common, scope_mask, interval_any, lot_layout, rotated_indices, lot_bootstrap

def run(dest, smoke=False):
    m, states, events, target, roles, w, meta = load_common()
    w = w[w.returned & scope_mask(w.time, smoke)].copy().reset_index(drop=True)
    union = target.any(axis=1)
    layout = lot_layout(m.lot)
    null = {}
    rng = np.random.default_rng(SEED)
    for cap in ['full', 'cap30']:
        starts = w.start.to_numpy()
        ends = w.end.to_numpy() if cap == 'full' else np.minimum(w.end, w.start + 30).to_numpy()
        w['inside_' + cap] = interval_any(union, starts, ends)
        null[cap] = np.empty((len(w), PERMUTATIONS), bool)
    w['before'] = interval_any(union, w.start - 5, w.start)
    before_null = np.empty((len(w), PERMUTATIONS), bool)
    for p in range(PERMUTATIONS):
        shifted = union[rotated_indices(layout, rng)]
        before_null[:, p] = interval_any(shifted, w.start - 5, w.start)
        for cap in null:
            end = w.end.to_numpy() if cap == 'full' else np.minimum(w.end, w.start + 30).to_numpy()
            null[cap][:, p] = interval_any(shifted, w.start, end)
    for cap in null:
        w['chance_' + cap] = null[cap].mean(axis=1)
    w['chance_before'] = before_null.mean(axis=1)
    lot = m.lot.fillna('__missing__').astype(str).to_numpy()
    region = m.region.to_numpy()
    lot_run = np.cumsum(np.r_[True, lot[1:] != lot[:-1]])
    target_prefix = np.r_[0, np.cumsum(union)]
    auto_prefix = {loop: np.r_[0, np.cumsum(~states[name].eq(1.0).to_numpy())] for loop, name in meta['mode_names'].items()}
    start_pools = pd.DataFrame({'lot': lot, 'region': region}).groupby(['lot', 'region'], sort=False).indices
    for cap in ['full', 'cap30']:
        values = np.full(len(w), np.nan)
        prior_values = values.copy()
        pool_sizes = np.zeros(len(w), int)
        cache = {}
        for j, r in enumerate(w.itertuples()):
            length = int(r.length if cap == 'full' else min(r.length, 30))
            key = (r.loop, r.lot, r.region, length)
            if key not in cache:
                prefix = auto_prefix[r.loop]
                pos = np.asarray(start_pools.get((r.lot, r.region), []), dtype=int)
                pos = pos[pos + length <= len(m)]
                valid = (lot_run[pos] == lot_run[pos + length - 1]) & (prefix[pos + length] - prefix[pos] == 0)
                pos = pos[valid]
                cache[key] = pos
            pool = cache[key]
            pool_sizes[j] = len(pool)
            if len(pool):
                sampled = rng.choice(pool, PERMUTATIONS, replace=True)
                values[j] = (target_prefix[sampled + length] - target_prefix[sampled] > 0).mean()
                prior_values[j] = (target_prefix[sampled] - target_prefix[np.maximum(0, sampled - 5)] > 0).mean()
        w['auto_chance_' + cap] = values
        w['auto_pool_' + cap] = pool_sizes
        w['auto_before_' + cap] = prior_values
    rows = []
    groups = [('all_returned', w), ('main_no_cross', w[~w.cross_lot & w.known_lot]), ('no_quick', w[~w.cross_lot & w.known_lot & ~w.quick])]
    for population, frame in groups:
        expanded = pd.concat([frame, frame.assign(loop='pooled')], ignore_index=True)
        for (loop, reg), g in expanded.groupby(['loop', 'region']):
            for cap in ['full', 'cap30']:
                for side in ['before', 'inside']:
                    obs = g.before.to_numpy(float) if side == 'before' else g['inside_' + cap].to_numpy(float)
                    chance = g.chance_before.to_numpy() if side == 'before' else g['chance_' + cap].to_numpy()
                    diff = obs - chance
                    ci = lot_bootstrap(diff, g.lot)
                    control_column = 'auto_chance_' + cap if side == 'inside' else 'auto_before_' + cap
                    matched = g[control_column].notna().to_numpy()
                    control_delta = obs[matched] - g.loc[matched, control_column].to_numpy()
                    control_ci = lot_bootstrap(control_delta, g.loc[matched, 'lot'])
                    rows.append({'population': population, 'loop': loop, 'region': reg, 'cap': cap, 'side': side, 'windows': len(g), 'lots': g.lot.nunique(), 'p_observed': float(obs.mean()), 'p_observed_matched': float(obs[matched].mean()) if matched.any() else None, 'matched_control_delta': float(control_delta.mean()) if len(control_delta) else None, 'matched_ci_low': control_ci[0], 'matched_ci_high': control_ci[1], 'p_chance': float(chance.mean()), 'delta': float(diff.mean()), 'ci_low': ci[0], 'ci_high': ci[1], 'matched_controls': int(g['auto_chance_' + cap].notna().sum()), 'p_auto_control': float(g['auto_chance_' + cap if side == 'inside' else 'auto_before_' + cap].mean()) if g['auto_chance_' + cap].notna().any() else None})
    w.to_parquet(dest / str(record_path('q1_windows_private_parquet')), index=False)
    pd.DataFrame(rows).to_csv(dest / 'q1_summary.csv', index=False)
    summary = {'returned_windows': len(w), 'normal_returned': int(w.region.eq('정상 생산').sum()), 'cross_lot_windows': int(w.cross_lot.sum()), 'permutations': PERMUTATIONS, 'main_windows': int((~w.cross_lot & w.known_lot).sum()), 'target_roles': roles}
    (dest / 'q1_manifest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({'Q1': summary}, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    d = OUT / 'smoke' if a.smoke else OUT
    d.mkdir(parents=True, exist_ok=True)
    run(d, a.smoke)
