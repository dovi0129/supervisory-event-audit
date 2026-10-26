from __future__ import annotations
import sys as _runtime_sys
from pathlib import Path as _RuntimePath
_runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[2]))
from audit_runtime import PACKAGE_ROOT, WORKSPACE, record_path, require_records
if __name__ == "__main__":
    require_records()
import json
import os
import sys
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np
import pandas as pd
from s77_common import ROOT, OUT, PERMUTATIONS, lot_bootstrap, interval_any, load_common
from s77_q2_fit import splits as q2_splits
DEST = OUT / 's89'
S89_SEED = 8901

def transition_anchor(starts, centers):
    centers = np.asarray(centers, int)
    starts = np.asarray(starts, int)
    if not len(centers):
        return np.full(len(starts), -1, int)
    right = np.searchsorted(centers, starts, side='left')
    left = right - 1
    lc, rc = (np.clip(left, 0, len(centers) - 1), np.clip(right, 0, len(centers) - 1))
    ld, rd = (starts - centers[lc], starts - centers[rc])
    ls = np.where((left >= 0) & (ld >= -30) & (ld <= 60), np.abs(ld), np.inf)
    rs = np.where((right < len(centers)) & (rd >= -30) & (rd <= 60), np.abs(rd), np.inf)
    result = np.where(ls <= rs, left, right)
    return np.where(np.isfinite(np.minimum(ls, rs)), result, -1).astype(int)

def contiguous_true_blocks(mask):
    mask = np.asarray(mask, bool)
    edge = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)))

def rotate_transition_only(flag, blocks, rng):
    result = np.asarray(flag, bool).copy()
    for start, stop in blocks:
        result[start:stop] = np.roll(flag[start:stop], int(rng.integers(stop - start)))
    return result

def recent_target_signal(features, lookback):
    columns = ['moisture_target_age', 'size_moisture_target_age', 'weight_target_age']
    return features[columns].ge(0).mul(features[columns].lt(lookback)).any(axis=1).to_numpy()

def matched_donor_pool(start, length, anchor, centers, region, lot_run):
    offset = start - centers[anchor]
    offsets = offset + np.arange(-5, 6)
    candidates = centers[:, None] + offsets[None, :]
    valid = (offsets[None, :] >= -30) & (offsets[None, :] <= 60)
    valid = np.broadcast_to(valid, candidates.shape).copy()
    valid &= (candidates >= 5) & (candidates + length <= len(region))
    clipped = np.clip(candidates, 0, len(region) - 1)
    ends = np.clip(candidates + length - 1, 0, len(region) - 1)
    valid &= region[clipped] == '지종 전환'
    valid &= lot_run[clipped] == lot_run[ends]
    valid[anchor] = False
    rows, cols = np.where(valid)
    if len(rows):
        assigned = transition_anchor(candidates[rows, cols], centers)
        valid[rows[assigned != rows], cols[assigned != rows]] = False
    return (candidates, valid)

def q1_elapsed():
    DEST.mkdir(parents=True, exist_ok=True)
    m, _, _, targets, _, _, _ = load_common()
    union = targets.any(axis=1)
    prefix = np.r_[0, np.cumsum(union)]
    w = pd.read_parquet(OUT / str(record_path('q1_windows_private_parquet')))
    w = w[w.returned & ~w.cross_lot & w.known_lot & w.region.eq('지종 전환')].copy().reset_index(drop=True)
    episodes = pd.read_parquet(ROOT / 'artifacts/transition/episodes.parquet')
    centers = np.sort(episodes.loc[episodes.cohort.eq('transition'), 'start_position'].to_numpy(int))
    w['anchor'] = transition_anchor(w.start, centers)
    assert w.anchor.ge(0).all()
    w['elapsed'] = w.start.to_numpy() - centers[w.anchor.to_numpy()]
    lot = m.lot.fillna('__missing__').astype(str).to_numpy()
    lot_run = np.cumsum(np.r_[True, lot[1:] != lot[:-1]])
    region = m.region.to_numpy()
    rng = np.random.default_rng(S89_SEED)
    windows = []
    for cap in ['full', 'cap30']:
        for j, r in enumerate(w.itertuples()):
            length = int(r.length if cap == 'full' else min(r.length, 30))
            candidates, valid = matched_donor_pool(r.start, length, r.anchor, centers, region, lot_run)
            donor_ids = np.flatnonzero(valid.any(axis=1))
            chance_inside = chance_before = np.nan
            if len(donor_ids):
                selected = rng.choice(donor_ids, PERMUTATIONS, replace=True)
                pos = np.array([rng.choice(candidates[k, valid[k]]) for k in selected], int)
                assert np.all(selected != r.anchor)
                assert np.all(np.abs(pos - centers[selected] - r.elapsed) <= 5)
                chance_inside = float((prefix[pos + length] - prefix[pos] > 0).mean())
                chance_before = float((prefix[pos] - prefix[pos - 5] > 0).mean())
            windows.append({'window_row': j, 'loop': r.loop, 'lot': r.lot, 'cap': cap, 'elapsed': r.elapsed, 'elapsed_side': 'before_key' if r.elapsed < 0 else 'after_key', 'length': length, 'donor_episodes': len(donor_ids), 'donor_windows': int(valid.sum()), 'obs_inside': bool(prefix[r.start + length] - prefix[r.start] > 0), 'obs_before': bool(prefix[r.start] - prefix[max(0, r.start - 5)] > 0), 'chance_inside': chance_inside, 'chance_before': chance_before, 'lot_chance_inside': getattr(r, 'chance_' + cap), 'lot_chance_before': r.chance_before})
    matched = pd.DataFrame(windows)
    blocks = contiguous_true_blocks(region == '지종 전환')
    rotation = {cap: np.zeros((len(w), PERMUTATIONS), bool) for cap in ['full', 'cap30', 'before']}
    rng = np.random.default_rng(S89_SEED + 1)
    for repeat in range(PERMUTATIONS):
        shifted = rotate_transition_only(union, blocks, rng)
        rotation['before'][:, repeat] = interval_any(shifted, w.start - 5, w.start)
        rotation['full'][:, repeat] = interval_any(shifted, w.start, w.end)
        rotation['cap30'][:, repeat] = interval_any(shifted, w.start, np.minimum(w.end, w.start + 30))
    rows = []
    for cap in ['full', 'cap30']:
        part = matched[matched.cap.eq(cap)].copy()
        for side in ['before', 'inside']:
            key = 'before' if side == 'before' else cap
            part['rotated_chance'] = rotation[key].mean(axis=1)
            for phase in ['all', 'before_key', 'after_key']:
                phase_group = part if phase == 'all' else part[part.elapsed_side.eq(phase)]
                combined = pd.concat([phase_group, phase_group.assign(loop='pooled')], ignore_index=True)
                for loop, g in combined.groupby('loop'):
                    for method in ['other_episode_elapsed', 'transition_block_rotation']:
                        col = 'chance_' + side if method == 'other_episode_elapsed' else 'rotated_chance'
                        supported = g[col].notna()
                        s = g[supported]
                        if len(s):
                            obs = s['obs_' + side].to_numpy(float)
                            chance = s[col].to_numpy()
                            delta = obs - chance
                            ci = lot_bootstrap(delta, s.lot)
                            old = obs - s['lot_chance_' + side].to_numpy()
                            rows.append({'method': method, 'cap': cap, 'side': side, 'elapsed_phase': phase, 'loop': loop, 'candidate_windows': len(g), 'matched_windows': len(s), 'unmatched_windows': int((~supported).sum()), 'lots': s.lot.nunique(), 'p_observed': float(obs.mean()), 'p_reference': float(chance.mean()), 'delta': float(delta.mean()), 'ci_low': ci[0], 'ci_high': ci[1], 'original_lot_delta_on_same_windows': float(old.mean()), 'median_donor_episodes': float(s.donor_episodes.median()) if method == 'other_episode_elapsed' else None})
    matched.to_parquet(DEST / str(record_path('q1_elapsed_windows_private_parquet')), index=False)
    pd.DataFrame(rows).to_csv(DEST / 'q1_elapsed_controls.csv', index=False)
    manifest = {'original_transition_windows': len(w), 'key_changes': len(centers), 'negative_elapsed_windows': int(w.elapsed.lt(0).sum()), 'permutations': PERMUTATIONS, 'transition_blocks': len(blocks), 'matching_seed': S89_SEED, 'anchor': 'nearest compatible key change in [-30,+60]; earlier center on tie', 'donor': 'different episode, same signed elapsed +/-5, same length, one LOT, unrestricted mode; episode uniformly then admissible offset uniformly', 'rotation': 'joint target union shifted within contiguous transition-stratum blocks; outside values unchanged', 'bootstrap': '200-draw reference estimated once, then fixed while LOT-resampling observed-minus-reference'}
    (DEST / 'q1_elapsed_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest

def recent_target_rules():
    sys.path.insert(0, str(ROOT / 'scripts/s70'))
    from s70_fit import metrics, rising_alarms, SEED as S70_SEED
    from s70_features import future_labels
    data = ROOT / 'artifacts/intervention/data'
    old = ROOT / 'artifacts/intervention/full'
    meta = json.loads((data / 'metadata.json').read_text())
    features = pd.read_parquet(data / str(record_path('features_private_parquet')), columns=['moisture_target_age', 'size_moisture_target_age', 'weight_target_age'])
    minutes = pd.read_parquet(data / str(record_path('minutes_private_parquet')))
    modes = pd.read_parquet(data / str(record_path('modes_private_parquet')))
    events = pd.read_parquet(data / str(record_path('events_private_parquet')))
    n = len(minutes)
    signals = {k: recent_target_signal(features, k) for k in [3, 5]}
    rows = []
    checks = []
    for j, loop in enumerate(meta['targets']):
        state = modes[meta['mode_names'][loop]]
        eligible = state.eq(1.0).to_numpy()
        episode = state.ne(state.shift()).cumsum().to_numpy()
        ev = events[events.loop.eq(loop) & events['shape'].eq('aut→man')]
        wanted = ~ev.flag_brk_near.fillna(True) & ev.return_observed & ~ev.quick & ~ev.short_previous
        pos = features.index.get_indexer(ev.loc[wanted, 'time'])
        assert (pos >= 0).all()
        accepted = np.zeros(n, bool)
        accepted[pos] = True
        y = future_labels(accepted, 5)
        alarms = {k: rising_alarms(signal.astype(float), 0.5, eligible, 5, episode) for k, signal in signals.items()}
        original_job = json.loads((old / 'jobs' / f'{j}_5_main_aut→man.json').read_text())
        for fold in ['time', 'lot0', 'lot1', 'lot2', 'lot3', 'lot4']:
            p = pd.read_parquet(old / str(record_path('predictions_private')) / f'{j}_5_main_aut→man' / f'{fold}.parquet')
            domain = np.zeros(n, bool)
            domain[p.position] = True
            assert np.array_equal(y[p.position], p.y.to_numpy())
            for region in ['all', 'normal', 'transition', 'other']:
                mask = domain & (True if region == 'all' else minutes.region.eq(region).to_numpy())
                old_alarm = np.zeros(n, bool)
                old_alarm[p.position] = p.tree_alarm.to_numpy(bool)
                reference = metrics(y, np.zeros(n), old_alarm, pos, mask, 5, np.random.default_rng(S70_SEED))
                old_row = next((z for z in original_job['folds'] if z['fold'] == fold and z['model'] == 'tree' and (z['region'] == region)))
                assert reference['events'] == old_row['events'] and reference['event_hits'] == old_row['event_hits']
                checks.append((loop, fold, region))
                for k, signal in signals.items():
                    alarm = alarms[k]
                    r = metrics(y, signal.astype(float), alarm, pos, mask, 5, np.random.default_rng(S70_SEED))
                    r['score_positive_minutes'] = int((mask & signal).sum())
                    r['score_true_positive_minutes'] = int((mask & signal & y).sum())
                    rows.append({'loop': loop, 'fold': fold, 'split': 'time' if fold == 'time' else 'group', 'model': f'recent_target_{k}min', 'lookback_minutes': k, 'prediction_horizon': 5, 'region': region, **r})
    full = pd.DataFrame(rows)
    simple = full.drop(columns=['lead_minutes', 'lead_p10_p50_p90', 'calibration'])
    simple.to_csv(DEST / 'recent_target_rule_fold_metrics.csv', index=False)
    aggregated = []
    expanded = pd.concat([full, full.assign(loop='pooled')], ignore_index=True)
    for keys, g in expanded.groupby(['loop', 'split', 'lookback_minutes', 'prediction_horizon', 'region']):
        r = dict(zip(['loop', 'split', 'lookback_minutes', 'prediction_horizon', 'region'], keys))
        for col in ['n_minutes', 'positive_minutes', 'alarms', 'true_alarm_minutes', 'events', 'event_hits', 'candidate_events', 'score_positive_minutes', 'score_true_positive_minutes']:
            r[col] = int(g[col].sum())
        r['event_recall'] = r['event_hits'] / r['events'] if r['events'] else np.nan
        r['minute_precision'] = r['true_alarm_minutes'] / r['alarms'] if r['alarms'] else np.nan
        ok = g.events.gt(0) & g.chance_recall.notna()
        r['chance_recall'] = float(np.average(g.loc[ok, 'chance_recall'], weights=g.loc[ok, 'events'])) if ok.any() else np.nan
        r['chance_excess'] = (r['event_recall'] - r['chance_recall']) / (1 - r['chance_recall']) if r['chance_recall'] < 1 else np.nan
        if r['positive_minutes']:
            recall_high = r['score_true_positive_minutes'] / r['positive_minutes']
            precision_high = r['score_true_positive_minutes'] / r['score_positive_minutes'] if r['score_positive_minutes'] else 0.0
            r['pr_auc'] = recall_high * precision_high + (1 - recall_high) * r['positive_minutes'] / r['n_minutes']
        else:
            r['pr_auc'] = np.nan
        r['model'] = f"recent_target_{r['lookback_minutes']}min"
        leads = [v for row in g.lead_minutes for v in row]
        for name, value in zip(['lead_p10', 'lead_p50', 'lead_p90'], np.quantile(leads, [0.1, 0.5, 0.9]) if leads else [np.nan] * 3):
            r[name] = float(value)
        aggregated.append(r)
    pd.DataFrame(aggregated).to_csv(DEST / 'recent_target_rule_metrics.csv', index=False)
    manifest = {'definition': 'any of the three existing raw-target backward-difference ages in [0,k), i.e. observed minute bins [t-k+1,t]', 'lookback_minutes': [3, 5], 'horizon': 5, 'alarm': 'S70 rising edge, same-auto-spell five-minute suppression', 'population': 'same saved S70 k=5 main aut-to-man prediction rows; no additional tuning', 'verified_saved_tree_rows': len(checks), 'metric_function': 'scripts/s70/s70_fit.py:metrics'}
    (DEST / 'recent_target_rule_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest

def reviewer_descriptives():
    _, _, events, _, _, _, _ = load_common()
    ep = pd.read_parquet(ROOT / 'artifacts/transition/episodes.parquet')
    ep = ep[ep.cohort.eq('transition')].copy()
    event_positions = np.unique(events.position.to_numpy(int))
    ep['pre30_bundles'] = np.searchsorted(event_positions, ep.start_position, side='left') - np.searchsorted(event_positions, ep.start_position - 30, side='left')
    episode_table = pd.read_parquet(OUT / str(record_path('q2_episodes_private_parquet')))
    _, train, valid, test = next((s for s in q2_splits(episode_table, False) if s[0] == 'time'))
    seen = set(episode_table[episode_table.episode_id.isin(train + valid)].pair)
    ep['time_test'] = ep.episode_id.isin(test)
    ep['seen'] = ep.pair.isin(seen)
    dist = []
    for name, g in [('all', ep), ('time_test', ep[ep.time_test]), ('time_test_seen', ep[ep.time_test & ep.seen]), ('time_test_unseen', ep[ep.time_test & ~ep.seen])]:
        x = g.pre30_bundles.to_numpy()
        dist.append({'stratum': name, 'episodes': len(g), 'with_pre30_bundle': int((x > 0).sum()), 'fraction_with_pre30': float((x > 0).mean()), 'mean': float(x.mean()), **dict(zip(['min', 'p10', 'p25', 'median', 'p75', 'p90', 'max'], np.quantile(x, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]).astype(float)))})
    ep[['episode_id', 'time_test', 'seen', 'pre30_bundles']].to_parquet(DEST / str(record_path('pre30_bundle_counts_private_parquet')), index=False)
    pd.DataFrame(dist).to_csv(DEST / 'pre30_bundle_distribution.csv', index=False)
    windows = pd.read_parquet(OUT / str(record_path('q1_windows_private_parquet')))
    rows = []
    for region in ['정상 생산', '지종 전환']:
        g = windows[windows.returned & windows.known_lot & ~windows.cross_lot & windows.region.eq(region)]
        for cap in ['full', 'cap30']:
            for side in ['before', 'inside']:
                refcol = 'auto_before_' + cap if side == 'before' else 'auto_chance_' + cap
                obscol = 'before' if side == 'before' else 'inside_' + cap
                valid = g[refcol].notna()
                s = g[valid]
                d = s[obscol].astype(float) - s[refcol]
                ci = lot_bootstrap(d, s.lot)
                rows.append({'region': region, 'cap': cap, 'side': side, 'windows': len(s), 'observed': float(s[obscol].mean()), 'automatic_reference': float(s[refcol].mean()), 'delta': float(d.mean()), 'ci_low': ci[0], 'ci_high': ci[1]})
    paired = pd.DataFrame(rows)
    paired.to_csv(DEST / 'q1_automatic_paired.csv', index=False)
    normal = paired[(paired.region == '정상 생산') & (paired.cap == 'full') & (paired.side == 'before')].iloc[0]
    assert np.isclose(normal.delta, 0.1937, atol=5e-05)
    return {'pre30_ep': len(ep), 'normal_before_auto_delta': float(normal.delta), 'bootstrap_reference_recomputed': False, 'original_code': ['scripts/s77/s77_q1.py:39-51', 'scripts/s77/s77_q1.py:118-119', 'scripts/s77/s77_common.py:104-115'], 'interpretation': '200-shift mean fixed per window before 2000 LOT bootstrap draws; bootstrap does not regenerate temporal shifts'}

def run():
    DEST.mkdir(parents=True, exist_ok=True)
    result = {'elapsed': q1_elapsed(), 'recent_targets': recent_target_rules(), 'descriptives': reviewer_descriptives()}
    (DEST / 'controls_manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({'controls_done': True}), flush=True)
    return result
if __name__ == '__main__':
    run()
