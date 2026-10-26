from __future__ import annotations
import sys as _runtime_sys
from pathlib import Path as _RuntimePath
_runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[2]))
from audit_runtime import PACKAGE_ROOT, WORKSPACE, record_path, require_records
if __name__ == "__main__":
    require_records()
import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT = WORKSPACE
OUT = ROOT / 'artifacts/transition'
sys.path.insert(0, str(ROOT / 'scripts/s70'))
from s70_features import SIGNALS
BANDS = {'primary': (2.0, 0.3), 'tight': (1.0, 0.2)}
SEED = 7401

def first_confirmation(within, consecutive=15):
    a = np.asarray(within, bool)
    if len(a) < consecutive:
        return None
    total = np.r_[0, np.cumsum(a)]
    good = np.flatnonzero(total[consecutive:] - total[:-consecutive] == consecutive)
    return int(good[0] + consecutive - 1) if len(good) else None

def endpoint(start, stop, within):
    confirmed = first_confirmation(within[start:stop])
    return (start + confirmed + 1, 'stable_band', start + confirmed - 14) if confirmed is not None else (stop, None, None)

@lru_cache(maxsize=1000000)
def edit_distance(a, b):
    if len(a) < len(b):
        return edit_distance(b, a)
    previous = list(range(len(b) + 1))
    for i, ai in enumerate(a, 1):
        current = [i]
        for j, bj in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (ai != bj)))
        previous = current
    return previous[-1]

def variance_parts(values, groups):
    values, groups = (np.asarray(values, float), np.asarray(groups))
    ok = np.isfinite(values)
    values, groups = (values[ok], groups[ok])
    if not len(values):
        return {'n': 0, 'groups': 0, 'within': None, 'between': None, 'ratio': None}
    mean = values.mean()
    within = between = 0.0
    for group in np.unique(groups):
        v = values[groups == group]
        within += np.sum((v - v.mean()) ** 2)
        between += len(v) * (v.mean() - mean) ** 2
    within, between = (float(within / len(values)), float(between / len(values)))
    return {'n': len(values), 'groups': len(np.unique(groups)), 'within': within, 'between': between, 'ratio': within / between if between > 1e-15 else None}

def load(month=None):
    cols = json.loads((ROOT / str(record_path('artifacts_private_s68_columns_private_json'))).read_text())
    maps = json.loads((ROOT / str(record_path('artifacts_operator_policy_tag_map_private_json'))).read_text())
    raw = pd.read_parquet(ROOT / str(record_path('artifacts_private_s68_cache_parquet')), columns=['time', '지종', '평량', '작업자'] + list(SIGNALS) + cols['kw']).set_index('time')
    features = pd.read_parquet(ROOT / str(record_path('artifacts_intervention_data_features_private_parquet')), columns=['weight_error'])
    minutes = pd.read_parquet(ROOT / str(record_path('artifacts_private_s68_minutes_parquet'))).set_index('time').reindex(features.index)
    raw = raw.reindex(features.index)
    grade, weight = (raw['지종'].ffill(limit=30), raw['평량'].ffill(limit=30))
    key = (grade.astype(str) + '|' + weight.astype(str)).where(grade.notna() & weight.notna())
    unique = sorted(key.dropna().unique())
    keymap = {k: f'grade {i + 1}' for i, k in enumerate(unique)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / str(record_path('grade_key_map_private_json'))).write_text(json.dumps(keymap, ensure_ascii=False, indent=1))
    grade_anon = key.map(keymap)
    changes = np.flatnonzero((key.notna() & key.shift().notna() & key.ne(key.shift())).to_numpy())
    n_global_changes = len(changes)
    data = raw[list(SIGNALS)].rename(columns=SIGNALS).apply(pd.to_numeric, errors='coerce')
    data['weight_error_raw'] = data.weight - data.weight_target
    data['moisture_error_raw'] = data.moisture - data.moisture_target
    data['weight_error_centered_s70'] = features.weight_error
    power = raw[cols['kw']].apply(pd.to_numeric, errors='coerce')
    data['power_sum'] = power.sum(axis=1, min_count=len(cols['kw']))
    data['power_channel_fraction'] = power.notna().mean(axis=1)
    data['grade'] = grade_anon
    data['crew'] = raw['작업자'].ffill(limit=30).map(maps['crews']).fillna('unknown')
    data['valid'] = minutes.valid.fillna(False).astype(bool)
    data['brk'] = minutes.brk.fillna(False).astype(bool)
    data['normal'] = minutes.region.eq('정상 생산')
    data['gc'] = minutes.gc.fillna(False).astype(bool)
    events = pd.read_parquet(ROOT / 'artifacts/operator_policy/events.parquet', columns=['time', 'loop', 'kind', 'shape', 'size'], filters=[('class', '==', '운전자 후보'), ('kind', 'in', ['target', 'mode', 'op', 'dcs'])]).sort_values(['time', 'kind', 'loop']).reset_index(drop=True)
    roles = sorted(events.loc[events.kind.eq('target'), 'loop'].unique())
    if month:
        mask = data.index.strftime('%Y-%m') == month
        data = data.loc[mask]
        changes = data.index.get_indexer(raw.index[changes])
        changes = changes[changes >= 0]
        events = events[events.time.dt.strftime('%Y-%m').eq(month)].reset_index(drop=True)
    positions = data.index.get_indexer(events.time)
    assert (positions >= 0).all(), 'Event/minute alignment'
    events['position'] = positions
    return (data, events, changes, roles, n_global_changes)

def episode_row(data, events, start, stop, reason, entry, episode_id, cohort, band):
    idx = data.index
    g = data.iloc[start:stop]
    actions = events[(events.position >= start) & (events.position < stop)].copy()
    actions['episode_id'] = episode_id
    actions['relative_min'] = actions.position - start
    actions['direction'] = np.where(actions['size'].gt(0), 'increase', np.where(actions['size'].lt(0), 'decrease', 'zero_or_missing'))
    actions.loc[actions.kind.eq('mode'), 'direction'] = actions.loc[actions.kind.eq('mode'), 'shape']
    actions = actions.rename(columns={'size': 'delta'})
    quality_ok = g.valid & g.weight_error_raw.notna() & g.moisture_error_raw.notna()
    bw, moisture = BANDS[band]
    inside = quality_ok & g.weight_error_raw.abs().le(bw) & g.moisture_error_raw.abs().le(moisture)
    speeds = g.speed.dropna()
    baseline = data.speed.iloc[start - 1] if start else np.nan
    previous = data.grade.iloc[start - 1] if start else 'unknown'
    row = {'episode_id': episode_id, 'cohort': cohort, 'band': band, 'grade_from': previous, 'grade_to': data.grade.iloc[start], 'pair': str(previous) + ' → ' + str(data.grade.iloc[start]), 'start': idx[start], 'end_exclusive': idx[stop] if stop < len(data) else idx[-1] + pd.Timedelta(minutes=1), 'start_position': start, 'end_position_exclusive': stop, 'end_reason': reason, 'duration_min': stop - start, 'band_confirmed_min': stop - start if reason == 'stable_band' else None, 'band_entry_min': entry - start if entry is not None else None, 'quality_observed_min': int(quality_ok.sum()), 'quality_invalid_min': int((~quality_ok).sum()), 'outside_band_min': int((quality_ok & ~inside).sum()), 'weight_iae_g_m2_min': float(g.loc[quality_ok, 'weight_error_raw'].abs().sum()) if quality_ok.any() else None, 'moisture_iae_pp_min': float(g.loc[quality_ok, 'moisture_error_raw'].abs().sum()) if quality_ok.any() else None, 'speed_change': float(speeds.iloc[-1] - baseline) if len(speeds) and pd.notna(baseline) else None, 'speed_missing_fraction': float(g.speed.isna().mean()), 'power_observed_mean': float(g.power_sum.mean()) if g.power_sum.notna().any() else None, 'power_observed_integral_unit_min': float(g.power_sum.sum()) if g.power_sum.notna().any() else None, 'power_missing_fraction': float(g.power_sum.isna().mean()), 'power_observed_minutes': int(g.power_sum.notna().sum()), 'power_channel_observed_fraction': float(g.power_channel_fraction.mean()), 'break_included': bool(g.brk.any()), 'break_minutes': int(g.brk.sum()), 'crew_start': str(data.crew.iloc[start]), 'crew_members': '+'.join(sorted(g.crew.unique())), 'first_action_min': int(actions.relative_min.min()) if len(actions) else None, 'last_action_min': int(actions.relative_min.max()) if len(actions) else None, 'action_count': len(actions)}
    for kind in ['target', 'mode', 'op', 'dcs']:
        row['n_' + kind] = int(actions.kind.eq(kind).sum())
    return (row, actions.drop(columns=['time', 'position']))

def build(data, events, changes, band, control_full=False):
    bw, moisture = BANDS[band]
    inside = (data.valid & data.weight_error_raw.abs().le(bw) & data.moisture_error_raw.abs().le(moisture)).to_numpy()
    rows, action_frames = ([], [])
    for i, start in enumerate(changes):
        bounds = [(min(int(start) + 240, len(data)), 'cap_240' if start + 240 <= len(data) else 'record_end')]
        if i + 1 < len(changes):
            bounds.append((int(changes[i + 1]), 'next_grade'))
        stop, censor_reason = min(bounds, key=lambda r: r[0])
        end, reason, entry = endpoint(int(start), stop, inside)
        row, actions = episode_row(data, events, int(start), end, reason or censor_reason, entry, f'T{i + 1:05d}_{band}', 'transition', band)
        rows.append(row)
        action_frames.append(actions)
    modes = events[events.kind.eq('mode')]
    targets = events.loc[events.kind.eq('target'), 'position'].to_numpy()
    control_stats = {'normal_manual_starts': 0, 'return_unobserved': 0, 'no_target_in_manual': 0, 'no_target_before_common_end': 0, 'included': 0}
    counter = 0
    for loop, g in modes.groupby('loop'):
        g = g.sort_values('position')
        records = list(g.itertuples())
        for i, event in enumerate(records):
            if event.shape != 'aut→man':
                continue
            start = int(event.position)
            if not data.normal.iloc[start] or data.gc.iloc[start]:
                continue
            control_stats['normal_manual_starts'] += 1
            if i + 1 == len(records) or records[i + 1].shape != 'man→aut':
                control_stats['return_unobserved'] += 1
                continue
            returned = int(records[i + 1].position)
            if not ((targets >= start) & (targets < returned)).any():
                control_stats['no_target_in_manual'] += 1
                continue
            if control_full:
                stop, censor_reason = (returned, 'auto_return')
                end, reason, entry = (stop, censor_reason, None)
            else:
                bounds = [(returned, 'auto_return'), (min(start + 240, len(data)), 'cap_240' if start + 240 <= len(data) else 'record_end')]
                nxt = changes[changes > start]
                if len(nxt):
                    bounds.append((int(nxt[0]), 'next_grade'))
                stop, censor_reason = min(bounds, key=lambda r: r[0])
                end, reason, entry = endpoint(start, stop, inside)
            if not ((targets >= start) & (targets < end)).any():
                control_stats['no_target_before_common_end'] += 1
                continue
            counter += 1
            row, actions = episode_row(data, events, start, end, reason or censor_reason, entry, f'C{counter:05d}_{band}', 'normal_manual', band)
            row['initiating_loop'] = loop
            row['manual_full_duration_min'] = returned - start
            rows.append(row)
            action_frames.append(actions)
            control_stats['included'] += 1
    return (pd.DataFrame(rows), pd.concat(action_frames, ignore_index=True), control_stats)

def summary_tables(episodes, actions, dest):
    episodes.groupby(['cohort', 'end_reason']).size().rename('episodes').reset_index().to_csv(dest / 'end_reasons.csv', index=False)
    quantities = ['duration_min', 'band_confirmed_min', 'outside_band_min', 'weight_iae_g_m2_min', 'moisture_iae_pp_min', 'speed_change', 'power_missing_fraction', 'power_observed_mean', 'action_count', 'n_target', 'n_mode', 'n_op', 'n_dcs', 'first_action_min']
    rows = []
    for cohort, g in episodes.groupby('cohort'):
        for q in quantities:
            a = g[q].dropna()
            rows.append({'cohort': cohort, 'measurement': q, 'n': len(a), **dict(zip(['p10', 'median', 'p90', 'max'], a.quantile([0.1, 0.5, 0.9, 1.0]).tolist() if len(a) else [None] * 4))})
    pd.DataFrame(rows).to_csv(dest / 'distributions.csv', index=False)
    pair = episodes[episodes.cohort.eq('transition')].groupby(['pair', 'grade_from', 'grade_to']).size().rename('episodes').reset_index().sort_values('episodes', ascending=False)
    pair.to_csv(dest / 'grade_pairs.csv', index=False)
    pair[pair.episodes.ge(3)].to_csv(dest / 'grade_pairs_at_least3.csv', index=False)
    assert episodes.n_target.add(episodes.n_mode).add(episodes.n_op).add(episodes.n_dcs).eq(episodes.action_count).all()
    assert episodes.action_count.sum() == len(actions)
    assert actions.relative_min.ge(0).all()
    sizes = episodes.set_index('episode_id').duration_min
    assert actions.relative_min.lt(actions.episode_id.map(sizes)).all()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--month')
    parser.add_argument('--control-full', action='store_true')
    parser.add_argument('--only', choices=['extract', 'all'], default='all')
    args = parser.parse_args()
    dest = OUT / 'smoke' if args.month else OUT
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'DONE_s74').exists():
        raise FileExistsError('Preserve completed S74 before another version')
    data, events, changes, roles, n_global = load(args.month)
    metadata = {'month': args.month, 'n_minutes': len(data), 'n_changes': len(changes), 'n_global_changes': n_global, 'roles': roles, 'control_full': args.control_full, 'bands': BANDS, 'seed': SEED, 'unit': 'minute', 'power_units': 'unverified; raw channel sum', 'grade_key': 'grade|basis-weight-class, forward-filled at most 30 minutes'}
    for band in BANDS:
        episodes, actions, controls = build(data, events, changes, band, args.control_full)
        band_dest = dest if band == 'primary' else dest / 'sensitivity_tight'
        band_dest.mkdir(parents=True, exist_ok=True)
        episodes.to_parquet(band_dest / 'episodes.parquet', index=False)
        actions.to_parquet(band_dest / 'episode_actions.parquet', index=False)
        summary_tables(episodes, actions, band_dest)
        metadata[band] = {'episodes': len(episodes), 'actions': len(actions), 'controls': controls}
        print(json.dumps({'band': band, **metadata[band]}), flush=True)
    (dest / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    if args.only == 'all':
        from s74_recipe import analyze
        from s74_report import report_and_figures
        analyze(dest)
        report_and_figures(dest)
        (dest / 'DONE_s74').write_text(json.dumps({'status': 'DONE', 'metadata': metadata}, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
