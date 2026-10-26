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
from s77_common import ROOT, OUT, load_common, scope_mask, stage_bin, SEED
MEASUREMENTS = ['moisture_error', 'size_moisture_error', 'weight_error', 'weight_error_raw', 'moisture_slope5', 'size_moisture_slope5', 'weight_slope5', 'moisture_duration', 'size_moisture_duration', 'weight_duration', 'speed', 'speed_change5', 'pressure_pre', 'pressure_pre_change5', 'pressure_after', 'pressure_after_change5']

def bundle_records(actions):
    result = []
    for start, g in actions.groupby('relative_min', sort=True):
        length = int(g.dur_min.max())
        result.append({'start': int(start), 'complete': int(start) + length - 1, 'members': [{'kind': r.kind, 'loop': r.loop, 'delta': float(r.delta), 'dur_min': int(r.dur_min)} for r in g.itertuples()]})
    return result

def decision_records(bundles, end):
    times = sorted({0} | {b['complete'] for b in bundles if 0 < b['complete'] < end})
    rows = []
    for t in times:
        completed = [] if t == 0 else [i for i, b in enumerate(bundles) if b['complete'] <= t]
        future = [i for i, b in enumerate(bundles) if b['start'] > t]
        nxt = future[0] if future else None
        observed = nxt is not None and bundles[nxt]['complete'] < end
        rows.append({'relative_min': t, 'completed': completed, 'next_bundle': nxt if observed else None, 'censored': not observed, 'followup_min': bundles[nxt]['start'] - t if observed else end - t, 'censor_reason': 'incomplete_next_bundle' if nxt is not None and (not observed) else 'episode_boundary' if not observed else '', 'step': len(completed), 'elapsed_bin': int(stage_bin(t))})
    return rows

def choose_primary(bundle, scales, roles):
    members = bundle['members']
    targets = [r for r in members if r['kind'] == 'target']
    if targets:
        selected = sorted(targets, key=lambda r: (-abs(r['delta']) / scales[r['loop']], roles.index(r['loop'])))[0]
        return (selected['loop'], float(np.sign(selected['delta'])), abs(selected['delta']) / scales[selected['loop']])
    for kind, label in [('mode', 'mode'), ('op', 'manual_output'), ('dcs', 'DCS')]:
        selected = sorted([r for r in members if r['kind'] == kind], key=lambda r: r['loop'])
        if selected:
            r = selected[0]
            return (label, float(np.sign(r['delta'])), abs(r['delta']))
    raise ValueError('Empty bundle')

def label_history(frame, bundles, scales, roles):
    result = frame.copy()
    kinds = []
    directions = []
    sizes = []
    lastkind = []
    lastdirection = []
    lastsize = []
    lastage = []
    counts = np.zeros((len(frame), 9), float)
    classes = roles + ['mode', 'manual_output', 'DCS']
    for pos, r in enumerate(frame.itertuples()):
        completed = json.loads(r.completed_bundle_ids)
        previous = [choose_primary(bundles[i], scales, roles) for i in completed]
        for p in previous:
            counts[pos, classes.index(p[0])] += 1
        if completed:
            j = max(completed, key=lambda i: (bundles[i]['complete'], bundles[i]['start']))
            k, d, s = choose_primary(bundles[j], scales, roles)
            lastkind.append(k)
            lastdirection.append(d)
            lastsize.append(s)
            lastage.append(r.relative_min - bundles[j]['complete'])
        else:
            lastkind.append('none')
            lastdirection.append(0.0)
            lastsize.append(0.0)
            lastage.append(np.nan)
        if r.next_bundle_id >= 0:
            k, d, s = choose_primary(bundles[r.next_bundle_id], scales, roles)
            kinds.append(k)
            directions.append(d)
            sizes.append(s if k in roles else np.nan)
        else:
            kinds.append(None)
            directions.append(np.nan)
            sizes.append(np.nan)
    result['label_kind'] = kinds
    result['label_direction'] = directions
    result['label_size'] = sizes
    result['hist__last_kind'] = lastkind
    result['hist__last_direction'] = lastdirection
    result['hist__last_size'] = lastsize
    result['hist__last_age'] = lastage
    for j, k in enumerate(classes):
        result[f'hist__count_{j}'] = counts[:, j]
    return result

def connected_groups(episodes, minutes):
    ids = episodes.episode_id.tolist()
    parent = list(range(len(ids)))
    owner = {}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, r in enumerate(episodes.itertuples()):
        lots = minutes.lot.iloc[r.start_position:r.end_position_exclusive].dropna().unique()
        for lot in lots:
            if lot in owner:
                a, b = (root(i), root(owner[lot]))
                parent[a] = b
            else:
                owner[lot] = i
    return {episode: f'G{root(i):05d}' for i, episode in enumerate(ids)}

def run(dest, smoke=False):
    m, states, events, target, roles, windows, meta = load_common()
    ep = pd.read_parquet(ROOT / 'artifacts/transition/episodes.parquet')
    ep = ep[ep.cohort.eq('transition') & scope_mask(ep.start, smoke)].copy()
    actions = pd.read_parquet(ROOT / 'artifacts/transition/episode_actions.parquet')
    actions = actions[actions.episode_id.isin(ep.episode_id)].copy()
    starts = ep.set_index('episode_id').start
    actions['time'] = actions.episode_id.map(starts) + pd.to_timedelta(actions.relative_min, unit='min')
    source = events[['time', 'loop', 'kind', 'dur_min']]
    assert not source.duplicated(['time', 'loop', 'kind']).any()
    actions = actions.merge(source, on=['time', 'loop', 'kind'], how='left', validate='many_to_one')
    assert actions.dur_min.notna().all()
    grouped = {k: g for k, g in actions.groupby('episode_id')}
    groups = connected_groups(ep, m)
    all_bundles = []
    rows = []
    no_actions = 0
    for r in ep.itertuples():
        g = grouped.get(r.episode_id)
        if g is None:
            no_actions += 1
            continue
        bundles = bundle_records(g)
        offset = len(all_bundles)
        all_bundles.extend(bundles)
        for record in decision_records(bundles, r.duration_min):
            completed = [i + offset for i in record.pop('completed')]
            next_id = record.pop('next_bundle')
            t = r.start_position + record['relative_min']
            rows.append({'episode_id': r.episode_id, 'episode_start': r.start, 'pair': r.pair, 'group': groups[r.episode_id], 'position': int(t), 'recipe__pair': r.pair, 'recipe__step': record['step'], 'recipe__elapsed_bin': str(record['elapsed_bin']), 'completed_bundle_ids': json.dumps(completed), 'next_bundle_id': next_id + offset if next_id is not None else -1, **record})
    frame = pd.DataFrame(rows)
    features = pd.read_parquet(ROOT / str(record_path('artifacts_intervention_data_features_private_parquet')), columns=MEASUREMENTS)
    assert features.index.equals(m.index)
    for name in MEASUREMENTS:
        frame['state__' + name] = features[name].iloc[frame.position].to_numpy()
    maps = json.loads((ROOT / str(record_path('artifacts_operator_policy_tag_map_private_json'))).read_text())
    tags = {tag: role for tag, role in maps['loops'].items() if role in roles}
    settings = pd.read_parquet(ROOT / str(record_path('artifacts_private_s68_cache_parquet')), columns=['time'] + list(tags)).set_index('time').reindex(m.index)
    for j, role in enumerate(roles):
        tag = next((t for t, r in tags.items() if r == role))
        value = pd.to_numeric(settings[tag], errors='coerce').ffill(limit=5)
        changed = value.notna() & value.shift().notna() & value.ne(value.shift())
        pos = pd.Series(np.arange(len(m), dtype=float), index=m.index)
        age = pos - pos.where(changed).ffill()
        frame[f'hist__target_{j}'] = value.iloc[frame.position].to_numpy()
        frame[f'hist__target_age_{j}'] = age.iloc[frame.position].to_numpy()
    for loop, name in meta['mode_names'].items():
        frame['hist__' + name] = states[name].iloc[frame.position].to_numpy()
    frame.to_parquet(dest / str(record_path('q2_decisions_private_parquet')), index=False)
    (dest / str(record_path('q2_bundles_private_json'))).write_text(json.dumps(all_bundles, ensure_ascii=False))
    ep[['episode_id', 'start', 'pair', 'action_count']].assign(group=ep.episode_id.map(groups)).to_parquet(dest / str(record_path('q2_episodes_private_parquet')), index=False)
    q = json.loads((ROOT / 'artifacts/operator_policy/loops.json').read_text())['loops']
    frozen = {role: float(q[role]['q99'] - q[role]['q1']) for role in roles}
    metadata = {'episodes': len(ep), 'no_action_episodes': no_actions, 'decision_rows': len(frame), 'censored_rows': int(frame.censored.sum()), 'groups': len(set(groups.values())), 'roles': roles, 'classes': roles + ['mode', 'manual_output', 'DCS'], 'frozen_s68_scales': frozen, 'scale_mode': 'train', 'scale_scope': 'initial training decision times; fixed through validation/test/refit', 'smoke': smoke, 'seed': SEED, 'initial_rule': 'one empty-history row at t0; completion rows at t0 share that one row; later completions unique by minute', 'history_rule': 'completed bundles only; tie at completion resolved by latest start for last-bundle summary'}
    (dest / 'q2_build_manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps({'Q2_build': metadata}, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    d = OUT / 'smoke' if a.smoke else OUT
    d.mkdir(parents=True, exist_ok=True)
    run(d, a.smoke)
