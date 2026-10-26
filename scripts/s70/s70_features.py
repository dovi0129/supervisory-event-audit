from __future__ import annotations
import sys as _runtime_sys
from pathlib import Path as _RuntimePath
_runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[2]))
from audit_runtime import PACKAGE_ROOT, WORKSPACE, record_path, require_records
if __name__ == "__main__":
    require_records()
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = WORKSPACE
PRIVATE = ROOT / str(record_path('artifacts_private'))
OUT = ROOT / 'artifacts/intervention'
SIGNALS = {'moisture': 'moisture', 'moisture_target': 'moisture_target', 'size_moisture': 'size_moisture', 'size_moisture_target': 'size_moisture_target', 'weight': 'weight', 'weight_target': 'weight_target', 'speed': 'speed', 'pressure_after': 'pressure_after', 'pressure_pre': 'pressure_pre'}

def elapsed(changed: pd.Series) -> pd.Series:
    pos = pd.Series(np.arange(len(changed), dtype=float), index=changed.index)
    return pos - pos.where(changed).ffill()

def causal_features(observed: pd.DataFrame) -> pd.DataFrame:
    p = observed.copy()
    x = pd.DataFrame(index=p.index)
    for prefix in ['moisture', 'size_moisture', 'weight']:
        d = p[prefix] - p[prefix + '_target']
        if prefix == 'weight':
            x['weight_error_raw'] = d
            d = d - d.rolling(60, min_periods=10).median()
        x[prefix + '_error'] = d
        x[prefix + '_slope5'] = (d - d.shift(5)) / 5
        sign = np.sign(d)
        group = (sign.ne(sign.shift()) | d.isna()).cumsum()
        x[prefix + '_duration'] = d.groupby(group).cumcount().add(1).where(d.notna())
        target = p[prefix + '_target']
        changed = target.notna() & target.shift().notna() & target.ne(target.shift())
        x[prefix + '_target_age'] = elapsed(changed)
    for c in ['speed', 'pressure_after', 'pressure_pre']:
        x[c] = p[c]
        x[c + '_change5'] = (p[c] - p[c].shift(5)) / 5
    for c in p.filter(like='mode_'):
        x[c] = p[c]
    changed = p['grade'].notna() & p['grade'].shift().notna() & p['grade'].ne(p['grade'].shift())
    x['grade_age'] = elapsed(changed)
    hour = x.index.hour
    x['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    x['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    x['shift'] = ((hour - 6) % 24 // 8).astype(float)
    x['grade'] = p['grade'].fillna('unknown').astype(str)
    nums = x.columns.difference(['grade'])
    x[nums] = x[nums].astype('float32')
    return x

def future_labels(events: np.ndarray, k: int) -> np.ndarray:
    events = np.asarray(events, dtype=bool)
    prefix = np.r_[0, np.cumsum(events)]
    t = np.arange(len(events))
    return prefix[np.minimum(t + k + 1, len(events))] - prefix[t + 1] > 0

def load_inputs(month: str | None=None):
    mapping = json.loads((ROOT / str(record_path('artifacts_operator_policy_tag_map_private_json'))).read_text())
    loops = json.loads((ROOT / 'artifacts/operator_policy/loops.json').read_text())['loops']
    modes = {tag: role for tag, role in mapping['loops'].items() if loops.get(role, {}).get('kind') == 'mode'}
    columns = ['time', '지종'] + list(SIGNALS) + list(modes)
    raw = pd.read_parquet(PRIVATE / 's68_cache.parquet', columns=columns).set_index('time')
    minutes = pd.read_parquet(PRIVATE / 's68_minutes.parquet').set_index('time')
    if not raw.index.equals(minutes.index) or not raw.index.is_unique:
        raise ValueError('S68 minute alignment/uniqueness failed')
    index = pd.date_range(raw.index.min(), raw.index.max(), freq='min', name='time')
    raw, minutes = (raw.reindex(index), minutes.reindex(index))
    if month:
        sel = index.strftime('%Y-%m') == month
        raw, minutes = (raw.loc[sel], minutes.loc[sel])
    obs = raw[list(SIGNALS)].rename(columns=SIGNALS).apply(pd.to_numeric, errors='coerce')
    obs['grade'] = raw['지종'].ffill(limit=30).map(mapping['grades'])
    mode_names = {}
    for i, (tag, role) in enumerate(modes.items()):
        name = f'mode_{i:02d}'
        state = raw[tag].ffill().map({'aut': 1.0, 'man': 0.0})
        obs[name] = state
        mode_names[role] = name
    events = pd.read_parquet(ROOT / 'artifacts/operator_policy/events.parquet', columns=['time', 'loop', 'kind', 'shape', 'flag_brk_near'])
    events = events[events['kind'].eq('mode')].sort_values(['loop', 'time']).copy()
    nxt = events.groupby('loop')['time'].shift(-1)
    prev = events.groupby('loop')['time'].shift(1)
    events['quick'] = (nxt - events['time']).le(pd.Timedelta(minutes=2))
    events['short_previous'] = (events['time'] - prev).le(pd.Timedelta(minutes=2))
    events['return_observed'] = nxt.notna()
    counts = events.groupby('loop').size()
    targets = sorted(counts[counts >= 1000].index)
    if month:
        events = events[events.time.dt.strftime('%Y-%m').eq(month)]
    return (obs, minutes, events, mode_names, targets)

def prepare(month: str | None=None) -> Path:
    obs, minutes, events, modes, targets = load_inputs(month)
    x = causal_features(obs)
    suffix = '_smoke' if month else ''
    dest = OUT / ('data' + suffix)
    dest.mkdir(parents=True, exist_ok=True)
    x.to_parquet(dest / str(record_path('features_private_parquet')))
    minutes['region'] = minutes['region'].map({'정상 생산': 'normal', '지종 전환': 'transition'}).fillna('other')
    minutes['brk_near'] = minutes['brk'].fillna(False).astype(bool).rolling(61, center=True, min_periods=1).max().astype(bool)
    minutes.to_parquet(dest / str(record_path('minutes_private_parquet')))
    obs.filter(like='mode_').to_parquet(dest / str(record_path('modes_private_parquet')))
    events.to_parquet(dest / str(record_path('events_private_parquet')), index=False)
    meta = {'month': month, 'n_minutes': len(x), 'targets': targets, 'mode_names': modes, 'features': list(x), 'max_horizon': 15, 'history_purge': 0, 'spec_version': '0.2', 'regions': ['normal', 'transition', 'other']}
    (dest / 'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps({'prepared_minutes': len(x), 'loops': len(targets), 'smoke': bool(month)}), flush=True)
    return dest
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--month')
    args = parser.parse_args()
    prepare(args.month)
