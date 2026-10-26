from __future__ import annotations
import sys as _runtime_sys
from pathlib import Path as _RuntimePath
_runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[2]))
from audit_runtime import PACKAGE_ROOT, WORKSPACE, record_path, require_records
if __name__ == "__main__":
    require_records()
import argparse
import json
import os
import time
from pathlib import Path
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_key] = '1'
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from s70_features import OUT, future_labels
SEED = 7001
RULE_GRID = {'moisture_error': [0.1, 0.2, 0.3, 0.5, 0.8], 'size_moisture_error': [0.2, 0.3, 0.5, 0.8, 1.0], 'weight_error': [0.5, 1.0, 2.0, 3.0, 5.0]}
RULE_DURATIONS = [1, 3, 5, 10]

def complete_windows(mask, k, history=0):
    mask = np.asarray(mask, dtype=bool)
    prefix = np.r_[0, np.cumsum(~mask)]
    t = np.arange(len(mask))
    lo, hi = (t - history, t + k + 1)
    ok = (lo >= 0) & (hi <= len(mask))
    return ok & (prefix[np.minimum(hi, len(mask))] - prefix[np.maximum(lo, 0)] == 0)

def lot_windows(lot, k):
    lot = np.asarray(lot, dtype=str)
    episode = np.cumsum(np.r_[True, lot[1:] != lot[:-1]])
    end = np.minimum(np.arange(len(lot)) + k, len(lot) - 1)
    return (episode == episode[end]) & (np.arange(len(lot)) + k < len(lot)) & (lot != '__missing__')

def splits(minutes, smoke, k):
    n, t = (len(minutes), minutes.index)
    lot = minutes['lot'].fillna('__missing__').astype(str).to_numpy()
    known = lot != '__missing__'
    same_lot = lot_windows(lot, k)
    if smoke:
        train = np.arange(n) < int(n * 0.55)
        valid = (np.arange(n) >= int(n * 0.55)) & (np.arange(n) < int(n * 0.75))
        test = np.arange(n) >= int(n * 0.75)
    else:
        train = t < '2022-07-01'
        valid = (t >= '2022-07-01') & (t < '2022-09-01')
        test = (t >= '2022-09-01') & (t < '2022-12-01')
    yield ('time', *(complete_windows(m, k) & same_lot for m in [train, valid, test]))
    ix = np.flatnonzero(known)
    folds = min(5, len(np.unique(lot[ix])))
    if folds < 2:
        raise ValueError('Not enough LOT groups')
    for f, (tr0, te0) in enumerate(GroupKFold(folds).split(ix, groups=lot[ix])):
        tr_pool, te = (ix[tr0], ix[te0])
        tr1, va1 = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(tr_pool, groups=lot[tr_pool]))
        groups = [set(lot[tr_pool[tr1]]), set(lot[tr_pool[va1]]), set(lot[te])]
        assert not any((groups[a] & groups[b] for a, b in [(0, 1), (0, 2), (1, 2)]))
        masks = []
        for pos in [tr_pool[tr1], tr_pool[va1], te]:
            mask = np.zeros(n, bool)
            mask[pos] = True
            masks.append(complete_windows(mask, k) & same_lot)
        yield (f'lot{f}', *masks)

def rising_alarms(score, threshold, mode_domain, k, episode):
    active = (np.asarray(score) >= threshold) & mode_domain
    edges = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    alarm = np.zeros(len(active), bool)
    last, last_episode = (-k - 1, -1)
    for t in edges:
        if episode[t] != last_episode or t - last > k:
            alarm[t] = True
            last, last_episode = (t, episode[t])
    return alarm

def event_hits(alarm, event_positions, domain, k):
    events = np.asarray(event_positions, dtype=int)
    events = events[(events > 0) & (events < len(domain))]
    prefix = np.r_[0, np.cumsum(domain)]
    events = events[prefix[events] - prefix[np.maximum(0, events - k)] > 0]
    fires = np.flatnonzero(alarm & domain)
    first = np.searchsorted(fires, np.maximum(0, events - k))
    candidate = np.r_[fires, len(domain) + k][first]
    hit = candidate < events
    leads = (events[hit] - candidate[hit]).astype(int).tolist()
    return (events, hit, leads)

def event_f1(y, alarm, events, domain, k):
    _, hit, _ = event_hits(alarm, events, domain, k)
    r = float(hit.mean()) if len(hit) else 0.0
    fires = alarm & domain
    p = float(y[fires].mean()) if fires.any() else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0

def choose_threshold(y, score, events, domain, mode_domain, episode, k):
    values = np.unique(score[domain])
    grid = values if len(values) <= 101 else np.unique(np.quantile(values, np.linspace(0, 1, 101)))
    best = (-1.0, 0.5)
    for threshold in grid:
        alarm = rising_alarms(score, threshold, mode_domain, k, episode)
        f1 = event_f1(y, alarm, events, domain, k)
        if (f1, float(threshold)) > best:
            best = (f1, float(threshold))
    return (best[1], best[0])

def select_rule(x, y, events, domain, mode_domain, episode, k):
    best, chosen = (-1.0, None)
    for feature, thresholds in RULE_GRID.items():
        for threshold in thresholds:
            condition = x[feature].abs().gt(threshold)
            run = condition.groupby(condition.ne(condition.shift()).cumsum()).cumcount().add(1).where(condition, 0)
            for duration in RULE_DURATIONS:
                score = (run >= duration).to_numpy(float)
                alarm = rising_alarms(score, 0.5, mode_domain, k, episode)
                f1 = event_f1(y, alarm, events, domain, k)
                if f1 > best:
                    best, chosen = (f1, (score, {'feature': feature, 'threshold': threshold, 'duration': duration, 'validation_event_f1': f1}))
    return chosen

def metrics(y, score, alarm, event_positions, domain, k, rng):
    fires = alarm & domain
    events, hit, leads = event_hits(alarm, event_positions, domain, k)
    r = float(hit.mean()) if len(hit) else None
    chance = []
    pool = np.flatnonzero(domain)
    if len(events) and len(pool) >= len(events):
        prefix = np.r_[0, np.cumsum(fires)]
        for _ in range(50):
            shuffled = rng.choice(pool, size=len(events), replace=False)
            random_hit = prefix[shuffled] - prefix[np.maximum(0, shuffled - k)] > 0
            chance.append(float(random_hit.mean()))
    c = float(np.mean(chance)) if chance else None
    e = (r - c) / (1 - c) if r is not None and c is not None and (c < 1) else None
    yy, pp = (y[domain], score[domain])
    return {'n_minutes': int(domain.sum()), 'positive_minutes': int(yy.sum()), 'alarms': int(fires.sum()), 'true_alarm_minutes': int((fires & y).sum()), 'events': len(events), 'event_hits': int(hit.sum()), 'candidate_events': len(events), 'event_recall': r, 'minute_precision': float(y[fires].mean()) if fires.any() else None, 'pr_auc': float(average_precision_score(yy, pp)) if yy.any() else None, 'chance_recall': c, 'chance_valid_shifts': len(chance), 'chance_excess': e, 'chance_sd': float(np.std(chance)) if chance else None, 'lead_minutes': leads, 'lead_p10_p50_p90': np.percentile(leads, [10, 50, 90]).tolist() if leads else [], 'calibration': []}

def make_model(x, kind, depth=4):
    num = [c for c in x if c != 'grade']
    pre = ColumnTransformer([('num', SimpleImputer(strategy='median', keep_empty_features=True), num), ('grade', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32), ['grade'])])
    model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=50, random_state=SEED) if kind == 'tree' else HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=7, learning_rate=0.08, early_stopping=False, random_state=SEED)
    return make_pipeline(pre, model)

def fit_job(data: Path, key: str):
    start = time.monotonic()
    meta = json.loads((data / 'metadata.json').read_text())
    if meta.get('spec_version') != '0.2':
        raise ValueError('v0.2 feature preparation required')
    loop_idx, k, variant, direction = key.split(':')
    k = int(k)
    loop = meta['targets'][int(loop_idx)]
    smoke = bool(meta['month'])
    destination = OUT / ('smoke' if smoke else 'full') / 'jobs'
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / (key.replace(':', '_') + '.json')
    if result_path.exists():
        raise FileExistsError('Existing result: explicitly archive before replacing')
    x = pd.read_parquet(data / str(record_path('features_private_parquet')))
    own_mode = meta['mode_names'][loop]
    x = x.drop(columns=[own_mode])
    minutes = pd.read_parquet(data / str(record_path('minutes_private_parquet')))
    states = pd.read_parquet(data / str(record_path('modes_private_parquet')), columns=[own_mode])[own_mode]
    mode_eligible = states.eq(1.0 if direction == 'aut→man' else 0.0).to_numpy()
    episode = states.ne(states.shift()).cumsum().to_numpy()
    events = pd.read_parquet(data / str(record_path('events_private_parquet')))
    ev = events[events.loop.eq(loop) & events['shape'].eq(direction)].copy()
    wanted = ~ev.flag_brk_near.fillna(True) & ev.return_observed
    if variant == 'main':
        wanted &= ~ev.quick & ~ev.short_previous
    pos = x.index.get_indexer(ev.time)
    if (pos < 0).any():
        raise ValueError('Event alignment failed')
    accepted = np.zeros(len(x), bool)
    rejected = accepted.copy()
    accepted[pos[wanted.to_numpy()]] = True
    rejected[pos[~wanted.to_numpy()]] = True
    event_positions = np.flatnonzero(accepted)
    y = future_labels(accepted, k)
    eligible = mode_eligible & ~future_labels(rejected, k) & ~rejected & ~minutes.brk_near.to_numpy(bool)
    result = {'key': key, 'spec_version': '0.2', 'loop': loop, 'k': k, 'variant': variant, 'direction': direction, 'smoke': smoke, 'accepted_events': int(accepted.sum()), 'excluded_events': int(rejected.sum()), 'quick_events': int(ev.quick.sum()), 'short_previous_events': int(ev.short_previous.sum()), 'return_unobserved_events': int((~ev.return_observed).sum()), 'selection': '60 rules; depth {2,4}; validation event F1; GBM 101-quantile threshold grid', 'folds': [], 'support': [], 'rules': [], 'event_region_sensitivity': []}
    for fold, train0, valid0, test0 in splits(minutes, smoke, k):
        train, valid, test = (m & eligible for m in [train0, valid0, test0])
        counts = {'train': int(train.sum()), 'validation': int(valid.sum()), 'test': int(test.sum())}
        if train.sum() < 50 or not valid.any() or len(np.unique(y[train])) < 2 or (not y[valid].any()):
            for kind in ['rule', 'tree', 'gbm']:
                for region in ['all', 'normal', 'transition', 'other']:
                    row = metrics(y, np.zeros(len(x)), np.zeros(len(x), bool), [], np.zeros(len(x), bool), k, np.random.default_rng(SEED))
                    row.update({'fold': fold, 'model': kind, 'region': region, 'threshold': None, 'split_counts': counts, 'unseen_grade_fraction': None, 'status': 'insufficient_support', 'unevaluated_test_minutes': counts['test']})
                    result['folds'].append(row)
            continue
        rule_score, rule_definition = select_rule(x, y, event_positions, valid, mode_eligible, episode, k)
        scores, thresholds = ({'rule': rule_score}, {'rule': 0.5})
        result['rules'].append({'fold': fold, 'model': 'rule', **rule_definition})
        for kind in ['tree', 'gbm']:
            best = None
            for depth in [2, 4] if kind == 'tree' else [4]:
                model = make_model(x, kind, depth)
                model.fit(x.loc[train], y[train])
                score = np.full(len(x), -np.inf, dtype=np.float32)
                score[:] = model.predict_proba(x)[:, 1]
                tau, f1 = choose_threshold(y, score, event_positions, valid, mode_eligible, episode, k)
                if best is None or f1 > best[0]:
                    best = (f1, model, tau, depth, score)
            f1, model, tau, depth, score = best
            scores[kind], thresholds[kind] = (score, tau)
            if kind == 'tree':
                result['rules'].append({'fold': fold, 'model': kind, 'depth': depth, 'threshold': tau, 'validation_event_f1': f1, 'text': export_text(model[-1], feature_names=model[0].get_feature_names_out().tolist(), decimals=5)})
        prediction_dir = destination.parent / str(record_path('predictions_private')) / key.replace(':', '_')
        prediction_dir.mkdir(parents=True, exist_ok=True)
        prediction = pd.DataFrame({'position': np.flatnonzero(test), 'y': y[test], 'region': minutes.loc[test, 'region'].to_numpy()})
        for kind, score in scores.items():
            alarm = rising_alarms(score, thresholds[kind], mode_eligible, k, episode)
            prediction[kind] = score[test]
            prediction[kind + '_alarm'] = alarm[test]
            for region in ['all', 'normal', 'transition', 'other']:
                domain = test & (True if region == 'all' else minutes.region.eq(region).to_numpy())
                row = metrics(y, score, alarm, event_positions, domain, k, np.random.default_rng(SEED))
                row.update({'fold': fold, 'model': kind, 'region': region, 'threshold': thresholds[kind], 'split_counts': counts, 'status': 'evaluated', 'unseen_grade_fraction': float((~x.loc[domain, 'grade'].isin(set(x.loc[train, 'grade']))).mean()) if domain.any() else None})
                result['folds'].append(row)
            for region in ['normal', 'transition', 'other']:
                ev_region = event_positions[minutes.region.iloc[event_positions].eq(region)]
                ee, hh, _ = event_hits(alarm, ev_region, test, k)
                result['event_region_sensitivity'].append({'fold': fold, 'model': kind, 'event_region': region, 'events': len(ee), 'event_hits': int(hh.sum())})
        prediction.to_parquet(prediction_dir / (fold + '.parquet'), index=False)
        for feature in x.columns.difference(['grade']):
            tr, te = (x.loc[train & y, feature].dropna(), x.loc[test, feature].dropna())
            result['support'].append({'fold': fold, 'feature': feature, 'train_event_n': len(tr), 'test_n': len(te), 'train_event_min': float(tr.min()) if len(tr) else None, 'train_event_max': float(tr.max()) if len(tr) else None, 'test_min': float(te.min()) if len(te) else None, 'test_max': float(te.max()) if len(te) else None, 'outside_event_range': float(((te < tr.min()) | (te > tr.max())).mean()) if len(tr) and len(te) else None})
        print(json.dumps({'job': key, 'fold': fold, 'minutes': counts, 'seconds': round(time.monotonic() - start, 2)}), flush=True)
    result['seconds'] = time.monotonic() - start
    temporary = result_path.with_suffix('.pending')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=1, allow_nan=False))
    temporary.rename(result_path)
    return result_path

def jobs(data):
    meta = json.loads((data / 'metadata.json').read_text())
    return [f'{i}:{k}:{variant}:{direction}' for i in range(len(meta['targets'])) for k, variant in [(5, 'main'), (15, 'main'), (5, 'quick')] for direction in ['aut→man', 'man→aut']]
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--only')
    parser.add_argument('--list-jobs', action='store_true')
    args = parser.parse_args()
    if args.list_jobs:
        print('\n'.join(jobs(args.data)))
    elif args.only:
        fit_job(args.data, args.only)
    else:
        parser.error('--only or --list-jobs is required')
