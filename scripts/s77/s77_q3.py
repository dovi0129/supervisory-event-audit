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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
from s77_common import OUT, ROOT, SEED, PERMUTATIONS, load_common, scope_mask, interval_any, lot_layout, rotated_indices, lot_bootstrap

def past_event_features(target, mode):
    x = pd.DataFrame(index=np.arange(len(target)))
    for kind, flag in [('target', np.asarray(target, bool)), ('mode', np.asarray(mode, bool))]:
        s = pd.Series(flag.astype(float))
        for length in [5, 15, 60]:
            x[f'{kind}_count_{length}'] = s.rolling(length, min_periods=1).sum()
        pos = pd.Series(np.arange(len(flag), dtype=float))
        x[kind + '_age'] = pos - pos.where(flag).ffill()
    return x

def run(dest, smoke=False):
    m, states, events, target, roles, w, meta = load_common()
    w = w[w.returned & w.region.eq('정상 생산') & scope_mask(w.time, smoke)].copy().reset_index(drop=True)
    union = target.any(axis=1)
    w['target_inside'] = interval_any(union, w.start, w.end)
    transition = pd.read_parquet(ROOT / 'artifacts/transition/episodes.parquet', columns=['cohort', 'start_position'])
    grade = np.zeros(len(m), bool)
    grade[transition.loc[transition.cohort.eq('transition'), 'start_position']] = True
    layout = lot_layout(m.lot)
    rng = np.random.default_rng(SEED + 3)
    labels = {h: interval_any(grade, np.arange(len(m)) + 1, np.arange(len(m)) + h + 1) for h in [60, 120]}
    null = {h: np.empty((len(w), PERMUTATIONS), float) for h in labels}
    for p in range(PERMUTATIONS):
        donor = rotated_indices(layout, rng)
        shifted_starts = donor[w.start]
        for h in labels:
            null[h][:, p] = np.where(shifted_starts + h < len(m), labels[h][shifted_starts].astype(float), np.nan)
    rows = []
    for h in labels:
        w['future_grade_' + str(h)] = np.where(w.start.to_numpy() + h < len(m), labels[h][w.start].astype(float), np.nan)
        w['chance_' + str(h)] = np.nanmean(null[h], axis=1)
    for pop, frame in [('all_returned', w), ('main_no_cross', w[~w.cross_lot & w.known_lot])]:
        for group, g in [('all', frame), ('target_yes', frame[frame.target_inside]), ('target_no', frame[~frame.target_inside])]:
            for h in labels:
                eligible = g[g['future_grade_' + str(h)].notna() & g['chance_' + str(h)].notna()]
                obs = eligible['future_grade_' + str(h)].to_numpy(float)
                chance = eligible['chance_' + str(h)].to_numpy()
                ci = lot_bootstrap(obs - chance, eligible.lot)
                rows.append({'population': pop, 'target_stratum': group, 'horizon': h, 'windows': len(eligible), 'incomplete_future_windows': len(g) - len(eligible), 'observed_hits': int(obs.sum()), 'p_observed': float(obs.mean()) if len(g) else None, 'p_chance': float(chance.mean()) if len(g) else None, 'delta': float((obs - chance).mean()) if len(g) else None, 'ci_low': ci[0], 'ci_high': ci[1]})
    pd.DataFrame(rows).to_csv(dest / 'q3_summary.csv', index=False)
    w.to_parquet(dest / str(record_path('q3_windows_private_parquet')), index=False)
    mode = np.zeros(len(m), bool)
    mode[events.loc[events.kind.eq('mode'), 'position']] = True
    completed_target = np.zeros(len(m), bool)
    target_events = events[events.kind.eq('target')]
    completed = target_events.position.to_numpy() + target_events.dur_min.to_numpy(int) - 1
    completed_target[completed[completed < len(m)]] = True
    x = past_event_features(completed_target, mode).fillna(1000000.0)
    t = m.index
    normal = m.region.eq('정상 생산').to_numpy()
    if smoke:
        cutoff = np.searchsorted(t.values, np.datetime64('2022-03-01'))
        train = np.arange(len(m)) < int(cutoff * 0.55)
        valid = (np.arange(len(m)) >= int(cutoff * 0.55)) & (np.arange(len(m)) < int(cutoff * 0.75))
        test = (np.arange(len(m)) >= int(cutoff * 0.75)) & (np.arange(len(m)) < cutoff)
    else:
        train = t < '2022-07-01'
        valid = (t >= '2022-07-01') & (t < '2022-09-01')
        test = (t >= '2022-09-01') & (t < '2022-12-01')

    def purge(mask):
        ids = np.flatnonzero(mask)
        if len(ids):
            mask = np.asarray(mask).copy()
            mask[max(ids[0], ids[-1] - 59):ids[-1] + 1] = False
        return mask & normal
    train, valid, test = (purge(mask) for mask in [train, valid, test])
    model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=100, random_state=SEED)
    model.fit(x.loc[train], labels[60][train])
    score = model.predict_proba(x.loc[valid])[:, list(model.classes_).index(True)]
    precision, recall, threshold = precision_recall_curve(labels[60][valid], score)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    tau = float(threshold[np.flatnonzero(f1 == f1.max())[-1]])
    score = model.predict_proba(x.loc[test])[:, list(model.classes_).index(True)]
    y = labels[60][test]
    pred = score >= tau
    detector = {'domain': 'retrospectively normal-production minutes; region never an input', 'train_minutes': int(train.sum()), 'validation_minutes': int(valid.sum()), 'test_minutes': int(test.sum()), 'tau': tau, 'positive_test_minutes': int(y.sum()), 'precision': float(precision_score(y, pred, zero_division=0)), 'recall': float(recall_score(y, pred, zero_division=0)), 'features': list(x)}
    pd.DataFrame({'position': np.flatnonzero(test), 'y': y, 'score': score, 'prediction': pred}).to_parquet(dest / str(record_path('q3_detector_predictions_private_parquet')), index=False)
    (dest / 'q3_detector.json').write_text(json.dumps(detector, ensure_ascii=False, indent=2))
    print(json.dumps({'Q3_windows': len(w), 'detector': detector}, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    d = OUT / 'smoke' if a.smoke else OUT
    d.mkdir(parents=True, exist_ok=True)
    run(d, a.smoke)
