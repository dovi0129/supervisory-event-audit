from __future__ import annotations
import sys as _runtime_sys
from pathlib import Path as _RuntimePath
_runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[2]))
from audit_runtime import PACKAGE_ROOT, WORKSPACE, record_path, require_records
if __name__ == "__main__":
    require_records()
import json
import os
import time
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from s77_common import OUT, SEED, BOOTSTRAPS
from s77_q2_build import label_history
from s77_q2_fit import GRID, make_model, preprocessing, probability, splits
DEST = OUT / 's89'
N_CLASSES = 9
HISTORY_FIELDS = {'hist__last_kind', 'hist__last_direction', 'hist__last_size', 'hist__last_age'}
MODELS = ['majority', 'markov', 'markov_pair', 'R0p', 'R1_history', 'R1', 'R2', 'R1_deep', 'R2_deep']

def history_columns(frame):
    return [c for c in frame if c.startswith('recipe__') or c in HISTORY_FIELDS or c.startswith('hist__count_')]

def frequency_predictions(frame, train, paired=False):
    source = frame.loc[train]

    def frequencies(y):
        c = np.bincount(np.asarray(y, int), minlength=N_CLASSES).astype(float) + 1e-06
        return c / c.sum()
    prior = frequencies(source.kind_code)
    keysets = [['pair', 'hist__last_kind'], ['pair'], ['hist__last_kind']] if paired else [['hist__last_kind']]
    maps = []
    for keys in keysets:
        mapping = {}
        grouper = keys if len(keys) > 1 else keys[0]
        for key, group in source.groupby(grouper, dropna=False):
            mapping[key if isinstance(key, tuple) else (key,)] = frequencies(group.kind_code)
        maps.append((keys, mapping))
    result = np.tile(prior, (len(frame), 1))
    chosen = {'global': 0}
    records = frame[['pair', 'hist__last_kind']].to_dict('records')
    for i, record in enumerate(records):
        selected = 'global'
        for keys, mapping in maps:
            key = tuple((record[c] for c in keys))
            if key in mapping:
                result[i] = mapping[key]
                selected = '+'.join(keys)
                break
        chosen[selected] = chosen.get(selected, 0) + 1
    return (result, prior, chosen)

def fit_fold(fold):
    start = time.monotonic()
    output = DEST / 'folds'
    output.mkdir(parents=True, exist_ok=True)
    target = output / f'{fold}.json'
    if target.exists():
        return json.loads(target.read_text())
    metadata = json.loads((OUT / 'q2_build_manifest.json').read_text())
    original = json.loads((OUT / 'folds' / f'{fold}.json').read_text())
    episodes = pd.read_parquet(OUT / str(record_path('q2_episodes_private_parquet')))
    frame = pd.read_parquet(OUT / str(record_path('q2_decisions_private_parquet')))
    bundles = json.loads((OUT / str(record_path('q2_bundles_private_json'))).read_text())
    _, tr_ids, va_ids, te_ids = next((s for s in splits(episodes, False) if s[0] == fold))
    assert not (set(tr_ids) & set(va_ids) or set(tr_ids) & set(te_ids) or set(va_ids) & set(te_ids))
    frame = label_history(frame, bundles, original['scales'], metadata['roles'])
    frame['kind_code'] = frame.label_kind.map(dict(zip(metadata['classes'], range(N_CLASSES))))
    observed = ~frame.censored
    tr = frame.episode_id.isin(tr_ids) & observed
    va = frame.episode_id.isin(va_ids) & observed
    te = frame.episode_id.isin(te_ids) & observed
    final = frame.episode_id.isin(tr_ids + va_ids) & observed
    test_frame = frame.loc[te].reset_index(drop=True)
    pred = pd.read_parquet(OUT / 'folds' / f'{fold}_predictions_PRIVATE.parquet')
    assert np.array_equal(pred.position, test_frame.position)
    assert np.array_equal(pred.episode_id, test_frame.episode_id)
    assert np.array_equal(pred.kind_code, test_frame.kind_code.to_numpy(int))
    assert len(tr_ids) == original['train_episodes'] and len(va_ids) == original['validation_episodes']
    assert int(final.sum()) == original['final_training_rows']
    assert test_frame.step.eq(0).eq(test_frame.completed_bundle_ids.eq('[]')).all()
    stats = {'fold': fold, 'seed': SEED, 'train_episodes': len(tr_ids), 'validation_episodes': len(va_ids), 'test_episodes': len(te_ids), 'test_decisions': len(pred), 'scales_reused': True, 'label_and_row_alignment_verified': True, 'fits': []}
    for name, paired in [('markov', False), ('markov_pair', True)]:
        p, prior, _ = frequency_predictions(frame, final, paired)
        for j in range(N_CLASSES):
            pred[f'{name}_p{j}'] = p[te, j]
            pred[f'majority_p{j}'] = prior[j]
    history = history_columns(frame)
    assert not any((c.startswith(('hist__target_', 'hist__target_age_', 'hist__mode_')) for c in history))
    variants = [('R1_history', history, [(d, 60) for d in GRID])]
    for name, include_state in [('R1_deep', False), ('R2_deep', True)]:
        columns = [c for c in frame if c.startswith(('recipe__', 'hist__')) or (include_state and c.startswith('state__'))]
        variants.append((name, columns, [(6, 500)]))
    model_dir = DEST / str(record_path('models_private')) / fold
    model_dir.mkdir(parents=True, exist_ok=True)
    ytr = frame.loc[tr, 'kind_code'].to_numpy(int)
    yva = frame.loc[va, 'kind_code'].to_numpy(int)
    yfinal = frame.loc[final, 'kind_code'].to_numpy(int)
    for name, columns, grid in variants:
        pre = preprocessing(frame, columns)
        xtr = pre.fit_transform(frame.loc[tr, columns]).astype('float32')
        xva = pre.transform(frame.loc[va, columns]).astype('float32')
        candidates = []
        for depth, iterations in grid:
            model = make_model('kind', depth).set_params(max_iter=iterations)
            model.fit(xtr, ytr)
            p = probability(model, xva, N_CLASSES)
            value = (float(accuracy_score(yva, p.argmax(1))), -float(log_loss(yva, p, labels=np.arange(N_CLASSES))))
            candidates.append((value, depth, iterations))
        selected = max(range(len(candidates)), key=lambda j: candidates[j][0])
        value, depth, iterations = candidates[selected]
        final_pre = preprocessing(frame, columns)
        xf = final_pre.fit_transform(frame.loc[final, columns]).astype('float32')
        xt = final_pre.transform(frame.loc[te, columns]).astype('float32')
        model = make_model('kind', depth).set_params(max_iter=iterations)
        model.fit(xf, yfinal)
        p = probability(model, xt, N_CLASSES)
        for j in range(N_CLASSES):
            pred[f'{name}_p{j}'] = p[:, j]
        joblib.dump({'model': model, 'preprocessor': final_pre, 'columns': columns, 'scales': original['scales']}, model_dir / f'{name}.joblib')
        stats['fits'].append({'model': name, 'columns': columns, 'grid': grid, 'chosen_depth': depth, 'iterations': iterations, 'validation_accuracy': value[0], 'validation_log_loss': -value[1]})
    assert np.isfinite(pred[[f'{m}_p{j}' for m in MODELS for j in range(N_CLASSES)]].to_numpy()).all()
    for name in MODELS:
        assert np.allclose(pred[[f'{name}_p{j}' for j in range(N_CLASSES)]].sum(axis=1), 1)
    pred.to_parquet(output / f'{fold}_predictions_PRIVATE.parquet', index=False)
    stats['seconds'] = time.monotonic() - start
    target.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(json.dumps({'fold_done': fold, 'seconds': round(stats['seconds'], 2)}, ensure_ascii=False), flush=True)
    return stats

def grouped_statistics(y, prob, episodes):
    ep, unique = pd.factorize(np.asarray(episodes), sort=True)
    n = len(unique)
    counts = np.bincount(ep, minlength=n).astype(float)
    predicted = prob.argmax(axis=1)
    cm = np.zeros((n, N_CLASSES, N_CLASSES), float)
    np.add.at(cm, (ep, y, predicted), 1)
    ll = -np.log(np.clip(prob[np.arange(len(y)), y], np.finfo(float).eps, 1))
    loss = np.bincount(ep, weights=ll, minlength=n)
    return (counts, cm, loss)

def measures(cm, loss, counts):
    cm = np.asarray(cm)
    truth = cm.sum(axis=-1)
    predicted = cm.sum(axis=-2)
    hits = np.diagonal(cm, axis1=-2, axis2=-1)
    accuracy = hits.sum(axis=-1) / counts
    recall = np.divide(hits, truth, out=np.zeros_like(hits), where=truth > 0)
    balanced = recall.sum(axis=-1) / (truth > 0).sum(axis=-1)
    f1 = np.divide(2 * hits, truth + predicted, out=np.zeros_like(hits), where=truth + predicted > 0).mean(axis=-1)
    return {'accuracy': accuracy, 'balanced_accuracy': balanced, 'macro_f1': f1, 'log_loss': loss / counts}

def score_frame(frame, models, bootstrap=True):
    y = frame.kind_code.to_numpy(int)
    ep = frame.episode_id.to_numpy()
    n_ep = frame.episode_id.nunique()
    weights = None
    if bootstrap and n_ep >= 2:
        rng = np.random.default_rng(SEED)
        sampled = rng.integers(0, n_ep, (BOOTSTRAPS, n_ep))
        weights = np.stack([np.bincount(row, minlength=n_ep) for row in sampled]).astype(float)
    result = {}
    for name in models:
        p = frame[[f'{name}_p{j}' for j in range(N_CLASSES)]].to_numpy()
        counts, cm, loss = grouped_statistics(y, p, ep)
        assert len(counts) == n_ep
        for weighting in ['decision', 'episode']:
            denom = np.ones(n_ep) if weighting == 'decision' else counts
            c = cm / denom[:, None, None]
            v = loss / denom
            nc = counts / denom
            point = measures(c.sum(0), v.sum(), nc.sum())
            draws = None
            if weights is not None:
                bc = (weights @ c.reshape(n_ep, -1)).reshape(BOOTSTRAPS, N_CLASSES, N_CLASSES)
                draws = measures(bc, weights @ v, weights @ nc)
            result[name, weighting] = (point, draws)
    return result

def aggregate():
    frames = []
    for fold in ['time', 'group0', 'group1', 'group2', 'group3', 'group4']:
        d = pd.read_parquet(DEST / 'folds' / f'{fold}_predictions_PRIVATE.parquet')
        d['split'] = 'time' if fold == 'time' else 'group'
        frames.append(d)
    predictions = pd.concat(frames, ignore_index=True)
    metric_rows, comparison_rows, initial_rows, verification = ([], [], [], [])
    contrasts = [('R1', 'R1_history'), ('R1_history', 'markov'), ('R1_history', 'markov_pair'), ('R1_history', 'R0p'), ('R1', 'markov'), ('R1', 'markov_pair'), ('R1', 'R0p'), ('R2', 'R1'), ('R2_deep', 'R1_deep'), ('R1_deep', 'R1'), ('R2_deep', 'R2')]
    for split, whole in predictions.groupby('split'):
        for stratum, sub in [('all', whole), ('seen', whole[whole.seen]), ('seen2', whole[whole.seen2]), ('unseen', whole[~whole.seen])]:
            if not len(sub):
                continue
            values = score_frame(sub, MODELS)
            meta = {'split': split, 'stratum': stratum, 'decisions': len(sub), 'episodes': sub.episode_id.nunique()}
            for (name, weighting), (point, _) in values.items():
                metric_rows.append(meta | {'model': name, 'weighting': weighting} | {m: float(v) for m, v in point.items()})
            for high, low in contrasts:
                for weighting in ['decision', 'episode']:
                    hp, hd = values[high, weighting]
                    lp, ld = values[low, weighting]
                    for metric in hp:
                        delta = float(hp[metric] - lp[metric])
                        ci = np.quantile(hd[metric] - ld[metric], [0.025, 0.975]) if hd is not None else [np.nan, np.nan]
                        comparison_rows.append(meta | {'contrast': high + '-' + low, 'weighting': weighting, 'metric': metric, 'difference': delta, 'ci_low': float(ci[0]), 'ci_high': float(ci[1])})
            for phase, part in [('initial', sub[sub.step.eq(0)]), ('subsequent', sub[sub.step.gt(0)])]:
                if not len(part):
                    continue
                vals = score_frame(part, ['R0p', 'R1', 'R2'])
                for high, low in [('R2', 'R1'), ('R1', 'R0p')]:
                    for weighting in ['decision', 'episode']:
                        hp, hd = vals[high, weighting]
                        lp, ld = vals[low, weighting]
                        for metric in ['accuracy', 'log_loss']:
                            ci = np.quantile(hd[metric] - ld[metric], [0.025, 0.975]) if hd is not None else [np.nan, np.nan]
                            initial_rows.append({'split': split, 'stratum': stratum, 'phase': phase, 'decisions': len(part), 'episodes': part.episode_id.nunique(), 'contrast': high + '-' + low, 'weighting': weighting, 'metric': metric, 'high_value': float(hp[metric]), 'low_value': float(lp[metric]), 'difference': float(hp[metric] - lp[metric]), 'ci_low': float(ci[0]), 'ci_high': float(ci[1])})
    metrics = pd.DataFrame(metric_rows)
    comparisons = pd.DataFrame(comparison_rows)
    metrics.to_csv(DEST / 'q2_metrics.csv', index=False)
    comparisons.to_csv(DEST / 'q2_paired_comparisons.csv', index=False)
    pd.DataFrame(initial_rows).to_csv(DEST / 'q2_initial_subsequent.csv', index=False)
    metrics[metrics.model.isin(['R1', 'R2', 'R1_deep', 'R2_deep'])].to_csv(DEST / 'q2_capacity_metrics.csv', index=False)
    old_metrics = pd.read_csv(OUT / 'q2_metrics.csv')
    old_comparisons = pd.read_csv(OUT / 'q2_paired_comparisons.csv')
    for row in old_metrics[old_metrics.model.isin(['R0p', 'R1', 'R2'])].itertuples():
        current = metrics[(metrics.split == row.split) & (metrics.stratum == row.stratum) & (metrics.model == row.model) & (metrics.weighting == 'decision')].iloc[0]
        assert np.isclose(current.accuracy, row.accuracy, atol=1e-12, rtol=0)
        assert np.isclose(current.log_loss, row.log_loss, atol=1e-12, rtol=0)
    for row in old_comparisons.itertuples():
        current = comparisons[(comparisons.split == row.split) & (comparisons.stratum == row.stratum) & (comparisons.contrast == row.contrast) & (comparisons.weighting == 'decision') & (comparisons.metric == 'accuracy')].iloc[0]
        assert np.allclose([current.difference, current.ci_low, current.ci_high], [row.accuracy_difference, row.ci_low, row.ci_high], atol=1e-12, rtol=0)
    verification.append({'check': 'existing_metrics_and_paired_intervals', 'passed': True, 'metric_rows': len(old_metrics[old_metrics.model.isin(['R0p', 'R1', 'R2'])])})
    (DEST / 'q2_verification.json').write_text(json.dumps(verification, indent=2))
    return {'metric_rows': len(metrics), 'comparison_rows': len(comparisons), 'phase_rows': len(initial_rows)}
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', choices=['time', 'group0', 'group1', 'group2', 'group3', 'group4'])
    parser.add_argument('--aggregate', action='store_true')
    args = parser.parse_args()
    if args.fold:
        fit_fold(args.fold)
    if args.aggregate:
        print(json.dumps(aggregate()), flush=True)
