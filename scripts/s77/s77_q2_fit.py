from __future__ import annotations
import sys as _runtime_sys
from pathlib import Path as _RuntimePath
_runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[2]))
from audit_runtime import PACKAGE_ROOT, WORKSPACE, record_path, require_records
if __name__ == "__main__":
    require_records()
import os
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_key] = '1'
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import log_loss, accuracy_score
import joblib
from s77_common import OUT, SEED, exchange_indices, kaplan_meier
from s77_q2_build import label_history
GRID = [2, 3]
LAYERS = ['R0p', 'R1', 'R2']
TASKS = ['kind', 'direction', 'size', 'time']

def splits(episodes, smoke):
    ep = episodes.sort_values('start').reset_index(drop=True)
    if smoke:
        n = len(ep)
        tr = ep.iloc[:int(n * 0.55)]
        va = ep.iloc[int(n * 0.55):int(n * 0.75)]
        te = ep.iloc[int(n * 0.75):]
    else:
        tr = ep[ep.start < '2022-07-01']
        va = ep[(ep.start >= '2022-07-01') & (ep.start < '2022-09-01')]
        te = ep[ep.start >= '2022-09-01']
    yield ('time', tr.episode_id.tolist(), va.episode_id.tolist(), te.episode_id.tolist())
    for k, (train, test) in enumerate(GroupKFold(5).split(ep, groups=ep.group)):
        pool = ep.iloc[train]
        a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(pool, groups=pool.group))
        yield (f'group{k}', pool.iloc[a].episode_id.tolist(), pool.iloc[b].episode_id.tolist(), ep.iloc[test].episode_id.tolist())

def preprocessing(frame, columns):
    categorical = [c for c in columns if frame[c].dtype == 'object' or isinstance(frame[c].dtype, pd.StringDtype)]
    numeric = [c for c in columns if c not in categorical]
    return ColumnTransformer([('num', SimpleImputer(strategy='median', keep_empty_features=True, add_indicator=False), numeric), ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32), categorical)])

def probability(model, x, n_classes):
    original = model.predict_proba(x)
    result = np.full((len(x), n_classes), 1e-09, float)
    for j, c in enumerate(model.classes_):
        result[:, int(c)] = original[:, j]
    return result / result.sum(axis=1, keepdims=True)

def regression_prediction(model, x, task):
    return np.maximum(1.0 if task == 'time' else 0.0, model.predict(x))

def make_model(task, depth):
    kw = dict(max_iter=60, max_depth=depth, max_leaf_nodes=2 ** depth, learning_rate=0.08, early_stopping=False, random_state=SEED, min_samples_leaf=20)
    return HistGradientBoostingClassifier(**kw) if task in ['kind', 'direction'] else HistGradientBoostingRegressor(loss='absolute_error', **kw)

def categorical_metrics(y, p, episodes):
    predicted = p.argmax(axis=1)
    correct = (predicted == y).astype(float)
    return {'n': len(y), 'episodes': len(np.unique(episodes)), 'accuracy': float(correct.mean()) if len(y) else None, 'episode_accuracy': float(pd.DataFrame({'ep': episodes, 'correct': correct}).groupby('ep').correct.mean().mean()) if len(y) else None, 'log_loss': float(log_loss(y, p, labels=np.arange(p.shape[1]))) if len(y) else None}

def baseline(frame, train, classes):
    output = {}
    for task in TASKS:
        col = {'kind': 'kind_code', 'direction': 'direction_code', 'size': 'label_size', 'time': 'followup_min'}[task]
        good = train & frame[col].notna().to_numpy()
        if task in ['direction', 'size']:
            good &= frame.label_kind.isin(classes[:6]).to_numpy()
        g = frame.loc[good]
        if not len(g):
            output[task] = None
            continue
        categoric = task in ['kind', 'direction']
        width = 9 if task == 'kind' else 3

        def aggregate(v):
            if categoric:
                counts = np.bincount(v.astype(int), minlength=width).astype(float) + 1e-06
                return counts / counts.sum()
            return float(np.median(v))
        global_value = aggregate(g[col].to_numpy())
        maps = []
        for keys in [['pair', 'step', 'elapsed_bin'], ['pair'], ['step', 'elapsed_bin']]:
            maps.append((keys, {key if isinstance(key, tuple) else (key,): aggregate(v[col].to_numpy()) for key, v in g.groupby(keys, dropna=False)}))
        values = []
        for r in frame.itertuples():
            value = global_value
            for keys, mapping in maps:
                key = tuple((getattr(r, k) for k in keys))
                if key in mapping:
                    value = mapping[key]
                    break
            values.append(value)
        output[task] = np.asarray(values)
    return output

def fit_fold(dest, fold, scale_mode):
    episodes = pd.read_parquet(dest / str(record_path('q2_episodes_private_parquet')))
    frame = pd.read_parquet(dest / str(record_path('q2_decisions_private_parquet')))
    bundles = json.loads((dest / str(record_path('q2_bundles_private_json'))).read_text())
    meta = json.loads((dest / 'q2_build_manifest.json').read_text())
    _, train_ids, valid_ids, test_ids = next((s for s in splits(episodes, meta['smoke']) if s[0] == fold))
    train0 = frame.episode_id.isin(train_ids).to_numpy()
    valid0 = frame.episode_id.isin(valid_ids).to_numpy()
    test0 = frame.episode_id.isin(test_ids).to_numpy()
    scales = {}
    for j, role in enumerate(meta['roles']):
        if scale_mode == 's68':
            scales[role] = meta['frozen_s68_scales'][role]
        else:
            values = frame.loc[train0, f'hist__target_{j}'].dropna()
            scale = float(values.quantile(0.99) - values.quantile(0.01)) if len(values) else 0.0
            if scale <= 0:
                raise ValueError('Training-only q99-q1 scale is not identified for ' + role)
            scales[role] = scale
    frame = label_history(frame, bundles, scales, meta['roles'])
    frame['kind_code'] = frame.label_kind.map({v: i for i, v in enumerate(meta['classes'])})
    frame['direction_code'] = frame.label_direction.map({-1.0: 0, 0.0: 1, 1.0: 2})
    observed = ~frame.censored.to_numpy()
    train = train0 & observed
    valid = valid0 & observed
    test = test0 & observed
    final_train = (train0 | valid0) & observed
    if not train.any() or not valid.any() or (not test.any()):
        raise ValueError('Missing observed train/validation/test rows')
    pair_counts = episodes[episodes.episode_id.isin(train_ids + valid_ids)].pair.value_counts()
    frame['previous_pair_episodes'] = frame.pair.map(pair_counts).fillna(0).astype(int)
    frame['seen'] = frame.previous_pair_episodes.gt(0)
    frame['seen2'] = frame.previous_pair_episodes.ge(2)
    test_frame = frame.loc[test].copy().reset_index(drop=True)
    test_frame['kind_code'] = test_frame.kind_code.astype(int)
    pred = test_frame[['episode_id', 'pair', 'step', 'elapsed_bin', 'position', 'seen', 'seen2', 'previous_pair_episodes', 'kind_code', 'direction_code', 'label_size', 'followup_min']].copy()
    pred['fold'] = fold
    result = {'fold': fold, 'scale_mode': scale_mode, 'scales': scales, 'smoke': meta['smoke'], 'model_config': [], 'train_episodes': len(train_ids), 'validation_episodes': len(valid_ids), 'test_episodes': len(test_ids), 'test_seen_episodes': int(episodes[episodes.episode_id.isin(test_ids)].pair.isin(pair_counts.index).sum()), 'test_unseen_episodes': int((~episodes[episodes.episode_id.isin(test_ids)].pair.isin(pair_counts.index)).sum()), 'test_decisions': int(test0.sum()), 'test_censored': int((test0 & ~observed).sum()), 'test_observed': int(test.sum()), 'final_training_rows': int(final_train.sum())}
    base = baseline(frame, final_train, meta['classes'])
    for j in range(9):
        pred[f'R0_p{j}'] = base['kind'][test, j]
    for task in ['direction', 'size', 'time']:
        values = base[task]
        pred[f'R0_{task}'] = values[test].argmax(axis=1) if task == 'direction' and values is not None else values[test] if values is not None else np.nan
    preps = {}
    arrays = {}
    kind_models = {}
    cross = []
    model_dir = dest / str(record_path('models_private')) / fold
    model_dir.mkdir(parents=True, exist_ok=True)
    for layer in LAYERS:
        columns = [c for c in frame if c.startswith('recipe__') or (layer in ['R1', 'R2'] and c.startswith('hist__')) or (layer == 'R2' and c.startswith('state__'))]
        pre = preprocessing(frame, columns)
        xtrain = pre.fit_transform(frame.loc[train, columns]).astype('float32')
        xvalid = pre.transform(frame.loc[valid, columns]).astype('float32')
        final_pre = preprocessing(frame, columns)
        xfinal = final_pre.fit_transform(frame.loc[final_train, columns]).astype('float32')
        xtest = final_pre.transform(frame.loc[test, columns]).astype('float32')
        preps[layer] = final_pre
        arrays[layer] = xtest
        for task in TASKS:
            label = {'kind': 'kind_code', 'direction': 'direction_code', 'size': 'label_size', 'time': 'followup_min'}[task]
            masktrain = frame.loc[train, label].notna().to_numpy()
            maskvalid = frame.loc[valid, label].notna().to_numpy()
            maskfinal = frame.loc[final_train, label].notna().to_numpy()
            if task in ['direction', 'size']:
                masktrain &= frame.loc[train, 'label_kind'].isin(meta['roles']).to_numpy()
                maskvalid &= frame.loc[valid, 'label_kind'].isin(meta['roles']).to_numpy()
                maskfinal &= frame.loc[final_train, 'label_kind'].isin(meta['roles']).to_numpy()
            if not masktrain.any() or not maskvalid.any():
                pred[f'{layer}_{task}'] = np.nan
                result['model_config'].append({'layer': layer, 'task': task, 'status': 'insufficient_target_support'})
                continue
            ytrain = frame.loc[train, label].to_numpy()[masktrain].astype(int if task in ['kind', 'direction'] else float)
            yvalid = frame.loc[valid, label].to_numpy()[maskvalid].astype(int if task in ['kind', 'direction'] else float)
            best = None
            for depth in GRID:
                model = make_model(task, depth)
                model.fit(xtrain[masktrain], ytrain)
                if task in ['kind', 'direction']:
                    pp = probability(model, xvalid[maskvalid], 9 if task == 'kind' else 3)
                    value = (float(accuracy_score(yvalid, pp.argmax(axis=1))), -float(log_loss(yvalid, pp, labels=np.arange(pp.shape[1]))))
                else:
                    value = (-float(np.abs(regression_prediction(model, xvalid[maskvalid], task) - yvalid).mean()), 0.0)
                if best is None or value > best[0]:
                    best = (value, depth)
            depth = best[1]
            model = make_model(task, depth)
            model.fit(xfinal[maskfinal], frame.loc[final_train, label].to_numpy()[maskfinal].astype(int if task in ['kind', 'direction'] else float))
            joblib.dump({'preprocessor': final_pre, 'model': model, 'columns': columns, 'scales': scales, 'spec': 'S77 v0.2', 'layer': layer, 'task': task}, model_dir / f'{layer}_{task}.joblib')
            if task in ['kind', 'direction']:
                pp = probability(model, xtest, 9 if task == 'kind' else 3)
                if task == 'kind':
                    for j in range(9):
                        pred[f'{layer}_p{j}'] = pp[:, j]
                    kind_models[layer] = model
                else:
                    pred[f'{layer}_direction'] = pp.argmax(axis=1)
            else:
                pred[f'{layer}_{task}'] = regression_prediction(model, xtest, task)
            result['model_config'].append({'layer': layer, 'task': task, 'depth': depth, 'iterations': 60, 'selection_score': best[0]})
        print(json.dumps({'fold': fold, 'layer': layer, 'completed_tasks': 4}), flush=True)
    for layer, prefix, reference in [('R2', 'state__', 'R1'), ('R1', 'hist__', 'R0p')]:
        names = preps[layer].get_feature_names_out()
        block = np.array([prefix in name for name in names])
        assert block.any()
        original_p = pred[[f'{layer}_p{j}' for j in range(9)]].to_numpy()
        reference_p = pred[[f'{reference}_p{j}' for j in range(9)]].to_numpy()
        truth = pred.kind_code.to_numpy()
        rng = np.random.default_rng(SEED)
        for repeat in range(50):
            strict, strict_changed = exchange_indices(test_frame, rng)
            loose, loose_changed = exchange_indices(test_frame, rng, ignore_stage=True)
            for scheme, donor, changed in [('pair_stage', strict, strict_changed), ('relax_unmatched', np.where(strict_changed, strict, loose), strict_changed | loose_changed)]:
                effective = np.zeros(len(pred), bool)
                if changed.any():
                    altered = arrays[layer].copy()
                    altered[:, block] = arrays[layer][donor][:, block]
                    effective = changed & np.any(altered != arrays[layer], axis=1)
                    pp = probability(kind_models[layer], altered, 9)
                    swap_accuracy = float((pp.argmax(axis=1)[changed] == truth[changed]).mean())
                    original_accuracy = float((original_p.argmax(axis=1)[changed] == truth[changed]).mean())
                    reference_accuracy = float((reference_p.argmax(axis=1)[changed] == truth[changed]).mean())
                else:
                    swap_accuracy = original_accuracy = reference_accuracy = None
                cross.append({'fold': fold, 'layer': layer, 'scheme': scheme, 'repeat': repeat, 'test_rows': len(pred), 'exchanged_rows': int(changed.sum()), 'exchange_fraction': float(changed.mean()), 'effective_value_changed_rows': int(effective.sum()), 'effective_value_changed_fraction': float(effective.mean()), 'effective_among_assigned': float(effective.sum() / changed.sum()) if changed.any() else None, 'original_accuracy': original_accuracy, 'swapped_accuracy': swap_accuracy, 'reference_accuracy': reference_accuracy, 'swapped_minus_reference': swap_accuracy - reference_accuracy if changed.any() else None})
    chance = {}
    for stratum in ['all', 'seen', 'seen2', 'unseen']:
        f = frame.loc[final_train]
        if stratum == 'seen2':
            f = f[f.seen2]
        if stratum == 'unseen':
            f = f.iloc[:0]
        fallback = not len(f)
        if fallback:
            f = frame.loc[final_train]
        counts = np.bincount(f.kind_code.astype(int), minlength=9)
        chance[stratum] = {'kind': int(np.argmax(counts)), 'training_rows': len(f), 'fallback_global': fallback}
    result['training_majority'] = chance
    km_rows = []
    censor = []
    test_all = frame.loc[test0]
    for stratum, sub in [('all', test_all), ('seen', test_all[test_all.seen]), ('seen2', test_all[test_all.seen2]), ('unseen', test_all[~test_all.seen])]:
        records, median = kaplan_meier(sub.followup_min, ~sub.censored)
        km_rows.extend(({'fold': fold, 'stratum': stratum, **r} for r in records))
        censor.append({'fold': fold, 'stratum': stratum, 'decisions': len(sub), 'censored': int(sub.censored.sum()), 'KM_median_min': median})
    output = dest / 'folds'
    output.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(output / f'{fold}_predictions_PRIVATE.parquet', index=False)
    frame.loc[test0, ['episode_id', 'pair', 'seen', 'seen2', 'followup_min', 'censored']].assign(fold=fold).to_parquet(output / f'{fold}_survival_PRIVATE.parquet', index=False)
    pd.DataFrame(cross).to_csv(output / f'{fold}_exchanges.csv', index=False)
    pd.DataFrame(km_rows).to_csv(output / f'{fold}_KM.csv', index=False)
    pd.DataFrame(censor).to_csv(output / f'{fold}_censoring.csv', index=False)
    (output / f'{fold}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({'fold_complete': fold, 'test_observed': int(test.sum()), 'test_censored': int((test0 & ~observed).sum())}), flush=True)

def audit_effective_exchange(dest, fold):
    frame = pd.read_parquet(dest / str(record_path('q2_decisions_private_parquet')))
    bundles = json.loads((dest / str(record_path('q2_bundles_private_json'))).read_text())
    meta = json.loads((dest / 'q2_build_manifest.json').read_text())
    manifest = json.loads((dest / 'folds' / f'{fold}.json').read_text())
    frame = label_history(frame, bundles, manifest['scales'], meta['roles'])
    pred = pd.read_parquet(dest / 'folds' / f'{fold}_predictions_PRIVATE.parquet')
    test = frame[~frame.censored & frame.episode_id.isin(pred.episode_id.unique())].copy().reset_index(drop=True)
    assert len(test) == len(pred) and np.array_equal(test.position, pred.position)
    old = pd.read_csv(dest / 'folds' / f'{fold}_exchanges.csv')
    rows = []
    for layer, prefix in [('R2', 'state__'), ('R1', 'hist__')]:
        model = joblib.load(dest / str(record_path('models_private')) / fold / f'{layer}_kind.joblib')
        pre = model['preprocessor']
        original = pre.transform(test[model['columns']]).astype('float32')
        block = np.array([prefix in name for name in pre.get_feature_names_out()])
        rng = np.random.default_rng(SEED)
        for repeat in range(50):
            strict, changed = exchange_indices(test, rng)
            loose, loose_changed = exchange_indices(test, rng, ignore_stage=True)
            for scheme, donor, assigned in [('pair_stage', strict, changed), ('relax_unmatched', np.where(changed, strict, loose), changed | loose_changed)]:
                effective = assigned & np.any(original[:, block] != original[donor][:, block], axis=1)
                reference = old[old.layer.eq(layer) & old.scheme.eq(scheme) & old.repeat.eq(repeat)].iloc[0]
                assert int(reference.exchanged_rows) == int(assigned.sum())
                rows.append({'fold': fold, 'layer': layer, 'scheme': scheme, 'repeat': repeat, 'effective_value_changed_rows': int(effective.sum()), 'effective_value_changed_fraction': float(effective.mean()), 'effective_among_assigned': float(effective.sum() / assigned.sum()) if assigned.any() else None})
    extra = pd.DataFrame(rows)
    old = old.drop(columns=[c for c in extra if c.startswith('effective_') and c in old])
    old.merge(extra, on=['fold', 'layer', 'scheme', 'repeat'], validate='one_to_one').to_csv(dest / 'folds' / f'{fold}_exchanges.csv', index=False)
    return fold
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--only', required=True)
    p.add_argument('--scales', choices=['train', 's68'], required=True)
    a = p.parse_args()
    fit_fold(OUT / 'smoke' if a.smoke else OUT, a.only, a.scales)
