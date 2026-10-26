from __future__ import annotations
import sys as _runtime_sys
from pathlib import Path as _RuntimePath
_runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[2]))
from audit_runtime import PACKAGE_ROOT, WORKSPACE, record_path, require_records
if __name__ == "__main__":
    require_records()
import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from s77_common import ROOT, OUT, SEED, BOOTSTRAPS, kaplan_meier

def paired_interval(diff, episode):
    g = pd.DataFrame({'diff': diff, 'episode': episode}).groupby('episode')['diff'].agg(['sum', 'count'])
    if len(g) < 2:
        return [None, None]
    rng = np.random.default_rng(SEED)
    ix = rng.integers(0, len(g), (BOOTSTRAPS, len(g)))
    values = g['sum'].to_numpy()[ix].sum(axis=1) / g['count'].to_numpy()[ix].sum(axis=1)
    return np.quantile(values, [0.025, 0.975]).tolist()

def strata(frame):
    return [('all', frame), ('seen', frame[frame.seen]), ('seen2', frame[frame.seen2]), ('unseen', frame[~frame.seen])]

def q2_tables(dest):
    manifests = {p.stem: json.loads(p.read_text()) for p in (dest / 'folds').glob('*.json')}
    if set(manifests) != {'time', 'group0', 'group1', 'group2', 'group3', 'group4'}:
        raise ValueError('Six completed fold manifests required')
    frames = [pd.read_parquet(dest / 'folds' / f'{fold}_predictions_PRIVATE.parquet') for fold in manifests]
    predictions = pd.concat(frames, ignore_index=True)
    predictions['split'] = np.where(predictions.fold.eq('time'), 'time', 'group')
    classes = json.loads((dest / 'q2_build_manifest.json').read_text())['classes']
    metric_rows = []
    pr_rows = []
    comparisons = []
    for split, part in predictions.groupby('split'):
        for stratum, g in strata(part):
            if not len(g):
                continue
            y = g.kind_code.to_numpy(int)
            ep = g.episode_id.to_numpy()
            null = np.array([manifests[fold]['training_majority'][stratum]['kind'] for fold in g.fold])
            for model in ['R0', 'R0p', 'R1', 'R2']:
                prob = g[[f'{model}_p{j}' for j in range(9)]].to_numpy()
                p = prob.argmax(axis=1)
                correct = p == y
                record = {'split': split, 'stratum': stratum, 'model': model, 'decisions': len(g), 'episodes': len(np.unique(ep)), 'accuracy': float(correct.mean()), 'episode_accuracy': float(pd.DataFrame({'ep': ep, 'correct': correct}).groupby('ep').correct.mean().mean()), 'log_loss': float(log_loss(y, prob, labels=np.arange(9))), 'training_majority_accuracy': float((null == y).mean())}
                for task, target in [('size', 'label_size'), ('time', 'followup_min')]:
                    valid = g[target].notna() & g[f'{model}_{task}'].notna()
                    record[task + '_n'] = int(valid.sum())
                    record[task + '_mae'] = float(np.abs(g.loc[valid, f'{model}_{task}'] - g.loc[valid, target]).mean()) if valid.any() else None
                target = g.label_size.notna() & g[f'{model}_direction'].notna()
                record['direction_n'] = int(target.sum())
                record['direction_accuracy'] = float((g.loc[target, f'{model}_direction'] == g.loc[target, 'direction_code']).mean()) if target.any() else None
                metric_rows.append(record)
                for j, kind in enumerate(classes):
                    hits = int(((p == j) & (y == j)).sum())
                    called = int((p == j).sum())
                    actual = int((y == j).sum())
                    pr_rows.append({'split': split, 'stratum': stratum, 'model': model, 'kind': kind, 'hits': hits, 'predicted': called, 'actual': actual, 'precision': hits / called if called else None, 'recall': hits / actual if actual else None})
            for stronger, weaker in [('R2', 'R1'), ('R1', 'R0p')]:
                a = g[[f'{stronger}_p{j}' for j in range(9)]].to_numpy().argmax(axis=1) == y
                b = g[[f'{weaker}_p{j}' for j in range(9)]].to_numpy().argmax(axis=1) == y
                diff = a.astype(float) - b.astype(float)
                ci = paired_interval(diff, ep)
                comparisons.append({'split': split, 'stratum': stratum, 'contrast': stronger + '-' + weaker, 'decisions': len(g), 'episodes': len(np.unique(ep)), 'accuracy_difference': float(diff.mean()), 'ci_low': ci[0], 'ci_high': ci[1], 'episode_difference': float(pd.DataFrame({'ep': ep, 'diff': diff}).groupby('ep')['diff'].mean().mean())})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(dest / 'q2_metrics.csv', index=False)
    pd.DataFrame(pr_rows).to_csv(dest / 'q2_class_precision_recall.csv', index=False)
    comparison = pd.DataFrame(comparisons)
    categories = []
    for r in comparison.itertuples():
        other = comparison[comparison.split.eq('group') & comparison.stratum.eq(r.stratum) & comparison.contrast.eq(r.contrast)]
        same_direction = bool(len(other) and other.accuracy_difference.iloc[0] > 0)
        label = '보류'
        if r.decisions >= 100 and pd.notna(r.ci_low) and (r.accuracy_difference >= 0.05) and (r.ci_low > 0) and same_direction:
            label = '측정 상태의 추가 이득 지지' if r.contrast == 'R2-R1' else '설정·행동 상태의 추가 이득 지지'
        elif r.decisions >= 100 and pd.notna(r.ci_high) and (r.ci_high < 0.05):
            label = '등록된 추가 이득 5 %p 미달'
        if r.split != 'time' or r.stratum != 'seen':
            label = '서술'
        elif label == '등록된 추가 이득 5 %p 미달':
            label = '등록된 측정 상태 추가 이득 5 %p 미달(해당 입력·학습기·지표)' if r.contrast == 'R2-R1' else '등록된 설정·행동 상태 추가 이득 5 %p 미달(해당 입력·학습기·지표)'
        categories.append(label)
    comparison['mechanical_category'] = categories
    comparison['is_primary'] = comparison.split.eq('time') & comparison.stratum.eq('seen')
    comparison.to_csv(dest / 'q2_paired_comparisons.csv', index=False)
    comparison[comparison.is_primary].to_csv(dest / 'q2_registered_mechanical.csv', index=False)
    auxiliary = []
    for split in metrics.split.unique():
        for stratum in metrics.stratum.unique():
            subset = metrics[metrics.split.eq(split) & metrics.stratum.eq(stratum)].set_index('model')
            if not len(subset):
                continue
            for high, low in [('R2', 'R1'), ('R1', 'R0p')]:
                for task in ['size', 'time']:
                    weak = subset.loc[low, task + '_mae']
                    strong = subset.loc[high, task + '_mae']
                    gain = 1 - strong / weak if pd.notna(weak) and weak > 0 and pd.notna(strong) else None
                    auxiliary.append({'split': split, 'stratum': stratum, 'contrast': high + '-' + low, 'task': task, 'weaker_mae': weak, 'stronger_mae': strong, 'relative_mae_reduction': gain, 'reduction_ge_10pct': gain >= 0.1 if gain is not None else None})
    pd.DataFrame(auxiliary).to_csv(dest / 'q2_auxiliary_comparisons.csv', index=False)
    censor = []
    km = []
    survival = pd.concat([pd.read_parquet(dest / 'folds' / f'{fold}_survival_PRIVATE.parquet') for fold in manifests], ignore_index=True)
    survival['split'] = np.where(survival.fold.eq('time'), 'time', 'group')
    for split, part in survival.groupby('split'):
        for stratum, g in strata(part):
            records, median = kaplan_meier(g.followup_min, ~g.censored)
            km.extend(({'split': split, 'stratum': stratum, **r} for r in records))
            censor.append({'split': split, 'stratum': stratum, 'decisions': len(g), 'episodes': g.episode_id.nunique(), 'observed_episodes': g.loc[~g.censored, 'episode_id'].nunique(), 'only_censored_episodes': g.episode_id.nunique() - g.loc[~g.censored, 'episode_id'].nunique(), 'censored': int(g.censored.sum()), 'censor_fraction': float(g.censored.mean()) if len(g) else None, 'KM_median_min': median})
    reason_map = pd.read_parquet(ROOT / 'artifacts/transition/episodes.parquet', columns=['episode_id', 'end_reason']).set_index('episode_id').end_reason
    survival['boundary_reason'] = survival.episode_id.map(reason_map)
    reason_rows = []
    for split, part in survival.groupby('split'):
        for stratum, g in strata(part):
            for reason, subset in g[g.censored].groupby('boundary_reason'):
                reason_rows.append({'split': split, 'stratum': stratum, 'boundary_reason': reason, 'censored_decisions': len(subset), 'episodes': subset.episode_id.nunique()})
    pd.DataFrame(reason_rows).to_csv(dest / 'q2_censoring_reasons.csv', index=False)
    pd.DataFrame(censor).to_csv(dest / 'q2_censoring.csv', index=False)
    pd.DataFrame(km).to_csv(dest / 'q2_KM.csv', index=False)
    for fold in manifests:
        path = dest / 'folds' / f'{fold}_exchanges.csv'
        if 'effective_value_changed_fraction' not in pd.read_csv(path, nrows=0).columns:
            from s77_q2_fit import audit_effective_exchange
            audit_effective_exchange(dest, fold)
    exchange = pd.concat([pd.read_csv(dest / 'folds' / f'{fold}_exchanges.csv') for fold in manifests], ignore_index=True)
    exchange.to_csv(dest / 'q2_exchanges.csv', index=False)
    exchange.groupby(['fold', 'layer', 'scheme'])[['exchange_fraction', 'original_accuracy', 'swapped_accuracy', 'reference_accuracy', 'swapped_minus_reference', 'effective_value_changed_fraction', 'effective_among_assigned']].mean().reset_index().to_csv(dest / 'q2_exchange_summary.csv', index=False)
    return (manifests, metrics, comparison)

def q1_mechanical(dest):
    frame = pd.read_csv(dest / 'q1_summary.csv')
    out = []
    for r in frame[frame.population.eq('main_no_cross') & frame.cap.eq('full')].itertuples():
        capped = frame[frame.population.eq(r.population) & frame.cap.eq('cap30') & frame.loop.eq(r.loop) & frame.region.eq(r.region) & frame.side.eq(r.side)]
        direction = bool(len(capped) and capped.delta.iloc[0] > 0)
        label = '보류'
        if r.windows >= 30 and pd.notna(r.ci_low):
            if r.delta >= 0.1 and r.ci_low > 0 and direction:
                label = '묶음 지지'
            elif r.ci_low > 0:
                label = '양의 연관이나 10 %p 기준 미달'
            else:
                label = '우연 동반과 구별되지 않음'
        if r.region not in ['정상 생산', '지종 전환']:
            label = '서술'
        out.append({'loop': r.loop, 'region': r.region, 'side': r.side, 'windows': r.windows, 'delta': r.delta, 'ci_low': r.ci_low, 'ci_high': r.ci_high, 'cap30_positive': direction, 'mechanical_category': label})
    pd.DataFrame(out).to_csv(dest / 'q1_mechanical.csv', index=False)

def markdown(frame):

    def cell(v):
        if pd.isna(v):
            return '—'
        if isinstance(v, (float, np.floating)):
            return str(int(v)) if float(v).is_integer() else f'{v:.4f}'
        return str(v).replace('|', '/').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(frame.columns) + ' |', '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'] + ['| ' + ' | '.join((cell(v) for v in row)) + ' |' for row in frame.itertuples(index=False, name=None)])

def figures(dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(ROOT / 'scripts/s74'))
    import paper_figstyle as ps
    sys.path.insert(0, str(PACKAGE_ROOT / 'scripts/paper1_figs'))
    from bbox_overlap_check import audit_fig
    layout_audits = []

    def check_figure(fig, name):
        findings = audit_fig(fig, verbose=False)
        critical = [r for r in findings if r.severity.lower() == 'critical']
        if critical:
            raise ValueError('Figure layout overlap: ' + name)
        layout_audits.append({'figure': name, 'critical': 0, 'warnings': len(findings)})
    q1 = pd.read_csv(dest / 'q1_summary.csv')
    q1 = q1[q1.population.eq('main_no_cross') & q1.loop.eq('pooled') & q1.cap.eq('full') & q1.region.isin(['정상 생산', '지종 전환'])]
    ps.apply(width='text', aspect=0.65)
    fig, ax = plt.subplots()
    q1 = q1.sort_values(['region', 'side'])
    y = q1.delta.to_numpy()
    low = q1.ci_low.to_numpy()
    high = q1.ci_high.to_numpy()
    ax.errorbar(np.arange(len(y)), y, yerr=np.array([np.maximum(0, y - low), np.maximum(0, high - y)]), fmt='o', color='#0072B2', capsize=3)
    ax.axhline(0, color='gray', ls=':')
    ax.axhline(0.1, color='#D55E00', ls='--')
    ax.set(xticks=np.arange(len(y)), xticklabels=[('Normal' if r.region == '정상 생산' else 'Transition') + '\n' + r.side for r in q1.itertuples()], ylabel='Observed minus LOT-shift chance')
    plt.rcParams['savefig.pad_inches'] = 0.05
    check_figure(fig, 'q1_excess')
    ps.save(fig, str(dest / 'q1_excess'), dpi=240)
    plt.close(fig)
    d = pd.read_csv(dest / 'q2_paired_comparisons.csv')
    d = d[d.stratum.eq('seen')].sort_values(['split', 'contrast'])
    ps.apply(width='text', aspect=0.65)
    fig, ax = plt.subplots()
    v = d.accuracy_difference.to_numpy()
    ax.errorbar(np.arange(len(d)), v, yerr=np.array([np.maximum(0, v - d.ci_low), np.maximum(0, d.ci_high - v)]), fmt='o', color='#009E73', capsize=3)
    ax.axhline(0.05, color='#D55E00', ls='--')
    ax.axhline(0, color='gray', ls=':')
    ax.set(xticks=np.arange(len(d)), xticklabels=[r.split + '\n' + r.contrast for r in d.itertuples()], ylabel='Top-1 accuracy difference')
    plt.rcParams['savefig.pad_inches'] = 0.05
    check_figure(fig, 'q2_accuracy_gain')
    ps.save(fig, str(dest / 'q2_accuracy_gain'), dpi=240)
    plt.close(fig)
    km = pd.read_csv(dest / 'q2_KM.csv')
    ps.apply(width='text', aspect=0.65)
    fig, ax = plt.subplots()
    for name, color, ls in [('seen', '#0072B2', '-'), ('unseen', '#D55E00', '--')]:
        k = km[km.split.eq('time') & km.stratum.eq(name)]
        ax.step(np.r_[0, k.time], np.r_[1, k.survival], where='post', label=name, color=color, linestyle=ls)
    ax.set(xlabel='Time until next bundle starts (min)', ylabel='Kaplan–Meier survival', ylim=(0, 1.02))
    ax.legend(loc='upper right')
    plt.rcParams['savefig.pad_inches'] = 0.05
    check_figure(fig, 'q2_survival')
    ps.save(fig, str(dest / 'q2_survival'), dpi=240)
    plt.close(fig)
    (dest / 'figure_layout_audit.json').write_text(json.dumps(layout_audits, indent=2))

def run(dest):
    q1_mechanical(dest)
    manifests, metrics, comparison = q2_tables(dest)
    figures(dest)
    q1 = pd.read_csv(dest / 'q1_summary.csv')
    q3 = pd.read_csv(dest / 'q3_summary.csv')
    build = json.loads((dest / 'q2_build_manifest.json').read_text())
    q1main = q1[q1.population.eq('main_no_cross') & q1.loop.eq('pooled') & q1.region.isin(['정상 생산', '지종 전환'])]
    selected = metrics[metrics.split.eq('time') & metrics.stratum.isin(['seen', 'seen2', 'unseen'])][['stratum', 'model', 'decisions', 'episodes', 'accuracy', 'episode_accuracy', 'log_loss', 'direction_accuracy', 'size_mae', 'time_mae', 'training_majority_accuracy']]
    detector = json.loads((dest / 'q3_detector.json').read_text())
    qm = json.loads((dest / 'q1_manifest.json').read_text())
    lines = ['# 기록된 감독 사건의 묶음·레시피 통제 감사', '', '## 자료와 정보 조건', '', 'Q1·Q3는 결과 조건 없는 모든 복귀 관측 수동 창을 사용하고, LOT를 가로지르는 창은 주 분석에서 제외했다. 앞 목표 변경은 [e−5,e−1], 창 안 변경은 [e,복귀)이며 모두 S68 여섯 품질 목표 후보 정의를 사용한다. 30분 상한, LOT 가로지름 포함, 빠른 왕복 제외 표를 별도로 저장했다.', '', f"복귀 관측 창은 {qm['returned_windows']}개이고 정상 생산 시작은 {qm['normal_returned']}개다. 주 분석 창은 {qm['main_windows']}개다. 대조가 없는 창은 이동 우연에는 포함하고 자동 대조 분모에서만 제외했다. Q1 원형 이동은 여섯 열에 같은 LOT별 이동량을 적용한 것과 동등한 공동 합집합 이동 200회다. LOT 부트스트랩은 2,000회다.", '', '표준화 q99−q1은 각 분할의 내부 학습 결정 시점에서 관측된 목표값만으로 계산하고 검증·시험과 최종 재적합에도 고정한다. Q2의 목적 지종은 시작에 알려졌다고 가정하는 레시피를 아는 조건의 재현이다. 동일 분 모든 종류를 묶고 최대 길이가 끝난 시각에만 최종 크기를 이력에 넣는다. 주 종류는 목표의 표준화 크기 최대, 모드, 수동 출력, DCS 순이며 목표 동률은 고정 루프 순이다.', '', 't0에는 이력이 빈 행을 한 개 둔다. 같은 분 완료 기록은 t0 행과 중복시키지 않고 이후 완료 시각은 분당 한 행으로 둔다. 다음 묶음은 결정 시각보다 엄격히 늦게 시작하고, 완료가 관찰 경계 밖이면 절단한다. 절단된 행에는 종료 종류를 부여하지 않고 분류·크기·방향·시간 회귀에서 제외한다. 시간은 절단을 포함한 Kaplan–Meier도 병기한다.', '', f"에피소드는 {build['episodes']}개이고 행동이 없는 {build['no_action_episodes']}개는 Q2에서 제외했다. 결정 행은 {build['decision_rows']}개, 절단 행은 {build['censored_rows']}개다.", '', 'R0는 1e−6 가산 평활한 쌍·단계 빈도와 중앙값이며 지원이 없으면 쌍, 단계, 전체 순으로 대체한다. R0′·R1·R2는 깊이 2·3, 60회, 학습률 0.08의 같은 부스팅 격자를 사용한다. 분류는 검증 정확도 최대·로그 손실 동률 해소, 회귀는 검증 MAE 최소로 선택한다. 내부 선택 뒤 학습·검증을 합쳐 재적합한다. 대치·범주 처리는 해당 학습 자료에만 적합하고 결측 지시자를 추가하지 않는다.', '', 'R1에는 완료 묶음 이력·현재 11개 모드·현재 여섯 목표·과거 차분의 목표 변경 경과를 준다. R2에는 편차·기울기·부호 지속·속도·압력 측정 상태만 더한다. 전력, 전력 결측 지시자, 지종 전환 표지, 최종 지속·행동 수는 입력에서 제외했다.', '', '시간순 내부 학습은 1~6월, 검증은 7~8월, 시험은 9~11월이다. 그룹 5겹은 LOT를 공유하는 모든 에피소드를 연결해 같은 겹에 두고, 각 학습 풀 안에 검증 그룹 20%를 둔다. seen은 최종 학습에 같은 쌍의 이전 에피소드가 있고, seen2는 두 번 이상인 경우다. 최빈 종류 우연은 학습 자료에서만 구하고 unseen은 전체 학습 최빈값으로 대체한다.', '', '## Q1 동반율과 대조', '', markdown(q1main[['region', 'cap', 'side', 'windows', 'p_observed', 'p_chance', 'delta', 'ci_low', 'ci_high', 'matched_controls']]), '', '길이 대응 자동 대조는 대응 창이 있는 동일한 수동 창 분모에서 비교했다.', '', markdown(q1main[['region', 'cap', 'side', 'matched_controls', 'p_observed_matched', 'p_auto_control', 'matched_control_delta', 'matched_ci_low', 'matched_ci_high']]), '', '## Q2 종류·방향·크기·간격의 분할 결과', '', markdown(selected), '', markdown(comparison), '', markdown(pd.read_csv(dest / 'q2_censoring.csv')), '', '정확도 주 값은 결정 시점 가중이고 보조 값은 에피소드별 정확도의 평균이다. 비교 구간은 같은 시험 결정 행의 정확도 차를 에피소드 단위로 2,000회 함께 재표집한 95% 구간이다. 종류별 적중·예측·정답 분모와 정밀도·재현율은 q2_class_precision_recall.csv, 크기·간격 MAE의 상대 감소와 10% 대조는 q2_auxiliary_comparisons.csv, 절단 수와 KM 중앙값은 q2_censoring.csv에 저장했다.', '', '설정·행동 상태 교환은 R1, 측정 상태 교환은 R2에 적용했다. 같은 쌍×묶음 순번×경과 구간 셀에서 다른 에피소드의 벡터를 교환한다. 상대가 없는 행에는 같은 쌍에서 단계를 무시한 교환을 추가한 민감도를 계산했다. 50회 교환의 다른 에피소드 벡터를 배정받은 분율·실제 값이 달라진 분율과 교환 가능한 부분집합의 정확도를 q2_exchange_summary.csv에 기록했다.', '', '## Q3 뒤따르는 지종 변경과 검출기', '', markdown(q3), '', f"검출기는 완료된 목표·모드 묶음의 과거 5·15·60분 수와 마지막 완료 뒤 경과만 사용했다. 정상 생산 분의 시간순 시험 {detector['test_minutes']}분에서 정밀도 {detector['precision']:.4f}, 재현율 {detector['recall']:.4f}였다. 정상 생산 표지는 평가 모집단을 정하는 사후 조건이며 입력에 넣지 않았다. 검증 분 F1으로 문턱을 정했다.", '', '## 기계적 대조와 그림', '', 'Q1은 q1_mechanical.csv, Q2는 시간순 seen 주 층의 q2_registered_mechanical.csv에서 등록 범주를 위에서부터 적용했다. 다른 Q2 층과 그룹 분할은 서술 표로 제시한다. 이 표는 수치와 등록 문턱의 기계적 대조다. q1_excess는 pooled 주 분석의 동반율 초과와 LOT 구간, q2_accuracy_gain은 seen의 두 입력 비교와 에피소드 구간, q2_survival은 시간순 다음 묶음 간격의 절단 포함 KM 곡선이다. 그림 폭은 5.5인치다.', '', '## 가정과 확인 범위', '', '2022 자료는 이전 분석에서 반복 열람한 자료이며 설계 고정 뒤의 분할 검증이다. 기록된 목표·모드·후보 사건의 조건부 연관을 집계했다. 공동 원형 이동은 LOT 안 빈도와 동시 변경을 보존하고, 서로 다른 LOT의 이동량은 독립이다. 앞 5분이 이전 LOT와 겹치면 해당 LOT의 이동을 함께 반영한다.', '', '교환 가능한 부분집합과 전체 시험의 지지 영역은 다를 수 있다. 목표 척도와 모든 선택은 fold manifest에 기록했다. 실제 생산계획 원문의 이용 가능 시점은 확인되지 않았다. 시간순 시험에서 목적 지종을 아는 조건과 실제 운전 시점의 정보 조건은 구별한다.', '', '대상 버전은 S77 v0.2다. 이번 공백은 묶음 이용 가능 시점·절단·같은 학습기 입력 비교였다. 시점·교환·생존 정답 예제와 행·모형·교환 수를 검사했다. S70·S74의 기존 분 정렬과 측정 특성 시점 검증을 재사용했다. 이 보고서는 수치와 정의를 기록하며 방향·효용 판단을 붙이지 않는다.', '']
    path = dest / 'smoke_report.md' if build['smoke'] else ROOT / 'generated/notes/S77_audit_W4.md'
    path.write_text('\n'.join(lines))
    status = {'folds': len(manifests), 'observed_prediction_rows': int(metrics[metrics.model.eq('R2') & metrics.stratum.eq('all')].decisions.sum()), 'new_model_folds': len(manifests), 'spec_version': '0.2', 'scale_mode': manifests['time']['scale_mode'], 'time_test_episodes': manifests['time']['test_episodes'], 'time_seen_episodes': manifests['time']['test_seen_episodes'], 'time_unseen_episodes': manifests['time']['test_unseen_episodes']}
    (dest / 'summary.json').write_text(json.dumps(status, ensure_ascii=False, indent=2))
    print(json.dumps(status, ensure_ascii=False), flush=True)
    return status
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    run(OUT / 'smoke' if a.smoke else OUT)
