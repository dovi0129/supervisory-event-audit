from __future__ import annotations
import sys as _runtime_sys
from pathlib import Path as _RuntimePath
_runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[2]))
from audit_runtime import PACKAGE_ROOT, WORKSPACE, record_path, require_records
if __name__ == "__main__":
    require_records()
import json
import pandas as pd
from s77_common import ROOT, OUT
DEST = OUT / 's89'
REPORT = ROOT / 'generated/notes/S89_reviewer_computations.md'
MODEL = {'majority': '최빈', 'markov': '1차 전이·무쌍', 'markov_pair': '1차 전이·쌍', 'R0p': 'R0′', 'R1_history': 'R1-이력만', 'R1': 'R1', 'R2': 'R2', 'R1_deep': 'R1·깊이6/500', 'R2_deep': 'R2·깊이6/500', 'tree': '기존 나무', 'recent_target_3min': '최근 목표3분', 'recent_target_5min': '최근 목표5분'}
SPLIT = {'time': '시간순', 'group': '그룹'}
WEIGHT = {'decision': '결정', 'episode': '에피소드'}
METRIC = {'accuracy': '정확도', 'balanced_accuracy': '균형 정확도', 'macro_f1': 'macro-F1', 'log_loss': '로그 손실'}
BASE = ['majority', 'markov', 'markov_pair', 'R0p', 'R1_history', 'R1', 'R2']

def number(value):
    return '—' if pd.isna(value) else f'{value:.4f}'

def interval(row):
    return f'{row.difference:+.4f} [{row.ci_low:+.4f}, {row.ci_high:+.4f}]'

def table(frame, filename):
    frame.to_csv(DEST / filename, index=False)
    display = frame.copy()
    for col in display:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(number)
    return display.to_markdown(index=False, disable_numparse=True) + f'\n\nCSV: `artifacts/stage1_audit/s89/{filename}`.\n'

def metric_table(metrics, split, models):
    part = metrics[(metrics.split == split) & (metrics.stratum == 'seen') & metrics.model.isin(models)].copy()
    part['order'] = part.model.map(dict(zip(models, range(len(models)))))
    part = part.sort_values(['weighting', 'order'])
    rows = []
    for r in part.itertuples():
        rows.append({'가중': WEIGHT[r.weighting], '모형': MODEL[r.model], **{v: getattr(r, k) for k, v in METRIC.items()}})
    return pd.DataFrame(rows)

def contrast_table(comparisons, contrasts, splits=('time', 'group')):
    rows = []
    for split in splits:
        for weight in WEIGHT:
            for contrast in contrasts:
                p = comparisons[(comparisons.split == split) & (comparisons.stratum == 'seen') & (comparisons.weighting == weight) & (comparisons.contrast == contrast)]
                rows.append({'분할': SPLIT[split], '가중': WEIGHT[weight], '차이': contrast, **{label: interval(p[p.metric.eq(metric)].iloc[0]) for metric, label in METRIC.items()}})
    return pd.DataFrame(rows)

def main():
    metrics = pd.read_csv(DEST / 'q2_metrics.csv')
    comparisons = pd.read_csv(DEST / 'q2_paired_comparisons.csv')
    phase = pd.read_csv(DEST / 'q2_initial_subsequent.csv')
    controls = pd.read_csv(DEST / 'q1_elapsed_controls.csv')
    rules = pd.read_csv(DEST / 'recent_target_rule_metrics.csv')
    auto = pd.read_csv(DEST / 'q1_automatic_paired.csv')
    pre = pd.read_csv(DEST / 'pre30_bundle_distribution.csv')
    control_manifest = json.loads((DEST / 'controls_manifest.json').read_text())
    verification = json.loads((DEST / 'q2_verification.json').read_text())

    def comp(contrast, split='time', weight='decision', metric='accuracy'):
        return comparisons[(comparisons.split == split) & (comparisons.stratum == 'seen') & (comparisons.weighting == weight) & (comparisons.contrast == contrast) & (comparisons.metric == metric)].iloc[0]
    history = comp('R1-R1_history')
    first = comp('R1_history-markov')
    deep = comp('R2_deep-R1_deep')
    initial = phase[(phase.split == 'time') & (phase.stratum == 'seen') & (phase.phase == 'initial') & (phase.weighting == 'decision') & (phase.metric == 'accuracy') & (phase.contrast == 'R2-R1')].iloc[0]
    elapsed = controls[(controls.loop == 'pooled') & (controls.cap == 'full') & (controls.side == 'inside') & (controls.elapsed_phase == 'all') & (controls.method == 'other_episode_elapsed')].iloc[0]
    reductions = {'elapsed_minus_original': float(elapsed.delta - elapsed.original_lot_delta_on_same_windows), 'elapsed_reduction_fraction': float(1 - elapsed.delta / elapsed.original_lot_delta_on_same_windows), 'elapsed_over_q1_delta': float(elapsed.delta / 0.1), 'history_increment_over_q2_delta': float(history.difference / 0.05), 'deep_increment_over_q2_delta': float(deep.difference / 0.05)}
    (DEST / 'report_derived_values.json').write_text(json.dumps(reductions, indent=2))
    lines = ['# S89 심사 답변용 계산\n', '기준은 S77의 고정된 분할·라벨·기존 입력이다. 추가 분석은 심사 의견을 받은 뒤 수행한 민감도 분석이다. 주 분석은 시간순 시험의 seen 층이며, 그룹 분할은 보조 결과다. 수치는 비율이며 정확도 차이 0.01은 1 %p다.\n', f"현재 설정을 추가한 R1−R1-이력만의 시간순 정확도 차이는 {interval(history)}다. R1-이력만−무쌍 1차 전이는 {interval(first)}다. 초기 시점 R2−R1은 {interval(initial)}, 깊이6·500회 민감도의 R2−R1은 {interval(deep)}다. 전환 경과 시간을 맞춘 창 내부 동반 초과는 {elapsed.delta:.4f} [{elapsed.ci_low:.4f}, {elapsed.ci_high:.4f}]이며, 같은 창의 원래 LOT 이동 기준 대비 {100 * reductions['elapsed_reduction_fraction']:.1f}% 작다.\n", '## 정의와 재사용 범위\n', '시간순 학습·검증·시험은 각각 1–6월, 7–8월, 9월 이후이며, 그룹 분할은 기존 5개 외부 분할과 내부 검증을 그대로 사용했다. seen은 해당 분할의 최종 학습 자료에 지종 쌍이 있는 층이다. R0′·R1·R2의 기본 결과는 저장된 시험 예측을 재집계했다. R1-이력만은 레시피 입력을 공통으로 유지하고 완료 묶음의 직전 종류·방향·크기·경과와 종류별 횟수만 더했다. 현재 목표값·현재 모드·목표 변경 경과는 제외했다. 이 차이는 이력만 대비 현재 설정 관련 입력의 조건부 예측 증분이다.\n', '빈도 기준의 확률은 최종 학습 자료에서 종류별 10⁻⁶ 가산 평활로 계산했다. 무쌍 전이는 직전 종류→전체 순서로, 쌍 전이는 (쌍, 직전 종류)→쌍→직전 종류→전체 순서로 미관측 조합을 대체했다. 초기 행의 직전 종류는 시작 상태로 처리했다. 최빈 기준의 top-1은 학습 최빈 종류이며, 로그 손실에는 학습 종류별 빈도 확률을 썼다.\n', 'R1-이력만은 기존 깊이 {2,3}·60회 격자에서 검증 정확도, 동률이면 로그 손실로 선택한 뒤 학습+검증 자료로 재적합했다. 용량 민감도는 R1과 R2 각각 깊이6·500회 하나를 고정했다. 같은 검증 절차를 거치되 용량 후보가 하나라 별도 선택은 없고, 시험 결과로 기본 설정과 깊은 설정 중 하나를 고르지 않았다. 학습률0.08, 최소 잎 표본20, 조기 종료 없음, 시드7702를 유지했다.\n', '결정 가중은 모든 결정 행에 같은 가중치를 준다. 에피소드 가중은 각 행에 해당 에피소드 결정 수의 역수를 부여한다. 균형 정확도는 참값이 존재하는 종류별 재현율의 평균이며, macro-F1은 고정된 9종류를 평균한다(분모0인 종류는0). 에피소드 가중의 두 지표도 가중 혼동행렬에서 계산했다. 로그 손실은 자연로그다. 모형 차이는 모두 앞 모형−뒤 모형이므로 로그 손실 차이는 음수가 개선이다. 같은 에피소드를 두 모형에서 함께 뽑는 2,000회 백분위 부트스트랩으로 95% 구간을 계산했다.\n', '## W1. 1차 전이와 이력만 기준\n']
    for split in ['time', 'group']:
        p = metrics[(metrics.split == split) & (metrics.stratum == 'seen')].iloc[0]
        lines.append(f'{SPLIT[split]} seen: 결정 {p.decisions:,}개, 에피소드 {p.episodes:,}개.\n')
        lines.append(table(metric_table(metrics, split, BASE), f'report_w1_{split}.csv'))
    lines.append('주요 짝 비교의 점추정 [95% 구간]은 다음과 같다. 나머지 대비와 all·seen2·unseen 층은 `q2_paired_comparisons.csv`에 있다.\n')
    for split in ['time', 'group']:
        lines.append(table(contrast_table(comparisons, ['R1-R1_history', 'R1_history-markov', 'R1_history-markov_pair', 'R1_history-R0p', 'R1-markov'], [split]), f'report_w1_contrasts_{split}.csv'))
    lines.append(f"현재 설정 관련 입력의 증분은 시간순·그룹 분할 모두 정확도와 로그 손실을 개선한다. 시간순 정확도 증분은 기존 δ=0.05의 {reductions['history_increment_over_q2_delta']:.3f}배다. 완료 이력을 추가한 모형도 무쌍 1차 전이보다 높으므로 기존 R1 이득을 직전 종류 하나의 전이로 환원하기 어렵다. 다만 무쌍 전이 대비 시간순 에피소드 가중 정확도 차이는 +0.0172 [−0.0008, +0.0345]로 구간이 0을 포함한다. 쌍 조건 빈도표는 무쌍 기준보다 정확도가 낮고 로그 손실이 크다. 이 비교에는 고정된10⁻⁶ 평활을 썼으며 시험 성능에 따라 바꾸지 않았다.\n")
    lines += ['## W2와 §1.5. 초기·후속 시점과 용량 민감도\n', '초기는 에피소드 내 완료 묶음 이력이 빈 첫 행이며, 후속은 한 묶음 이상 완료된 행이다. 다음 묶음 종류가 관측된 행만 분류 평가에 포함했다. 표는 정확도와 로그 손실의 차이를 함께 제시한다.\n']
    phase_rows = []
    p = phase[phase.stratum.eq('seen')]
    for (split, timing, weight, contrast), g in p.groupby(['split', 'phase', 'weighting', 'contrast']):
        a, b = (g[g.metric.eq('accuracy')].iloc[0], g[g.metric.eq('log_loss')].iloc[0])
        phase_rows.append({'분할': SPLIT[split], '시점': '초기' if timing == 'initial' else '후속', '가중': WEIGHT[weight], '차이': contrast, '결정/에피소드': f'{a.decisions}/{a.episodes}', '정확도 차이 [95% 구간]': interval(a), '로그 손실 차이 [95% 구간]': interval(b)})
    lines.append(table(pd.DataFrame(phase_rows), 'report_initial_subsequent.csv'))
    for split in ['time', 'group']:
        lines.append(f'{SPLIT[split]} seen의 용량 민감도다.\n')
        lines.append(table(metric_table(metrics, split, ['R1', 'R2', 'R1_deep', 'R2_deep']), f'report_capacity_{split}.csv'))
    lines.append(table(contrast_table(comparisons, ['R2-R1', 'R2_deep-R1_deep', 'R1_deep-R1', 'R2_deep-R2']), 'report_capacity_contrasts.csv'))
    lines.append('기존 Q2의 실용 기준은 정확도 증분 δ=0.05다. 기본 설정의 시간순 R2−R1 정확도는 기준 미달이지만 로그 손실은0.0163 낮아져 작은 확률 예측 개선이 있다. 초기 R1−R0′ 정확도 구간은0을 포함하고, 후속 구간은0보다 높다. 초기 R1−R0′도 로그 손실에서는 개선된다.\n')
    lines.append(f"깊은 설정의 시간순 R2−R1 정확도 증분은 δ의 {reductions['deep_increment_over_q2_delta']:.3f}배이고 구간 상한이0.05를 넘으므로 같은 수치 기준으로는 보류 영역이다. 그룹 분할에서도 증분은 양수다. 결정 가중에서는 깊은 R1과 R2의 정확도가 각각 기본 설정보다 낮고, 두 모형의 로그 손실도 커졌다. 시간순 에피소드 가중에서 깊은 R2−기본 R2는 +0.0056 [−0.0184, +0.0289]다. 깊은 설정 안에서도 R2의 로그 손실 점추정은 R1보다 나쁘다. 따라서 이 결과는 용량에 따른 조건부 증분의 민감성을 보여 준다. 기본 설정의 판정을 바꾸거나 깊은 모형이 전반적으로 개선됐다고 서술할 근거로 쓰지 않는다.\n")
    lines += ['## W4. 전환 경과 시간을 보존한 동반 기준\n', '기존 주 분석과 같은 복귀 관측·LOT 식별·LOT 비횡단 전환 수동 창 1,520개를 사용했다. 창 시작을 키 변경 기준 [−30,+60]분 안의 가장 가까운 전환에 배정하고 동률이면 이른 전환을 골랐다. 1,022개는 키 변경 전이었다. 기존 모집단을 유지하도록 음의 경과도 보존했다.\n', '경과 대응은 다른 전환 에피소드의 같은 경과±5분에서 같은 길이의 창을 골랐다. 대조 창의 시작은 전환 구간이며 창 전체가 하나의 LOT 안에 있어야 한다. 적격 전환을 균등 추출한 뒤 적격 분을 균등 추출하는 과정을 창마다 200회 반복했다(시드8901). 대조 창의 모드는 제한하지 않았다. 전체 길이에서는 대응 가능한 창이 1,518개였고 2개는 대조 창이 없어 제외됐다. 30분 상한에서는 1,520개 모두 대응됐다. 적격 대조 에피소드 수의 중앙값은1,096개다.\n', '두 번째 기준은 연속된 전환 구간 1,060개 안에서만 목표 변경 신호를 원형 이동한다(200회, 시드8902). 6개 목표 채널의 공동 이동과 동등하게 채널 합집합을 이동했고 구간 밖 값은 고정했다. 이 기준은 전환 구간 소속을 보존하며 경과 분까지 맞추는 첫 기준과는 다르다. 앞5분은 [시작−5, 시작), 창 내부는 [시작, 종료)다. Δ=관측 동반율−기준 동반율이며, 2,000회 LOT 부트스트랩을 적용했다.\n']
    p = controls[(controls.loop == 'pooled') & (controls.elapsed_phase == 'all')]
    rows = []
    for r in p.itertuples():
        rows.append({'기준': '경과 대응' if r.method == 'other_episode_elapsed' else '전환 안 이동', '길이': r.cap, '위치': '앞5분' if r.side == 'before' else '내부', '창 수': r.matched_windows, '관측': r.p_observed, '기준율': r.p_reference, 'Δ': r.delta, '95% 구간': f'[{r.ci_low:.4f}, {r.ci_high:.4f}]', '같은 창 기존 Δ': r.original_lot_delta_on_same_windows})
    lines.append(table(pd.DataFrame(rows), 'report_elapsed_controls.csv'))
    p = controls[(controls.loop == 'pooled') & (controls.cap == 'full') & (controls.elapsed_phase != 'all') & (controls.method == 'other_episode_elapsed')]
    rows = [{'시점': '키 변경 전' if r.elapsed_phase == 'before_key' else '키 변경 이후', '위치': '앞5분' if r.side == 'before' else '내부', '창 수': r.matched_windows, 'Δ': r.delta, '95% 구간': f'[{r.ci_low:.4f}, {r.ci_high:.4f}]'} for r in p.itertuples()]
    lines.append(table(pd.DataFrame(rows), 'report_elapsed_phases.csv'))
    lines.append(f"전체 길이의 경과 대응 Δ는 기존 Q1 실용 기준 δ=0.10의 {reductions['elapsed_over_q1_delta']:.3f}배다. 점추정은 기준을 근소하게 넘고 구간은0보다 높으며 30분 상한에서도 같은 방향이다. 키 변경 이후만 보면 내부 Δ는0.0953이므로 전체 전환 창과 같은 모집단으로 서술하면 안 된다. 공통 전환 시각을 맞춘 뒤에도 양의 동반은 남지만 원래0.3014의 크기를 그대로 유지하지 않는다.\n")
    lines += ['## W5. 최근 목표 변경 규칙\n', '기존 S70의 세 목표값 변경 경과 입력만 사용했다. 규칙은 결정 분 t에서 경과가0 이상 k 미만인 채널이 하나라도 있으면 활성화하며, k=3·5는 현재 분을 포함한 관측 분 [t−k+1,t]를 뜻한다. 이는 S70의 원시 목표값 후방 차분 기준이며 Q1의6채널 후보 사건 정의와 다르다. 미래5분의 수동 전환을 대상으로 같은 자동 모드 적격 분·상승 에지·같은 자동 구간 내5분 억제를 적용했다. 경보 임계값을 시험 결과로 조정하지 않았다. 분할·제외 조건·시험 분은 저장된 S70 k=5 주 분석과 같다.\n', 'R은 사건 재현율, P는 경보 분 정밀도, C는 기존 metrics 함수의50회 무작위 사건 재배치 재현율이다. E=(R−C)/(1−C)다. 아래 기존 나무는 S70 저장 집계이며 새 규칙과 같은 평가 모집단이다.\n']
    original = pd.read_csv(ROOT / 'artifacts/intervention/full/metrics.csv')
    original = original[(original.k == 5) & (original.variant == 'main') & (original.direction == 'aut→man') & original.model.eq('tree')].copy()
    original['split'] = original.split.replace({'lot': 'group'})
    combined = pd.concat([rules, original], ignore_index=True)
    cols = ['loop', 'split', 'model', 'region', 'n_minutes', 'positive_minutes', 'alarms', 'true_alarm_minutes', 'events', 'event_hits', 'candidate_events', 'event_recall', 'minute_precision', 'chance_recall', 'chance_excess', 'pr_auc', 'lead_p10', 'lead_p50', 'lead_p90']
    combined[cols].to_csv(DEST / 'recent_target_vs_tree.csv', index=False)

    def alarm_rows(p):
        output = []
        for r in p.sort_values(['split', 'region', 'loop', 'model']).itertuples():
            output.append({'분할': SPLIT[r.split], '구간': '정상' if r.region == 'normal' else '전환', **({'루프': r.loop} if r.loop != 'pooled' else {}), '모형': MODEL[r.model], '적중/사건': f'{int(r.event_hits)}/{int(r.events)}', '경보 수': int(r.alarms), 'R': r.event_recall, 'P': r.minute_precision, 'C': r.chance_recall, 'E': r.chance_excess})
        return pd.DataFrame(output)
    lines.append(table(alarm_rows(combined[combined.loop.eq('pooled') & combined.region.isin(['normal', 'transition'])]), 'report_recent_target_pooled.csv'))
    lines.append('시간순 시험의 루프별 결과다. 그룹 루프별·other·all 구간과 AP·선행 시간은 `recent_target_vs_tree.csv`에 함께 저장했다.\n')
    lines.append(table(alarm_rows(combined[combined.loop.ne('pooled') & combined.split.eq('time') & combined.region.isin(['normal', 'transition'])]), 'report_recent_target_loops.csv'))
    lines.append('최근 목표 변경 규칙은 전환 사건을 더 많이 적중하지만 경보도 크게 늘어 정밀도가 낮다. 따라서 재현율만으로 기존 나무와 동등하다고 할 수 없다. 두 규칙 모두 정상 생산과 전환 구간에서 기존 정밀도 기준0.30에 못 미친다.\n')
    lines += ['## §1.5. 키 변경 전 묶음과 자동 창 대조\n', '키 변경 전 묶음은 모든 후보 종류를 합쳐 같은 시작 분을 하나로 묶고 [키 변경−30분, 키 변경)에서 셌다. 키 변경 분은 제외한다. 다음 표는 다음 행동의 관측 여부로 제한하기 전 전환 에피소드 분포이며, 시간순 seen177개 중 분류의 초기 행이 관측된 것은168개다.\n']
    lines.append(table(pre.rename(columns={'stratum': '층', 'episodes': '에피소드', 'with_pre30_bundle': '앞 묶음 있음', 'fraction_with_pre30': '비율', 'mean': '평균', 'median': '중앙값'}), 'report_pre30_bundles.csv'))
    lines.append('전체 전환의95.64%에 키 변경 전 묶음이 있었다. 따라서 이력이 빈 초기 행도 실제 운전의 첫 행동을 뜻하지 않는다. 현재 목표값과 모드는 키 변경 전 조정을 반영할 수 있어, 초기 R2−R1의 부호만으로 상태→설정·행동의 매개 가설을 판정할 수 없다.\n')
    rows = [{'구간': '정상' if r.region == '정상 생산' else '전환', '길이': r.cap, '위치': '앞5분' if r.side == 'before' else '내부', '대응 창': r.windows, '관측': r.observed, '자동 기준': r.automatic_reference, 'Δ': r.delta, '95% 구간': f'[{r.ci_low:.4f}, {r.ci_high:.4f}]'} for r in auto.itertuples()]
    lines.append(table(pd.DataFrame(rows), 'report_automatic_control.csv'))
    lines.append('정상 생산 앞5분의 길이·루프·구간·LOT 대응 자동 대조 Δ=0.1937은 대응 가능한282개 창의 값이다. 원래 LOT 이동 대조의 앞5분 Δ=0.0993과는 기준과 모집단이 다르다.\n')
    lines += ['## 계산 확인과 해석 한계\n', f"새 계산의 공백은 빈도 대체·가중 집계·경과 대응·경보 시점이었다. 작은 정답 시험5개로 학습 밖 라벨이 빈도표에 들어가는 오류, 에피소드 가중 오류, 전환 기준 시점 오류, 구간 밖 이동, 미래 목표값 사용이 발생하지 않는지 점검했다. `s89_tests.py`의 5개 시험과 ruff 검사를 통과했다. 기존 R0′·R1·R2의 {verification[0]['metric_rows']}개 집계 행에서 정확도·로그 손실을 재현했고 기존 정확도 차이 구간도 일치했다. S70의 {control_manifest['recent_targets']['verified_saved_tree_rows']}개 루프·분할·구간에서 저장 나무의 사건 수·적중 수를 재현했다. 새 적합6개 분할 모두 원래 시험 행·라벨·학습 행 수가 일치했다.\n", '기존 Q1은 `scripts/s77/s77_q1.py:39–51`에서200회 이동 평균을 창별로 먼저 정한 뒤, 같은 파일118–119행과 `s77_common.py:104–115`에서 고정된 관측−기준 차이를 LOT 단위로 재표본한다. 부트스트랩마다 이동 기준을 다시 만들지 않는다. 이번 두 추가 기준도 같은 방식이다. 따라서 구간은 이미 추정한 이동·대응 기준에 조건부이며,200회 몬테카를로 오차나 공통 대조 에피소드의 불확실성을 완전히 전파하지 않는다.\n', '이 분석은 관측 로그의 조건부 예측과 동반을 다룬다. 모형 입력을 순차 추가한 차이는 정보 출처의 인과적 분해가 아니며, 초기 행은 키 변경 전 이력을 잘라낸 인덱스다. 깊이6·500회는 고정된 한 가지 용량 민감도이고, 경과 대응은 전환 공통 시각을 맞추지만 지종 쌍·운전 맥락을 모두 통제하지 않는다. 추가 비교의 구간은 다중 비교 보정 없는 사후 구간이다.\n', '대상 버전: S77 고정 산출물과 S70 k=5 주 분석이다. 이번 변경은 브리프의 추가 기준선·민감도·재집계뿐이다. 기존 분할·학습 스케일·예측·정보 시점 시험은 재사용했다. 코드와 결과는 각각 `scripts/s77/s89_*.py`, `artifacts/stage1_audit/s89/`에 저장했다. 원고와 기존 코드·산출물은 수정하지 않았다.\n']
    REPORT.write_text('\n'.join(lines), encoding='utf-8')
    finished = {'status': 'complete', 'report': str(REPORT), 'folds': ['time', 'group0', 'group1', 'group2', 'group3', 'group4'], 'tables': len(list(DEST.glob('report_*.csv'))), 'existing_metric_rows_reproduced': verification[0]['metric_rows'], 'saved_tree_strata_reproduced': control_manifest['recent_targets']['verified_saved_tree_rows']}
    (DEST / 'finish_status.json').write_text(json.dumps(finished, ensure_ascii=False, indent=2))
    print(json.dumps(finished, ensure_ascii=False))
if __name__ == '__main__':
    main()
