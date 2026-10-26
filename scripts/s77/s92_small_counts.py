import sys as _runtime_sys
from pathlib import Path as _RuntimePath
_runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[2]))
from audit_runtime import PACKAGE_ROOT, WORKSPACE, record_path, require_records
if __name__ == "__main__":
    require_records()
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT = WORKSPACE
OUT = ROOT / 'artifacts/stage1_audit/s92'

def main():
    minutes = pd.read_parquet(ROOT / str(record_path('artifacts_intervention_data_minutes_private_parquet')), columns=['lot'])
    episodes = pd.read_parquet(ROOT / 'artifacts/transition/episodes.parquet', columns=['cohort', 'start_position', 'start'])
    episodes = episodes[episodes.cohort.eq('transition')]
    assert len(episodes) == 1123
    pos = episodes.start_position.to_numpy(int)
    assert (pd.to_datetime(minutes.index[pos]).to_numpy() == pd.to_datetime(episodes.start).to_numpy()).all()
    lot = minutes.lot
    known = lot.notna() & lot.astype(str).ne('')
    comparable = known & known.shift(fill_value=False)
    boundary = comparable & lot.ne(lot.shift())
    compared = comparable.iloc[pos].to_numpy()
    matches = boundary.iloc[pos].to_numpy()
    decisions = pd.read_parquet(ROOT / str(record_path('artifacts_stage1_audit_q2_decisions_private_parquet')), columns=['episode_id', 'position', 'censored'])
    predictions = pd.read_parquet(ROOT / str(record_path('artifacts_stage1_audit_folds_time_predictions_private_parquet')))
    q = predictions.loc[~predictions.seen].copy()
    aligned = q.merge(decisions, on=['episode_id', 'position'], how='left', validate='one_to_one', indicator=True)
    assert aligned['_merge'].eq('both').all() and (~aligned.censored).all()
    assert len(q) == 947
    a = q[[f'R1_p{i}' for i in range(9)]].to_numpy().argmax(axis=1)
    b = q[[f'R2_p{i}' for i in range(9)]].to_numpy().argmax(axis=1)
    y = q.kind_code.to_numpy()
    result = {'authorization': '메인 허가로 추가 집계', 'lot_grade_relation': {'definition': 'same-minute grade-key episode start and change between adjacent known lot labels', 'grade_changes': len(pos), 'comparable_grade_changes': int(compared.sum()), 'unknown_adjacent_lot': int((~compared).sum()), 'coincident_lot_changes': int((matches & compared).sum()), 'grade_changes_without_lot_change': int((~matches & compared).sum()), 'all_adjacent_known_lot_changes': int(boundary.sum())}, 'unseen_predictions': {'decisions': len(q), 'episodes': int(q.episode_id.nunique()), 'R1_correct': int((a == y).sum()), 'R2_correct': int((b == y).sum()), 'same_prediction': int((a == b).sum()), 'different_prediction': int((a != b).sum()), 'both_correct': int(((a == y) & (b == y)).sum()), 'R1_only_correct': int(((a == y) & (b != y)).sum()), 'R2_only_correct': int(((b == y) & (a != y)).sum())}}
    assert result['unseen_predictions']['R1_correct'] == result['unseen_predictions']['R2_correct']
    OUT.mkdir(exist_ok=True)
    (OUT / 'small_counts.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
