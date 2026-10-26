import sys as _runtime_sys
from pathlib import Path as _RuntimePath
_runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[2]))
from audit_runtime import PACKAGE_ROOT, WORKSPACE, record_path, require_records
if __name__ == "__main__":
    require_records()
import json
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from s77_common import OUT, lot_bootstrap
from s77_q2_fit import audit_effective_exchange
from s77_report import run

def matched_denominators(dest):
    windows = pd.read_parquet(dest / str(record_path('q1_windows_private_parquet')))
    summary = pd.read_csv(dest / 'q1_summary.csv')
    rows = []
    for r in summary.itertuples():
        w = windows
        if r.population == 'main_no_cross':
            w = w[~w.cross_lot & w.known_lot]
        if r.population == 'no_quick':
            w = w[~w.cross_lot & w.known_lot & ~w.quick]
        if r.loop != 'pooled':
            w = w[w.loop.eq(r.loop)]
        w = w[w.region.eq(r.region)]
        column = 'auto_chance_' + r.cap if r.side == 'inside' else 'auto_before_' + r.cap
        w = w[w[column].notna()]
        obs = w['inside_' + r.cap].to_numpy(float) if r.side == 'inside' else w.before.to_numpy(float)
        chance = w[column].to_numpy()
        diff = obs - chance
        ci = lot_bootstrap(diff, w.lot)
        rows.append({'p_observed_matched': float(obs.mean()) if len(w) else None, 'matched_control_delta': float(diff.mean()) if len(w) else None, 'matched_ci_low': ci[0], 'matched_ci_high': ci[1]})
    extra = pd.DataFrame(rows)
    summary = summary.drop(columns=[c for c in extra if c in summary])
    pd.concat([summary.reset_index(drop=True), extra], axis=1).to_csv(dest / 'q1_summary.csv', index=False)
if __name__ == '__main__':
    matched_denominators(OUT)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for fold in pool.map(lambda fold: audit_effective_exchange(OUT, fold), ['time', 'group0', 'group1', 'group2', 'group3', 'group4']):
            print(json.dumps({'audit_fold': fold}), flush=True)
    run(OUT)
