from __future__ import annotations
import sys as _runtime_sys
from pathlib import Path as _RuntimePath
_runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[2]))
from audit_runtime import PACKAGE_ROOT, WORKSPACE, record_path, require_records
if __name__ == "__main__":
    require_records()
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
ROOT = WORKSPACE
PYTHON = __import__('sys').executable
FOLDS = ['time', 'group0', 'group1', 'group2', 'group3', 'group4']

def run_mode(mode, scales, workers):
    out = ROOT / 'artifacts/stage1_audit'
    dest = out / 'smoke' if mode == 'smoke' else out
    dest.mkdir(parents=True, exist_ok=True)
    spec = ROOT / 'generated/protocol/S77_spec_v0.1_recipe_audit.md'

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {'spec_sha256': sha(spec), 'source_sha256': {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}, 'scale_mode': scales, 'workers': workers, 'mode': mode, 'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'cpu_affinity': sorted(os.sched_getaffinity(0)), 'gate_reused': 'generated/notes/S77_skeptic_result.md; accepted seven corrections in v0.2', 'reused': []}
    start = time.monotonic()

    def command(script, *args, log):
        with (ROOT / 'artifacts/logs' / log).open('w') as stream:
            subprocess.run([PYTHON, str(ROOT / 'scripts/s77' / script), *args], stdout=stream, stderr=subprocess.STDOUT, check=True)
    smoke_args = ['--smoke'] if mode == 'smoke' else []
    for script, marker in [('s77_q1.py', 'q1_manifest.json'), ('s77_q3.py', 'q3_detector.json'), ('s77_q2_build.py', 'q2_build_manifest.json')]:
        if (dest / marker).exists():
            manifest['reused'].append(marker)
        else:
            command(script, *smoke_args, log=f's77_{mode}_{script[:-3]}.log')
    (dest / 'run_provenance.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    def fold_job(fold):
        marker = dest / 'folds' / f'{fold}.json'
        if marker.exists():
            payload = json.loads(marker.read_text())
            if payload['scale_mode'] != scales:
                raise ValueError('Existing fold has another scale definition')
            return (fold, 'reused')
        command('s77_q2_fit.py', *smoke_args, '--only', fold, '--scales', scales, log=f's77_{mode}_{fold}.log')
        if not marker.exists():
            raise ValueError('Missing fold output manifest')
        return (fold, 'executed')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fold_job, fold) for fold in FOLDS]
        for future in as_completed(futures):
            fold, status = future.result()
            print(json.dumps({'mode': mode, 'fold': fold, 'status': status}), flush=True)
    if sha(spec) != manifest['spec_sha256']:
        raise ValueError('Specification changed during the run')
    command('s77_report.py', *smoke_args, log=f's77_{mode}_report.log')
    summary = json.loads((dest / 'summary.json').read_text())
    if summary['folds'] != 6 or summary['observed_prediction_rows'] <= 0:
        raise ValueError('Incomplete aggregate outputs')
    import fitz
    import pandas as pd
    checks = []
    for pdf in dest.glob('*.pdf'):
        doc = fitz.open(pdf)
        page = doc[0]
        spans = [s for b in page.get_text('dict')['blocks'] for line in b.get('lines', []) for s in line['spans'] if s['text'].strip()]
        clipped = [s for s in spans if not page.rect.contains(fitz.Rect(s['bbox']))]
        minimum = min((s['size'] for s in spans)) * 5.5 * 72 / page.rect.width
        if clipped or minimum < 7:
            raise ValueError('PDF clipping/font check: ' + pdf.name)
        checks.append({'pdf': pdf.name, 'minimum_print_font_pt': minimum, 'clipped_spans': 0})
    q1 = json.loads((dest / 'q1_manifest.json').read_text())
    build = json.loads((dest / 'q2_build_manifest.json').read_text())
    if mode == 'full' and (q1['normal_returned'] != 612 or build['episodes'] != 1123 or build['no_action_episodes'] != 25):
        raise ValueError('Registered population reproduction failed')
    mapping = json.loads((ROOT / str(record_path('artifacts_operator_policy_tag_map_private_json'))).read_text())
    report = dest / 'smoke_report.md' if mode == 'smoke' else ROOT / 'generated/notes/S77_audit_W4.md'
    if any((tag in report.read_text() for tag in mapping['loops'])):
        raise ValueError('Source tag exposed in report')
    metrics = pd.read_csv(dest / 'q2_metrics.csv')
    if not metrics.accuracy.between(0, 1).all():
        raise ValueError('Accuracy out of bounds')
    manifest['seconds'] = time.monotonic() - start
    manifest['pdf_checks'] = checks
    manifest['summary'] = summary
    (dest / 'run_provenance.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    (dest / ('DONE_s77_smoke' if mode == 'smoke' else 'DONE_s77')).write_text(json.dumps({'status': 'DONE', **summary}, ensure_ascii=False, indent=2))
    return summary

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scales', choices=['train', 's68'], required=True)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--mode', choices=['smoke', 'full', 'both'], default='both')
    a = p.parse_args()
    if not 1 <= a.workers <= 16:
        p.error('workers must be 1..16')
    subprocess.run(['taskset', '-a', '-cp', '0-15,32-47', str(os.getpid())], stdout=subprocess.DEVNULL, check=True)
    os.chdir(ROOT)
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[key] = '1'
    out = ROOT / 'artifacts/stage1_audit'
    out.mkdir(parents=True, exist_ok=True)
    status = out / 'finish_status.json'
    lock = out / 'RUNNING_s77'
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    status.write_text(json.dumps({'status': 'RUNNING', 'pid': os.getpid(), 'phase': a.mode}))
    try:
        for mode in ['smoke', 'full'] if a.mode == 'both' else [a.mode]:
            status.write_text(json.dumps({'status': 'RUNNING', 'pid': os.getpid(), 'phase': mode}))
            run_mode(mode, a.scales, a.workers)
        status.write_text(json.dumps({'status': 'DONE', 'phase': a.mode}))
    except BaseException as error:
        status.write_text(json.dumps({'status': 'FAILED', 'error_type': type(error).__name__, 'error': str(error)}))
        raise
    finally:
        lock.unlink()
if __name__ == '__main__':
    main()
