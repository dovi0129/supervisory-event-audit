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
CPU = '0-15,32-47'
PYTHON = __import__('sys').executable

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['smoke', 'full'], required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error('workers must be in [1,16]')
    subprocess.run(['taskset', '-a', '-cp', CPU, str(os.getpid())], stdout=subprocess.DEVNULL, check=True)
    os.chdir(ROOT)
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[key] = '1'
    out = ROOT / 'artifacts/intervention'
    base = out / args.mode
    base.mkdir(parents=True, exist_ok=True)
    lock = base / 'RUNNING_s70'
    with lock.open('x') as fp:
        fp.write(str(os.getpid()))
    start = time.monotonic()
    status_path = base / 'finish_status.json'
    status_path.write_text(json.dumps({'status': 'RUNNING', 'pid': os.getpid()}))
    done = out / ('DONE_s70' if args.mode == 'full' else 'DONE_s70_smoke')
    fail = out / ('FAIL_s70' if args.mode == 'full' else 'FAIL_s70_smoke')
    if done.exists() or (fail.exists() and (not args.resume)):
        lock.unlink()
        raise FileExistsError('Preserve completed/failed run; archive its directory before restarting')
    spec = ROOT / 'generated/protocol/S70_spec_v0.1_intervention.md'

    def digest(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    provenance = {'mode': args.mode, 'workers': args.workers, 'cpu_affinity': CPU, 'spec_sha256': digest(spec), 'source_sha256': {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')}, 'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'python': PYTHON, 'input_source': 'existing private S68 cache; no source data copied to reports'}
    if args.resume:
        previous = base / 'provenance.json'
        if previous.exists():
            prior = json.loads(previous.read_text())
            if prior['spec_sha256'] != provenance['spec_sha256']:
                raise ValueError('Resume requires unchanged specification')
            provenance['previous_attempt'] = prior
        if fail.exists():
            fail.rename(base / ('FAILED_attempt_' + str(time.time_ns()) + '.json'))
    (base / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    logs = ROOT / 'artifacts/logs'

    def command(script, *arguments, log):
        with (logs / log).open('w') as stream:
            subprocess.run([PYTHON, str(ROOT / 'scripts/s70' / script), *arguments], stdout=stream, stderr=subprocess.STDOUT, check=True)
    try:
        data = out / ('data_smoke' if args.mode == 'smoke' else 'data')
        feature_args = ['--month', '2022-01'] if args.mode == 'smoke' else []
        command('s70_features.py', *feature_args, log=f's70_{args.mode}_features.log')
        from s70_fit import jobs
        keys = jobs(data)

        def run_job(key):
            path = base / 'jobs' / (key.replace(':', '_') + '.json')
            if path.exists() and (not args.resume):
                raise FileExistsError('Unregistered existing job; archive before a fresh launch')
            if not path.exists():
                log_name = f"s70_{args.mode}_{key.replace(':', '_')}.log"
                previous_log = logs / log_name
                if previous_log.exists():
                    previous_log.rename(previous_log.with_suffix('.attempt_' + str(time.time_ns()) + '.log'))
                command('s70_fit.py', '--data', str(data), '--only', key, log=log_name)
            payload = json.loads(path.read_text())
            if len(payload['folds']) != 72:
                raise ValueError('Expected 6 folds × 3 models × 4 strata')
            return payload['seconds']
        durations = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_job, key): key for key in keys}
            for future in as_completed(futures):
                durations.append(future.result())
                print(json.dumps({'finished': len(durations), 'total': len(keys), 'key': futures[future]}), flush=True)
        if digest(spec) != provenance['spec_sha256']:
            raise RuntimeError('Specification changed while run was active; main must reconcile before aggregation')
        command('s70_report.py', '--mode', args.mode, log=f's70_{args.mode}_report.log')
        summary = json.loads((base / 'summary.json').read_text())
        if summary['completed_jobs'] != len(keys) or summary['prediction_rows'] <= 0:
            raise ValueError('Incomplete output manifest')
        provenance['elapsed_seconds'] = time.monotonic() - start
        provenance['job_seconds'] = durations
        provenance['completed_jobs'] = len(keys)
        (base / 'provenance.json').write_text(json.dumps(provenance, indent=2))
        done.write_text(json.dumps({'completed_jobs': len(keys), 'summary': str(base / 'summary.json')}))
        status_path.write_text(json.dumps({'status': 'DONE', 'completed_jobs': len(keys), 'elapsed_seconds': provenance['elapsed_seconds']}))
        print(json.dumps({'complete': args.mode, 'seconds': provenance['elapsed_seconds']}), flush=True)
    except BaseException as exc:
        fail.write_text(json.dumps({'error_type': type(exc).__name__, 'message': str(exc)}, ensure_ascii=False))
        status_path.write_text(json.dumps({'status': 'FAILED', 'error_type': type(exc).__name__}))
        raise
    finally:
        lock.unlink()
if __name__ == '__main__':
    main()
