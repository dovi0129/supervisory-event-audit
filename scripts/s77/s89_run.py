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
from concurrent.futures import ProcessPoolExecutor, as_completed
from s89_q2 import DEST, fit_fold

def main():
    start = time.monotonic()
    allowed = set(range(16)) | set(range(32, 48))
    assert set(os.sched_getaffinity(0)) <= allowed
    folds = [f'group{i}' for i in range(5)]
    with ProcessPoolExecutor(max_workers=5) as pool:
        pending = {pool.submit(fit_fold, fold): fold for fold in folds}
        for future in as_completed(pending):
            result = future.result()
            print(json.dumps({'completed': result['fold']}), flush=True)
    result = {'folds': folds, 'seconds': time.monotonic() - start, 'status': 'group_folds_complete'}
    (DEST / 'group_finish_status.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
if __name__ == '__main__':
    main()
