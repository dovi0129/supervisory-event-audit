"""Run public tests/plots while rejecting private records and external project inputs."""
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.dontwritebytecode = True
os.environ.pop('PLANT_RECORDS_CONFIG', None)
os.environ.pop('AUDIT_WORKSPACE', None)
os.environ['MPLCONFIGDIR'] = str(ROOT / '.mplconfig')
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'
opened = set()


def audit(event, args):
    if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    if path.suffix.lower() in ['.parquet', '.joblib', '.xlsx', '.xls']:
        raise RuntimeError('Public reproduction attempted to open a record/model file')
    if any(part in {'.agents', '.claude', '.codex', 'RawData', 'private'} for part in path.parts):
        raise RuntimeError('Public reproduction attempted to access an external/private helper')
    if 'artifacts' in path.parts and not path.is_relative_to(ROOT):
        raise RuntimeError('Public reproduction attempted to access external artifacts')
    if path.is_relative_to(ROOT):
        opened.add(str(path.relative_to(ROOT)))


def main():
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    sys.addaudithook(audit)
    import pytest
    code = pytest.main(['tests', '-q'])
    if code:
        raise SystemExit(code)
    sys.path.insert(0, str(ROOT / 'scripts/paper1_figs'))
    # Clear the unrelated historical helper if a test imported it.
    sys.modules.pop('paper_figstyle', None)
    runpy.run_path(str(ROOT / 'scripts/paper1_figs/build_figures.py'), run_name='__main__')
    records = {'tests_exit_code': int(code), 'record_files_opened': 0,
               'external_artifacts_opened': 0,
               'package_inputs': sorted(x for x in opened if x.endswith(('.csv', '.json')) and not x.startswith('.mplconfig/')),
               'pdfs': sorted(p.name for p in (ROOT / 'figures').glob('*.pdf'))}
    (ROOT / 'validation/reproduction.json').write_text(json.dumps(records, indent=2)+'\n')
    print('Public reproduction completed using packaged aggregates and synthetic data only.')


if __name__ == '__main__':
    main()
