"""Explicit, external inputs for the optional plant-record pipeline."""
from functools import lru_cache
import importlib
import json
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get('AUDIT_WORKSPACE', PACKAGE_ROOT)).resolve()


@lru_cache(maxsize=1)
def configuration():
    path = os.environ.get('PLANT_RECORDS_CONFIG')
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding='utf-8'))


def record_path(role):
    config = configuration()
    supplied = config.get('paths', {}).get(role)
    if supplied is None:
        return WORKSPACE / 'unavailable_records' / role
    return Path(supplied).expanduser().resolve()


def require_records():
    if not os.environ.get('PLANT_RECORDS_CONFIG'):
        print('SKIP: requires plant records; public aggregates and synthetic tests remain available.')
        raise SystemExit(0)
    if WORKSPACE == PACKAGE_ROOT:
        raise RuntimeError('Use a separate AUDIT_WORKSPACE for plant-record processing.')
    configuration()


class PlantAdapter:
    """The data owner supplies masking and role-loading implementations privately."""
    def __getattr__(self, name):
        provider = configuration().get('adapter_module')
        if not provider:
            raise RuntimeError('requires plant records and an external adapter_module')
        return getattr(importlib.import_module(provider), name)


plant_events = PlantAdapter()


def load_roles(*args, **kwargs):
    return plant_events.load_roles(*args, **kwargs)


def fill_and_mask(*args, **kwargs):
    return plant_events.fill_and_mask(*args, **kwargs)
