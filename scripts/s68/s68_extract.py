"""Owner-provided ingestion adapter; the public package contains no plant schema."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from audit_runtime import require_records, plant_events  # noqa: E402


def main():
    require_records()
    # The owner's adapter must rename source columns to the public role keys.
    # File paths, source tags, masking rules and source records stay external.
    plant_events.extract_role_records()


if __name__ == '__main__':
    main()
