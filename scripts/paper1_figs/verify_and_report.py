"""Verify aggregate-derived values and exported PDF geometry/metadata."""
import json
from pathlib import Path
import fitz

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    numbers = json.loads((HERE / 'numeric_checks.json').read_text())
    assert len(numbers) == 18
    for row in numbers:
        expected = row['hits']/row['n'] if row['figure'] == 3 else row['csv']
        assert abs(expected-row['independent']) < 1e-12
    layout = json.loads((HERE / 'layout_checks.json').read_text())
    assert all(not findings for findings in layout.values()), layout
    checks = []
    for path in sorted((ROOT / 'figures').glob('*.pdf')):
        with fitz.open(path) as document:
            assert len(document) == 1
            page = document[0]
            width = page.rect.width/72
            assert abs(width-12.2/2.54) < .001
            spans = [span for block in page.get_text('dict')['blocks'] if 'lines' in block
                     for line in block['lines'] for span in line['spans'] if span['text'].strip()]
            assert spans and min(span['size'] for span in spans) >= 7
            assert not document.metadata.get('author')
            assert page.get_text().strip()
            checks.append({'file': path.name, 'pages': len(document), 'width_inches': width,
                           'minimum_font_points': min(span['size'] for span in spans),
                           'metadata': document.metadata, 'text_available': True})
    assert len(checks) == 3
    (ROOT / 'validation/figure_checks.json').write_text(json.dumps(checks, indent=2)+'\n')
    print('Verified 18 aggregate values and 3 single-page PDFs; author fields are empty.')


if __name__ == '__main__':
    main()
