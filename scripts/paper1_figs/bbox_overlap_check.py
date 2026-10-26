"""Check exported text against the canvas; does not certify pairwise overlap."""
from matplotlib.text import Text


def audit_fig(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    findings = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text():
            continue
        if text.axes is not None and not text.axes.get_visible():
            continue
        box = text.get_window_extent(renderer)
        if box.x0 < canvas.x0-1 or box.y0 < canvas.y0-1 or box.x1 > canvas.x1+1 or box.y1 > canvas.y1+1:
            findings.append('Text outside canvas: ' + text.get_text())
    return findings
