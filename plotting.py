"""Match the paper's font sizes after figures are scaled to the text width."""
import re
from matplotlib.text import Text


def save(fig, path, kind, pad=.06, **kwargs):
    if not hasattr(fig, '_print_bbox'):
        fig.canvas.draw()
        ticks = {id(t) for ax in fig.axes for axis in (ax.xaxis, ax.yaxis)
                 for tick in axis.get_major_ticks()+axis.get_minor_ticks()
                 for t in (tick.label1, tick.label2)}
        labels = {id(t) for ax in fig.axes for t in (ax.xaxis.label, ax.yaxis.label)}
        titles = {id(ax.title) for ax in fig.axes}
        targets = []
        for text in fig.findobj(Text):
            s = text.get_text()
            if not s or not text.get_visible():
                continue
            if text.get_path_effects():
                text.set_path_effects([])
                if text.get_bbox_patch() is None:
                    text.set_bbox(dict(facecolor='white', edgecolor='none', alpha=.85, pad=.15))
            if re.fullmatch(r'\([a-f]\)', s) or id(text) in titles or text is fig._suptitle:
                size = 9.
            elif id(text) in ticks:
                size = 7.
            elif id(text) in labels or (kind == 'regret' and 'gate-open fraction' in s):
                size = 8.
            elif kind == 'regret' and not any(x in s for x in ('Rapp', 'Gupta', 'Example for', 'Free-running optimal')):
                size = 8. if text.get_fontsize() >= 10.7 else 7.
            else:
                size = 7.
            targets.append((text, size))
        printed_width = 6.151*(.97 if kind == 'regret' else 1.)
        width = fig.get_tightbbox(fig.canvas.get_renderer()).width+2*pad
        for _ in range(18):
            for text, size in targets:
                text.set_fontsize(size*width/printed_width*72.27/72)
            fig.canvas.draw()
            bbox = fig.get_tightbbox(fig.canvas.get_renderer()).padded(pad)
            if abs(bbox.width-width) < 1e-5:
                break
            width = bbox.width
        else:
            raise RuntimeError('Font scaling did not converge')
        fig._print_bbox = bbox
    fig.savefig(path, bbox_inches=fig._print_bbox, pad_inches=0, **kwargs)
