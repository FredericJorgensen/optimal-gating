"""Plot the six panels of Figure 4."""
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch
from core import C, DT, K, SIGMA
from plotting import save


def label_curve(ax, line, x, position=None, fontsize=8.5):
    xs, ys = line.get_data()
    y = np.exp(np.interp(x, xs, np.log(ys))) if ax.get_yscale() == 'log' else np.interp(x, xs, ys)
    color = line.get_color()
    arrow = None
    if position is not None:
        ax.plot(x, y, 'o', color=color, ms=2.2, mew=0, zorder=9)
        arrow = dict(arrowstyle='-', color=color, lw=.65, shrinkA=1.5, shrinkB=0)
    ax.annotate(line.get_label(), (x, y), xytext=position or (x, y),
                color=color, fontsize=fontsize, ha='center', va='center', zorder=10, linespacing=1.05,
                bbox=dict(facecolor='white', edgecolor='none', alpha=.94, pad=.12),
                arrowprops=arrow)


def plot(data, output):
    mpl.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Computer Modern Roman'], 'mathtext.fontset': 'cm',
        'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}',
        'text.color': '#253039', 'axes.labelcolor': '#253039',
        'axes.edgecolor': '#90989E', 'axes.linewidth': .65,
        'xtick.color': '#58636B', 'ytick.color': '#58636B', 'savefig.transparent': True,
    })
    t, ranges, marks = (data[k] for k in ('t', 'R_curve', 'R_marks'))
    keys = ('lpu', 'hpu')
    colors = ('#185A9D', '#B43C18')
    range_colors = ('#6E8054', '#7C6A91', '#B77935')
    method_colors = (('#123F70', '#185A9D', '#417AAA'), ('#7F2A16', '#B43C18', '#B85A40'))
    methods = ('free', 'opt', 'oracle')
    styles = ((0, (5, 2.5)), '-', (0, (.5, 2.5)))
    labels = ('free-running', 'suffix opt.', 'oracle')
    regimes = ('low pile-up', 'high pile-up')
    cmap = LinearSegmentedColormap.from_list('improvement',
        ['#FFFDF6', '#F4D99E', '#DFA653', '#AF6D30', '#633A20'])
    reduction = np.maximum(data['reduction'], 0)
    norm = Normalize(0, max(np.nanmax(reduction), 1))

    fig = plt.figure(figsize=(10.5, 7.3), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.10, .95], width_ratios=[1, 1.25, 1.10],
                          left=.065, right=.935, bottom=.100, top=.900, wspace=.58, hspace=.66)
    a, b, c, *gamma_axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    pos = c.get_position()
    c.set_position([pos.x0+.012, pos.y0, pos.width, pos.height])
    width = max(ax.get_position().width for ax in gamma_axes)
    for i, ax in enumerate(gamma_axes):
        pos = ax.get_position()
        ax.set_position([pos.x0 if i < 2 else pos.x1-width, pos.y0-.030, width, pos.height])

    lines = [a.plot(t, np.cumsum(data[key+'_q']), color=colors[i], lw=2,
                    label=regimes[i], zorder=3+i)[0] for i, key in enumerate(keys)]
    a.axhline(1, color='#111111', lw=1.35, ls=styles[0], alpha=.9, zorder=2, clip_on=False)
    a.annotate('free-running', (12, 1), xytext=(14, .89), color='#111111',
               fontsize=7.4, ha='center', va='center', zorder=8,
               arrowprops=dict(arrowstyle='-', color='#111111', lw=.65, shrinkA=2, shrinkB=2))
    for line, x, position in zip(lines, [13, 29], [(12, .52), (37, .65)]):
        label_curve(a, line, x, position, fontsize=7.3)
    a.set(xlim=(0, 50), ylim=(0, 1), yticks=np.linspace(0, 1, 6),
          xlabel=r'$t$ (ns)', ylabel=r'probability $H_t = 1$')
    a.spines['top'].set_visible(False)

    b.set(yscale='log', xlim=(ranges[0], ranges[-1]),
          xticks=np.arange(ranges[0], ranges[-1]+.25, .5),
          xlabel=r'$R$ (m)', ylabel=r'$L_R^{\mathrm{period}}$ (m$^2$)')
    positions = [(3, (3, .104)), (2.55, (2.55, .245)), (3.05, (3.05, .049)),
                 (3.75, (3.25, 1.18)), (4.2, (4.5, .94)), (4.5, (4.5, .28))]
    lines = []
    for i, key in enumerate(keys):
        for j, method in enumerate(methods):
            x = data[key+'_oracle_R'] if method == 'oracle' else ranges
            line, = b.plot(x, data[key+'_'+method], color=method_colors[i][j],
                           ls=styles[j], lw=(1.35, 1.8, 1.5)[j], alpha=1 if j != 1 else None,
                           label=labels[j]+',\n'+regimes[i],
                           **({'dash_capstyle': 'round'} if j == 2 else {}))
            lines.append(line)
    for line, position in zip(lines, positions):
        label_curve(b, line, *position)
    for r, color in zip(marks, range_colors):
        b.scatter(r, 0, transform=b.get_xaxis_transform(), marker='v', s=62,
                  color=color, edgecolor='white', linewidth=.45, zorder=9, clip_on=False)
        for tick, label in zip(b.get_xticks(), b.get_xticklabels()):
            if np.isclose(tick, r):
                label.set_color(color)
                label.set_fontweight('normal')

    bs, ds = data['B_grid'], data['dead_grid']
    b_edges = np.r_[bs[0], np.sqrt(bs[:-1]*bs[1:]), bs[-1]]
    d_edges = np.r_[ds[0], (ds[:-1]+ds[1:])/2, ds[-1]]
    mesh = c.pcolormesh(d_edges, b_edges, reduction, cmap=cmap, norm=norm,
                       shading='flat', edgecolors='none', antialiased=False, rasterized=True)
    c.set(yscale='log', xlabel=r'$t_d$ (ns)', ylabel=r'$B$')
    for dead, background, color, regime in zip([10, 40], [1, 10], colors, regimes):
        c.scatter(dead, background, s=48, color=color, edgecolor='white', linewidth=.8, zorder=5)
        c.text(dead, background/1.18, regime+'\nexample', color=color,
               fontsize=7.8, ha='center', va='top', weight='normal', linespacing=.88,
               bbox=dict(facecolor='white', edgecolor='none', alpha=.76, pad=.55))
    cbar = fig.colorbar(mesh, cax=c.inset_axes([.18, -.46, .64, .045]), orientation='horizontal')
    cbar.set_label('suffix worst-case error red.\ncomp. to free-running (\\%)', fontsize=8, labelpad=5)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.ax.tick_params(labelsize=7.2, length=2.2, pad=.8)
    cbar.outline.set_visible(False)

    pulses = np.array([.5*(7.5/r)**2*DT*(sum(
        np.exp(-.5*((t-2*r/C-shift*K*DT)/SIGMA)**2) for shift in range(-2, 3)
        )/(np.sqrt(2*np.pi)*SIGMA))+10/K for r in marks])
    positions = [
        [(6.5, (7.5, .73)), (25, (35.5, .73)), (35, None), (35, (40, .52))],
        [(10, (9, .76)), (17, (12.5, .48)), (35, None), (36, (39, .45))],
        [(10, (9, .76)), (21, (24, .59)), (4, (9, .35)), (34.5, (40, .19))],
    ]
    for i, (ax, r, color) in enumerate(zip(gamma_axes, marks, range_colors)):
        ribbon = 1+.18*(pulses[i]/max(pulses.max(), 1e-30))
        ax.fill_between(t, 1, ribbon, color=color, alpha=.18, linewidth=0, zorder=.4)
        ax.axhline(1, color=color, lw=.75, alpha=.3, zorder=.45)
        ax.plot(t, ribbon, color=color, lw=.95, alpha=.55, zorder=.5)
        ax.text(min(2*r/C+4.2, 46), 1+.67*.18, r'$\lambda_r(\theta)$', color=color,
                fontsize=7.2, ha='left', va='center', zorder=1,
                bbox=dict(facecolor='white', edgecolor='none', alpha=.72, pad=.2))
        lines = []
        for j, key in enumerate(keys):
            for k, method in enumerate(methods[:2]):
                line, = ax.plot(t, data[key+'_'+method+'_gamma'][i],
                                color=method_colors[j][k], ls=styles[k], lw=(1.35, 1.7)[k],
                                label=labels[k]+',\n'+regimes[j], alpha=1, zorder=2.5)
                lines.append(line)
        ax.set(xlim=(0, 50), ylim=(-.02, 1.22), yticks=np.linspace(0, 1, 6), xlabel=r'$t$ (ns)')
        if i == 0:
            ax.set_ylabel(r'$\gamma(t)$')
        else:
            ax.tick_params(labelleft=False)
        for line, position in zip(lines, positions[i]):
            label_curve(ax, line, *position)
        ax.text(.5, 1.105, rf'$R={r:g}\,\mathrm{{m}}$', transform=ax.transAxes,
                color=color, fontsize=9, fontweight='normal', ha='center', va='bottom')

    callouts = []
    for i, key in enumerate(keys):
        values = [data[key+'_'+method] for method in methods[:2]]
        points = [(ranges[np.argmax(v)], v.max()) for v in values]
        for j, point in enumerate(points):
            b.scatter(*point, marker='*', facecolor='white', edgecolor=method_colors[i][j],
                      s=(65, 80)[j], linewidth=(.95, 1)[j], zorder=7, clip_on=False)
        for j, point in enumerate(points):
            b.hlines(point[1], min(p[0] for p in points) if j == 0 else point[0], ranges[-1],
                     colors=method_colors[i][j], linestyles=(0, (1, 2)), lw=1.05, alpha=.72, zorder=4)
        free, opt = (point[1] for point in points)
        percent, y = 100*(free-opt)/free, np.sqrt(max(free, 1e-30)*max(opt, 1e-30))
        callouts.append((percent, y))
        transform = b.get_yaxis_transform()
        b.plot([1.045, 1.045], [opt, free], color=colors[i], lw=1.15,
               transform=transform, clip_on=False, zorder=8)
        for value in (free, opt):
            b.plot([1.045-.010, 1.045+.010], [value, value], color=colors[i], lw=1.15,
                   transform=transform, clip_on=False, zorder=8)
    b.annotate('max', points[0], xytext=(-34, 8), textcoords='offset points',
               color=method_colors[1][0], fontsize=7.8, fontweight='normal', ha='right', va='bottom',
               bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=.8), zorder=10,
               arrowprops=dict(arrowstyle='->', color=method_colors[1][0], lw=.95, shrinkA=1, shrinkB=1))
    for i in (1, 0):
        percent, y = callouts[i]
        b.text(1.088, y, f'{percent:.{2 if percent < 1 else 1}f}\\%\nred.',
               transform=transform, color=colors[i], fontsize=7.8, fontweight='normal',
               ha='left', va='center', clip_on=False, zorder=9)
        b.scatter(1.285, y, transform=transform, marker='s', s=74, facecolor=cmap(norm(max(percent, 0))),
                  edgecolor='#5A5A5A', linewidth=.6, clip_on=False, zorder=9)

    for ax, title in zip([a, b, c], ['Optimal design', 'Resulting error', 'Worst-case improvement']):
        ax.set_title(title, fontsize=10.5, pad=18, fontweight='normal')
    for i, ax in enumerate([a, b, c, *gamma_axes]):
        ax.text(-.09 if i > 2 else (-.18 if i == 2 else -.10),
                1.065 if i < 3 else 1.095, f'({chr(97+i)})',
                transform=ax.transAxes, fontsize=11, fontweight='bold', ha='left', va='center')
        ax.tick_params(labelsize=9.4)
        ax.xaxis.label.set_size(10.5)
        ax.yaxis.label.set_size(10.5)
        if ax is not c:
            ax.grid(axis='y', color='#D9DEE1', lw=.45)

    for left, right, text, fraction in [(a, b, 'evaluate objective', .14), (b, c, 'worst range case', .34)]:
        p, q = left.get_position(), right.get_position()
        gap = max(q.x0-p.x1, 0)
        pad = min(.016, .22*gap)
        x0, x1 = p.x1+pad, q.x0-pad
        length = min(.086, .9*max(x1-x0, 1e-4))
        center = np.clip(p.x1+fraction*gap, x0+.5*length, x1-.5*length)
        y = b.get_position().y0-.026
        fig.add_artist(FancyArrowPatch((center-length/2, y), (center+length/2, y),
            transform=fig.transFigure, arrowstyle='-|>', mutation_scale=13, lw=1.45,
            facecolor='#6E7276', edgecolor='#6E7276', connectionstyle='arc3,rad=.28',
            alpha=.82, clip_on=False, zorder=20))
        fig.text(center, y-.018, text, ha='center', va='top', fontsize=7.2,
                 color='#55585C', fontweight='semibold', linespacing=.88, zorder=25,
                 bbox=dict(facecolor='white', edgecolor='none', alpha=.86, pad=.35))
    for left, right in zip(gamma_axes, gamma_axes[1:]):
        p, q = left.get_position(), right.get_position()
        gap = max(q.x0-p.x1, 1e-4)
        x0, x1, y = p.x1+.26*gap, q.x0-.38*gap, max(p.y1, q.y1)-.041
        fig.add_artist(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure,
            arrowstyle='-|>', mutation_scale=11.5, lw=1.25, facecolor='#7A7D80',
            edgecolor='#7A7D80', alpha=.74, clip_on=False, zorder=18))
        fig.text(.5*(x0+x1), y+.008, 'distance\nattenuation', ha='center', va='bottom',
                 fontsize=6.8, color='#666A6D', linespacing=.86, zorder=19,
                 bbox=dict(facecolor='white', edgecolor='none', alpha=.82, pad=.25))

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save(fig, output/'navspdp_suffix_composite_figure.pdf', 'suffix', pad=.04)
    save(fig, output/'navspdp_suffix_composite_figure.png', 'suffix', pad=.04, dpi=180)
    plt.close(fig)
