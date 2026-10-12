"""Section 4: oracle-normalized regret; theta=(a, tau in ns, b per bin)."""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import linprog, minimize, minimize_scalar
from scipy.sparse.linalg import spsolve
from scipy.special import ndtr
from oracle import constraints

K, D, DELTA, SIGMA, TAU = 1000, 500, .1, 1., 30.
ROOT = Path(__file__).resolve().parent


def model(a, B, tau=TAU):
    edges = np.arange(K + 1) * DELTA
    f, df = np.zeros(K), np.zeros(K)
    for shift in (-K * DELTA, 0., K * DELTA):
        z = (edges - tau + shift) / SIGMA
        f += np.where(z[:-1] >= 0, ndtr(-z[:-1]) - ndtr(-z[1:]), ndtr(z[1:]) - ndtr(z[:-1]))
        phi = np.exp(-z * z / 2) / np.sqrt(2 * np.pi)
        df += (phi[:-1] - phi[1:]) / SIGMA
    lam = a * f + B / K
    v = np.column_stack((f, a * df, np.ones(K)))
    return -np.expm1(-lam), np.einsum("ri,rj,r->rij", v, v, 1 / np.expm1(lam)) / K, f


def oracle(a, B, tol=1e-6):
    p, S, pulse = model(a, B)
    eq, ub = constraints(p, D)
    fr = spsolve(sparse.vstack([eq, ub], format="csc"), np.r_[np.zeros(K), np.ones(K)])[:K]
    vertices, mats = [fr], [np.einsum("r,rij->ij", fr, S)]
    scale = np.trace(np.linalg.inv(mats[0]))
    weights, gamma = np.ones(1), fr.copy()
    for _ in range(250):
        I = np.einsum("r,rij->ij", gamma, S)
        inv = np.linalg.inv(I)
        obj = np.trace(inv)
        grad = -np.einsum("ij,rij->r", np.dot(inv, inv), S)
        gscale = np.max(np.abs(grad))
        c = np.r_[grad / gscale, np.zeros(K)]
        lp = linprog(c, A_ub=ub, b_ub=np.ones(K), A_eq=eq, b_eq=np.zeros(K),
                     bounds=(0, 1), method="highs", options={"dual_feasibility_tolerance": 1e-9,
                     "primal_feasibility_tolerance": 1e-9})
        if not lp.success:
            raise RuntimeError(lp.message)
        y = np.minimum(lp.ineqlin.marginals, 0)
        residual = c - ub.T @ y - eq.T @ lp.eqlin.marginals
        lower = gscale * (y.sum() + np.minimum(residual, 0).sum())
        gap = (np.dot(grad, gamma) - lower) / obj
        if gap <= tol:
            return dict(regret=scale/obj-1, opt_obj=obj, fr_obj=scale, relative_gap=gap,
                        gamma=gamma, gamma_fr=fr, pulse=pulse)
        vertices.append(lp.x[:K])
        mats.append(np.einsum("r,rij->ij", vertices[-1], S))
        M = np.array(mats)

        def loss(w):
            inv = np.linalg.inv(np.einsum("v,vij->ij", w, M))
            return np.trace(inv)/scale, -np.einsum("ij,vij->v", np.dot(inv, inv), M)/scale

        step = minimize_scalar(lambda t: np.trace(np.linalg.inv((1-t)*I+t*M[-1]))/scale,
                               bounds=(0, 1), method="bounded", options={"xatol": 1e-12}).x
        fit = minimize(loss, np.r_[weights*(1-step), step], jac=True, method="SLSQP",
                       bounds=[(0, 1)]*len(M), constraints={"type": "eq", "fun": lambda w: w.sum()-1,
                       "jac": lambda w: np.ones_like(w)}, options={"ftol": 1e-13, "maxiter": 300})
        weights = np.maximum(fit.x, 0)
        weights /= weights.sum()
        keep = weights > 1e-10
        weights = weights[keep] / weights[keep].sum()
        vertices, mats = np.array(vertices)[keep], M[keep]
        gamma = np.einsum('v,vr->r', weights, vertices)
        vertices, mats = list(vertices), list(mats)
    raise RuntimeError(f"Oracle gap {gap:g} exceeds tolerance at a={a:g}, B={B:g}")


def compute(workers):
    a, B = np.geomspace(1e-3, 10, 49), np.geomspace(.01, 20, 49)
    tasks = [(x, y) for y in B for x in a]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(oracle, *zip(*tasks)))
    data = {key: np.array([r[key] for r in results]).reshape(49, 49)
            for key in ("regret", "opt_obj", "fr_obj", "relative_gap")}
    data.update(a_grid=a, B_grid=B)
    inset = oracle(1, 3, tol=1e-8)
    data.update({"inset_"+key: inset[key] for key in ("gamma", "gamma_fr", "pulse", "opt_obj", "fr_obj")})
    return data


def plot(data, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
    from plotting import save
    plt.rcParams.update({"font.size": 10, "font.family": "serif", "font.serif": ["Computer Modern Roman"],
        "text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "axes.linewidth": .65, "savefig.transparent": True, "axes.labelsize": 12, "axes.titlesize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.dpi": 160, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#90989E", "axes.labelcolor": "#111827", "xtick.color": "#111827",
        "ytick.color": "#111827", "hatch.color": "#3D6D99", "hatch.linewidth": .35})
    a, B, regret, pulse = data["a_grid"], data["B_grid"], 100*data["regret"], data["inset_pulse"]
    blue, gray, dark, amber = "#3D6D99", "#58636B", "#253039", "#B77935"
    box = dict(facecolor="white", edgecolor="none", alpha=.82, pad=1)
    stroke = [pe.withStroke(linewidth=2.4, foreground="white", alpha=.95)]
    mass = max(sum(np.roll(pulse, u) for u in range(1, D+1)))
    fig = plt.figure(figsize=(9.2, 5.9))
    ax = fig.add_axes([.075, .125, .495, .810])
    gates = fig.add_axes([.665, .615, .300, .300], zorder=2)
    bar = fig.add_axes([.1225, .985, .400, .025], zorder=3)
    bound = fig.add_axes([.665, .145, .300, .295])
    ax.set(xscale="log", yscale="log", xlim=(1e-3, 10), ylim=(.01, 20))
    cmap = LinearSegmentedColormap.from_list("regret_amber", ["#FFFDF6", "#F4D99E", "#DFA653", "#AF6D30", "#633A20"])
    positive = regret[np.isfinite(regret) & (regret > 0)]
    lo = max(.02, np.percentile(positive, 5))
    hi = max(lo*10, np.percentile(positive, 98))
    ax.contourf(a, B, np.maximum(regret, lo), levels=np.geomspace(lo, hi, 72), cmap=cmap,
                norm=LogNorm(lo, hi), extend="both", antialiased=False, zorder=-1)
    ax.set_rasterization_zorder(0)
    aa, bb = np.geomspace(.001, 10, 140), np.geomspace(.01, 20, 140)
    rho = aa[None, :]*mass + bb[:, None]*D/K
    ax.contourf(aa, bb, rho, levels=[0, np.log(2)], colors=[(1, 1, 1, 0)], hatches=["///"], zorder=3)
    ax.contour(aa, bb, rho, levels=[np.log(2)], colors=[blue], linewidths=1.55, zorder=6)
    ax.text(.022, .14, "Free-running optimal\nfor uniform cyclic-average\nand worst-case loss among\npre-planned schedules\n($\\rho_\\star\\leq\\log 2$)",
            color=blue, fontsize=10.8, ha="center", va="center", bbox={**box, "alpha": .76}, zorder=7)
    contours = ax.contour(a, B, regret, levels=[3, 10, 30, 100, 300], colors="#f8fafc", linewidths=.8, alpha=.72)
    for level, target in [(10, (.30, .13)), (30, (.72, .35)), (100, (2.8, .62)), (300, (3.25, 2.45))]:
        segments = contours.allsegs[list(contours.levels).index(level)]
        points = np.concatenate([s for s in segments if len(s) >= 3])
        xy = points[np.argmin(np.sum((ax.transData.transform(points)-ax.transData.transform(target))**2, axis=1))]
        ax.text(*xy, rf"{level}\%", color=dark, fontsize=10.8, ha="center", va="center", zorder=5,
                bbox={**box, "alpha": .8, "pad": .35})
    ax.annotate("", xy=(.41, .91), xytext=(.15, .75), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="white", lw=1.5, mutation_scale=12), zorder=9)
    ax.text(.255, .855, "Increasing pile-up", transform=ax.transAxes, ha="center", va="center",
            rotation=31, color="white", fontsize=10.3, zorder=9)
    bar.imshow(np.linspace(np.log10(lo), np.log10(hi), 512)[None, :], aspect="auto", origin="lower",
               cmap=cmap, extent=[np.log10(lo), np.log10(hi), 0, 1])
    ticks = np.array([.3, 1, 3, 10, 30, 100, 300]); ticks = ticks[(ticks >= lo) & (ticks <= hi)]
    bar.set(xticks=np.log10(ticks), xticklabels=[f"{t:g}" for t in ticks], yticks=[])
    bar.xaxis.set_label_position("top")
    bar.set_xlabel(r"Oracle-normalized regret (\%)", labelpad=3, fontsize=10.8, color=dark)
    bar.tick_params(axis="x", labelsize=8.5, width=.8, length=3)
    for spine in bar.spines.values(): spine.set_visible(False)
    ax.set_xlabel(r"$a$ (signal photons/period)", fontsize=10.8)
    ax.set_ylabel(r"$B$ (background photons/period)", fontsize=10.8)
    ax.grid(which="major", color="white", alpha=.18, lw=.45)
    for label, cite, xy, text, align in [
        ("Rapp low-flux", "50", (.00135, .0338), (.00158, .0395), (0, .5)),
        ("Rapp high-flux", "52", (.31, 1.7), (.31, 1.30), (.5, 1)),
        ("Gupta high-ambient", "22", (.22, 11), (.27, 12.5), (0, .5))]:
        ax.scatter(*xy, s=46, marker="o", color=dark, edgecolor="white", linewidth=.85, zorder=9)
        parts = [TextArea(s, textprops=dict(color=c, fontsize=9))
                 for s, c in [(label+"~[", dark), (cite, "#008000"), ("]", "black")]]
        ax.add_artist(AnnotationBbox(HPacker(children=parts, align="baseline", pad=0, sep=0), text,
                      box_alignment=align, xybox=(.77 if align[0] == 0 else -.2, -.2), boxcoords="offset points",
                      pad=.0972, bboxprops=dict(facecolor="white", edgecolor="none", alpha=.82), zorder=9))
    ax.scatter(1, 3, marker="D", s=64, color=gray, edgecolor="white", linewidth=.95, zorder=8)
    ax.text(.66, 3, "Example for panel (b)", color=gray, fontsize=10.8, ha="right", va="center", bbox=box, zorder=8)
    phase = np.arange(K)*DELTA
    gates.set_facecolor("none")
    pulse_y = 1+.13*pulse/pulse.max()
    gates.fill_between(phase, 1, pulse_y, color=blue, alpha=.18, linewidth=0, zorder=0)
    gates.plot(phase, pulse_y, color=blue, lw=1, alpha=.78, zorder=1)
    gates.axhline(1, color=blue, lw=.9, alpha=.72, zorder=1)
    for x in (TAU, TAU+D*DELTA):
        gates.plot([x, x], [0, 1.19], color="#374151", lw=.65, ls=":", alpha=.65, zorder=1.6)
    gates.annotate("", xy=(80, 1.19), xytext=(30, 1.19), zorder=3,
                   arrowprops=dict(arrowstyle="|-|", color="#374151", lw=1, mutation_scale=5, shrinkA=0, shrinkB=0))
    gates.text(55, 1.255, r"dead time $t_d=50\,\mathrm{ns}$", color="#374151", fontsize=9,
               ha="center", va="center", path_effects=stroke, zorder=3)
    gates.plot(phase, data["inset_gamma"], color=amber, lw=1.75)
    gates.plot(phase, data["inset_gamma_fr"], color=gray, lw=1.35, ls="--")
    gates.set(xlim=(0, 104.5), ylim=(0, 1.3), yticks=[0, .5, 1])
    gates.set_xlabel(r"phase time $r\Delta$ (ns)", fontsize=10.8, labelpad=1.5)
    fig.text(.614, .765, r"gate-open fraction $\gamma_r$", color="#111827", fontsize=10,
             rotation=90, ha="center", va="center", zorder=4)
    gates.tick_params(labelsize=8, pad=1.5)
    for xy, text, color, ha in [((37.5, 1.095), "pulse shape (rescaled)", blue, "left"),
        ((37, .575), r"$\gamma_{\star,r}$ (oracle)", amber, "left"),
        ((96.5, .70), r"$\gamma^{\mathrm{fr}}_r$ (free-running)", gray, "right")]:
        gates.text(*xy, text, color=color, fontsize=9, ha=ha, va="center", path_effects=stroke)
    for side, spine in gates.spines.items():
        spine.set(linewidth=.8, edgecolor="#90989E", visible=side in {"left", "bottom"})
    rho = a[None, :]*mass + B[:, None]*D/K
    x = np.geomspace(.005, 1.5, 401)
    bound.plot(x, 100*np.expm1(x), "--", color=gray, lw=1.2)
    for col, color, label, row, offset, ha in zip([0, 12, 24, 36],
        ["#356C99", "#2D8C7C", "#BE2D4B", "#85628A"], ["0.001", "0.01", "0.1", "1"],
        [6, 11, 11, 23], [(-7, -18), (4, -13), (12, -8), (-7, 0)], ["center", "center", "center", "right"]):
        use = rho[:, col] <= 1.5
        bound.plot(rho[use, col], regret[use, col], color=color, lw=1.35, marker="o", markersize=2.4,
                   markevery=3, markeredgecolor="white", markeredgewidth=.3, zorder=3)
        bound.annotate(f"$a={label}$", (rho[row, col], regret[row, col]), xytext=offset, textcoords="offset points",
                       color=color, fontsize=8.5, ha=ha, va="center", zorder=5, path_effects=stroke)
    bound.set(xscale="log", yscale="log", xlim=(.005, 1.6), ylim=(.1, 500))
    bound.set_xlabel(r"Pile-up $\rho$", fontsize=10.8)
    bound.set_ylabel("Oracle-normalized\nregret (\\%)", fontsize=10, labelpad=3)
    bound.tick_params(labelsize=9)
    bound.grid(color="#E9ECEE", lw=.5)
    near = np.array([[x, 100*np.expm1(x)] for x in (.019, .019*1.2)])
    dx, dy = np.diff(bound.transData.transform(near), axis=0)[0]
    bound.annotate(r"$e^\rho-1$ (Theorem 4.1)", near[0], xytext=(0, 5), textcoords="offset points",
                   color=gray, fontsize=8.3, rotation=np.degrees(np.arctan2(dy, dx)), rotation_mode="anchor",
                   ha="left", va="bottom", zorder=5)
    bound.set_title("Regret versus the bound", fontsize=10.8, pad=8)
    for axis, letter in ((ax, "(a)"), (gates, "(b)"), (bound, "(c)")):
        axis.text(0, 1.035, letter, transform=axis.transAxes, fontsize=11, ha="left", va="bottom")
    for extension in ("png", "pdf"):
        save(fig, output/f"figure2.{extension}", "regret", pad=.1)
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="plot existing output without recomputing")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    output = ROOT / "output"
    output.mkdir(exist_ok=True)
    if args.plot:
        with np.load(output/"regret.npz") as z: data = dict(z)
    else:
        data = compute(args.workers)
        np.savez_compressed(output/"regret.npz", **data)
    plot(data, output)
