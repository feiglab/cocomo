from __future__ import annotations

import gzip
import logging
import re
import sys
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp

logging.disable(logging.CRITICAL)

kb = 0.008314462618
T = 300

tics = {}
tics["dist"] = [5.6, 5.8, 6.0, 6.2, 6.4, 6.6, 6.8, 7.0, 7.2]

minmax = {}
minmax["dist"] = [5.5, 7.2]

label = {}
label["dist"] = "Distance [nm]"

colors1d = ["blue", "red", "green", "orange", "magenta", "cyan", "brown", "pink", "lime"]

plt.rcParams.update(
    {
        "font.size": 20,
        "font.family": "monospace",
        "font.weight": "normal",
        "axes.titlesize": 24,
        "axes.labelsize": 22,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 18,
        "figure.titlesize": 20,
    }
)


# umbrella sampling


def _open_text_auto(fname):
    path = Path(fname)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def _choose_optional_gz(fname):
    fname = Path(fname)
    if fname.exists():
        return fname

    gz_name = Path(str(fname) + ".gz")
    if gz_name.exists():
        return gz_name
    return None


def _parse_simple_header(fname, known_cols):
    with _open_text_auto(fname) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue

            tokens = [tok.lstrip("#!").strip().lower() for tok in s.split()]
            tokens = [tok for tok in tokens if tok]
            if tokens and tokens[0] == "fields":
                tokens = tokens[1:]

            if tokens and all(tok in known_cols for tok in tokens):
                return tokens
            return None
    return None


def _count_data_columns(fname, *, skiprows=1):
    with _open_text_auto(fname) as fh:
        for i, line in enumerate(fh):
            if i < skiprows:
                continue
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            return len(s.split())
    return None


def _read_umbrella_table(
    fname,
    *,
    required_cols,
    optional_cols=(),
    verbose=False,
):
    chosen = _choose_optional_gz(fname)
    if chosen is None:
        if verbose:
            gz_name = Path(str(Path(fname)) + ".gz")
            print(f"WARNING: no {fname} or {gz_name} found")
        return None

    if verbose:
        print(f"reading umbrella data from {chosen}")

    required_cols = list(required_cols)
    optional_cols = list(optional_cols)
    all_cols = required_cols + optional_cols

    known_cols = set(all_cols)
    cols = _parse_simple_header(chosen, known_cols)
    if cols is None:
        ncols = _count_data_columns(chosen, skiprows=1)
        if ncols is None:
            return pd.DataFrame(columns=all_cols)
        if ncols < len(required_cols) or ncols > len(all_cols):
            raise ValueError(
                f"Could not determine columns for {chosen}: found {ncols} data columns"
            )
        cols = all_cols[:ncols]

    dtype = {}
    for col in cols:
        dtype[col] = int if col.endswith("step") else float

    return pd.read_csv(
        chosen,
        sep=r"\s+",
        engine="python",
        names=cols,
        usecols=range(len(cols)),
        comment="#",
        skiprows=1,
        dtype=dtype,
        na_values=["nan", "NaN", "INF", "inf", "-inf"],
        on_bad_lines="skip",
    )


def read_umbrella_bias(dir, umbrellas, *, verbose=False):
    required_cols = [
        "step",
        "xbias",
        "ybias",
        "zbias",
    ]
    optional_cols = ["extra"]

    frames = []
    dir = Path(dir)

    for u in umbrellas:
        fname = dir / u / "bias.dat"
        df = _read_umbrella_table(
            fname,
            required_cols=required_cols,
            optional_cols=optional_cols,
            verbose=False,
        )

        if df is None:
            if verbose:
                print(f"WARNING: no bias.dat or bias.dat.gz for umbrella {u}")
            continue

        df["ubias"] = df["xbias"] + df["ybias"] + df["zbias"]
        df.insert(0, "umbrella", u)
        frames.append(df)

        if verbose:
            print(f"read {fname} ({len(df)} rows)")

    if not frames:
        if verbose:
            print("No bias data read from any umbrella.")
        return {}

    out = pd.concat(frames, ignore_index=True).set_index(["umbrella", "step"]).sort_index()

    return {u: g.droplevel(0) for u, g in out.groupby(level=0)}


def read_umbrella_geometry(fname, *, verbose=False):
    required_cols = [
        "gstep",
        "xdist",
        "ydist",
        "zdist",
    ]
    optional_cols = ["dist", "extra"]

    df = _read_umbrella_table(
        fname,
        required_cols=required_cols,
        optional_cols=optional_cols,
        verbose=verbose,
    )
    if df is not None and "dist" not in df.columns:
        xyz = df[["xdist", "ydist", "zdist"]].to_numpy(dtype=float)
        df["dist"] = np.linalg.norm(xyz, axis=1)
    return df


_RUN_RE = re.compile(r"^run_" r"(\d+(?:\.\d+)?)" r"(?:_(\d+(?:\.\d+)?))?$")


def find_run_dirs(dir: str) -> list[str]:
    base = Path(dir)
    paths: list[str] = []

    for p in base.iterdir():
        if p.is_dir() and _RUN_RE.match(p.name):
            paths.append(p.name)

    def sort_key(name: str) -> tuple[float, int, float]:
        m = _RUN_RE.match(name)
        assert m is not None
        first = float(m.group(1))
        second_s = m.group(2)

        if second_s is None:
            return first, 0, 0.0
        return first, 1, float(second_s)

    return sorted(paths, key=sort_key)


def _normalize_bias_tags(
    biasval: str | list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    if isinstance(biasval, str):
        tags = [biasval]
    else:
        tags = list(biasval)

    if not tags:
        raise ValueError("biasval must contain at least one tag")

    out: list[str] = []
    for tag in tags:
        if tag not in out:
            out.append(tag)
    return tuple(out)


def _combined_bias_tag(tags: tuple[str, ...]) -> str:
    if len(tags) == 1:
        return tags[0]
    return "_".join(tags)


def _sum_bias_columns(df: pd.DataFrame, tags: tuple[str, ...]) -> pd.Series:
    missing = [tag for tag in tags if tag not in df.columns]
    if missing:
        raise KeyError(f"Missing bias columns: {missing}")
    return df.loc[:, list(tags)].sum(axis=1)


def _ensure_combined_bias_column(df: pd.DataFrame, tags: tuple[str, ...]) -> str:
    bias_key = _combined_bias_tag(tags)
    if len(tags) > 1 or bias_key not in df.columns:
        df[bias_key] = _sum_bias_columns(df, tags)
    return bias_key


def process_umbrella(
    tag="hh",
    *,
    dir=".",
    path=None,
    verbose=False,
    biasval="xbias",
    skip=0,
    trajname="CA.xtc",
):

    bias_tags = _normalize_bias_tags(biasval)
    bias_key = _combined_bias_tag(bias_tags)

    if path is None:
        path = find_run_dirs(dir)
    path = [str(p) for p in path]
    if not path:
        raise ValueError("process_umbrella: no umbrella run directories found")

    geo_path = Path(dir) / path[0] / "geometry.dat"
    df = read_umbrella_geometry(geo_path, verbose=verbose)
    if df is None:
        raise FileNotFoundError(f"Missing geometry data: {geo_path}")

    nwin = len(path)
    nper = len(df) // nwin
    data = {path[i]: df.iloc[i * nper : (i + 1) * nper].reset_index(drop=True) for i in range(nwin)}

    bias = read_umbrella_bias(dir, path, verbose=verbose)
    for p in path:
        _ensure_combined_bias_column(bias[p], bias_tags)

    for i in range(nwin):
        bindiv = bias[path[i]].iloc[i * nper : (i + 1) * nper].reset_index(drop=True)
        data[path[i]] = pd.merge(
            data[path[i]], bindiv, left_index=True, right_index=True, how="inner"
        )
        _ensure_combined_bias_column(data[path[i]], bias_tags)

    mask = {}
    for p in path:
        mask[p] = pd.Series(True, index=data[p].index)
        if skip > 0:
            mask[p].iloc[:skip] = False
        data[p] = data[p].loc[mask[p]].copy()
        data[p].reset_index(drop=True, inplace=True)

    for p in path:
        wham = unbias_wham(np.asarray(data[p][[bias_key]], dtype=float))
        data[p]["ww"] = pd.Series(np.exp(wham["logW"]) / np.sum(np.exp(wham["logW"])))

    combmask = pd.concat([mask[p] for p in path], ignore_index=True)
    combmask_arr = combmask.to_numpy(dtype=bool)
    data["comb"] = pd.concat([data[p] for p in path], ignore_index=True)

    bias_matrix = np.column_stack(
        [np.asarray(bias[p][bias_key].iloc[combmask_arr], dtype=float) for p in path]
    )
    counts = [len(data[p]) for p in path]

    mbar = unbias_mbar(bias_matrix, counts=counts)

    data["comb"]["ww"] = pd.Series(mbar["ww"])

    data["mbar"] = mbar
    data["bias_matrix"] = bias_matrix

    data["bias"] = bias
    data["counts"] = counts
    data["sets"] = path

    return data


# unbiasing and projecting onto reaction coordinates


def unbias_wham(
    bias,
    *,
    kT: float = kb * T,
    frame_weight=None,
    traj_weight=None,
    maxiter: int = 1000,
    threshold: float = 1e-20,
    verbose: bool = False,
):

    nframes = bias.shape[0]
    ntraj = bias.shape[1]

    if frame_weight is None:
        frame_weight = np.ones(nframes)
    if traj_weight is None:
        traj_weight = np.ones(ntraj)

    assert len(traj_weight) == ntraj
    assert len(frame_weight) == nframes

    shifted_bias = bias / kT

    shifts0 = np.min(shifted_bias, axis=0)
    shifted_bias -= shifts0[np.newaxis, :]
    shifts1 = np.min(shifted_bias, axis=1)
    shifted_bias -= shifts1[:, np.newaxis]

    expv = np.exp(-shifted_bias)

    Z = np.ones(ntraj)

    Zold = Z.copy()

    if verbose:
        sys.stderr.write("WHAM: start\n")
    for nit in range(maxiter):
        weight = 1.0 / np.matmul(expv, traj_weight / Z) * frame_weight
        Z = np.matmul(weight, expv)
        Z /= np.sum(Z * traj_weight)
        ratio = np.maximum(Z, 1e-300) / np.maximum(Zold, 1e-300)
        eps = np.sum(np.log(ratio) ** 2)
        Zold = Z.copy()
        if verbose:
            sys.stderr.write("WHAM: iteration " + str(nit) + " eps " + str(eps) + "\n")
        if eps < threshold:
            break
    logW = np.log(weight) + shifts1

    if verbose:
        sys.stderr.write("WHAM: end")

    return {
        "logW": logW,
        "logZ": np.log(Z) - shifts0,
        "nit": nit,
        "eps": eps,
        "ww": np.exp(logW) / np.sum(np.exp(logW)),
    }


def unbias_mbar(bias, *, kT=kb * T, counts=None, verbose=False):
    beta = 1.0 / (kT) if kT is not None else 1.0
    u_kn = (beta * np.asarray(bias, float)).T  # (K,N)

    if counts is None:
        N, K = bias.shape
        n_per = N // K
        state_of_sample = np.repeat(np.arange(K, dtype=int), n_per)
        counts = np.bincount(state_of_sample, minlength=K)

    from pymbar import MBAR

    mbar = MBAR(u_kn, counts, verbose=verbose, maximum_iterations=500)

    log_den = logsumexp(mbar.f_k[:, None] - mbar.u_kn + np.log(mbar.N_k)[:, None], axis=0)
    logw = -log_den
    ww = np.exp(logw - logsumexp(logw))

    return {"mbar": mbar, "logW": logw, "ww": ww}


def _is_energy_vector(obj) -> bool:
    if isinstance(obj, pd.Series):
        return True
    if isinstance(obj, np.ndarray):
        return obj.ndim <= 1
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return True
        return np.isscalar(obj[0])
    return False


def _validate_energy_offset(energy_offset, nframes: int, *, name="energy_offset") -> np.ndarray:
    arr = np.asarray(energy_offset, float).ravel()
    if arr.size != nframes:
        raise ValueError(f"{name} length mismatch: got {arr.size}, expected {nframes}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _get_frame_energy(
    data: pd.DataFrame,
    *,
    energy_offset=None,
    energy_col=None,
) -> np.ndarray | None:
    if energy_offset is not None and energy_col is not None:
        raise ValueError("Provide either energy_offset or energy_col, not both")
    if energy_col is not None:
        if energy_col not in data.columns:
            raise KeyError(f"Missing energy column: {energy_col}")
        energy_offset = data[energy_col]
    if energy_offset is None:
        return None
    return _validate_energy_offset(energy_offset, len(data))


def _apply_energy_offset_to_weights(
    w: np.ndarray,
    energy_offset=None,
    *,
    kT: float = kb * T,
) -> np.ndarray:
    w = np.asarray(w, float).ravel()
    if energy_offset is None:
        return w

    de = _validate_energy_offset(energy_offset, w.size)
    shift = float(np.min(de))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        fac = np.exp(-(de - shift) / float(kT))
    return w * fac


def _normalize_energy_structure(obj, energy_offset):
    if _is_tiledata_dict(obj):
        return energy_offset

    if isinstance(obj, (list, tuple)):
        items = list(obj)
    else:
        raise TypeError("energy_offset shape does not match input data")

    if energy_offset is None:
        return [_normalize_energy_structure(it, None) for it in items]

    if len(items) == 1 and _is_tiledata_dict(items[0]) and _is_energy_vector(energy_offset):
        return [_normalize_energy_structure(items[0], energy_offset)]

    if not isinstance(energy_offset, (list, tuple)):
        raise ValueError("energy_offset must mirror the input data structure")

    if len(energy_offset) != len(items):
        raise ValueError("energy_offset length must match the input data structure")

    return [_normalize_energy_structure(it, eo) for it, eo in zip(items, energy_offset)]


def _pmf1d_from_tiledata(
    data,
    tag="dist",
    *,
    usembar=False,
    kT=kb * T,
    nbins=50,
    rang=None,
    energy_offset=None,
    energy_col=None,
):
    if isinstance(tag, list):
        taglist = list(tag)
        stag = taglist[0].rstrip("0123456789")
    else:
        taglist = None
        stag = str(tag).rstrip("0123456789")

    energy = _get_frame_energy(
        data["comb"],
        energy_offset=energy_offset,
        energy_col=energy_col,
    )

    if taglist is not None:
        dplotlist = []
        eplotlist = []
        for t in taglist:
            dp = data["comb"][[t, "ww"]].fillna(0).copy()
            dp.columns = [stag, "ww"]
            dplotlist.append(dp)
            if energy is not None:
                eplotlist.append(energy)

        dplot = pd.concat(dplotlist, ignore_index=True)
        denergy = None if energy is None else np.concatenate(eplotlist)
        return pmf1d_from_weights(
            dplot,
            stag,
            nbins=nbins,
            kT=kT,
            rang=rang,
            energy_offset=denergy,
        )

    if usembar:
        res = pmf1d_mbar(
            data["mbar"],
            data["comb"],
            tag,
            nbins=nbins,
            kT=kT,
            rang=rang,
            energy_offset=energy,
        )
        if stag != tag:
            res["pmf"] = res["pmf"].rename(columns={tag: stag})
            if res["dpmf"] is not None:
                res["dpmf"] = res["dpmf"].rename(columns={tag: stag})
            res["ranges"] = res["ranges"].rename(columns={tag: stag})
        return res

    dplot = data["comb"][[tag, "ww"]].fillna(0).copy()
    if stag != tag:
        dplot = dplot.rename(columns={tag: stag})
    return pmf1d_from_weights(
        dplot,
        stag,
        nbins=nbins,
        kT=kT,
        rang=rang,
        energy_offset=energy,
    )


def pmf1d_mbar(
    mbar,
    data,
    tag,
    *,
    kT=kb * T,
    nbins=100,
    rang=None,
    verbose=False,
    energy_offset=None,
    energy_col=None,
):
    """1D PMF via PyMBAR FES with optional target-state reweighting."""
    if "mbar" in mbar:
        mbar = mbar["mbar"]

    x = np.asarray(data[tag], float).ravel()
    energy = _get_frame_energy(
        data,
        energy_offset=energy_offset,
        energy_col=energy_col,
    )
    if energy is None:
        u_n = np.zeros(x.shape[0], float)
    else:
        u_n = energy / float(kT)

    if rang is None:
        eps = 1e-12 * (float(x.max()) - float(x.min()) + 1.0)
        edges_1d = np.linspace(float(x.min()), float(x.max()) + eps, nbins + 1)
    else:
        xmin, xmax = float(rang[0]), float(rang[1])
        edges_1d = np.linspace(xmin, xmax, nbins + 1)

    centers = 0.5 * (edges_1d[:-1] + edges_1d[1:])

    from pymbar import FES

    fes = FES(
        mbar.u_kn,
        mbar.N_k,
        mbar_options=dict(verbose=verbose, maximum_iterations=500),
    )
    _ = fes.generate_fes(
        u_n,
        x[:, None],
        fes_type="histogram",
        histogram_parameters={"bin_edges": [edges_1d]},
    )

    hist, _ = np.histogram(x, bins=edges_1d)
    occ = hist > 0
    if not np.any(occ):
        raise ValueError("pmf1d_mbar: no occupied bins (check input data/range)")

    centers_occ = centers[occ]
    out = fes.get_fes(
        centers_occ,
        reference_point="from-lowest",
        uncertainty_method="analytical",
    )

    f_i = np.full(centers.shape, np.nan, dtype=float)
    df_i = np.full(centers.shape, np.nan, dtype=float)
    f_i[occ] = np.asarray(out["f_i"], float).ravel()
    if out.get("df_i") is not None:
        df_i[occ] = np.asarray(out["df_i"], float).ravel()

    F_kT = f_i * float(kT)
    dF_kT = df_i * float(kT)

    if np.any(np.isfinite(F_kT)):
        F_kT = F_kT - np.nanmin(F_kT)

    idx = pd.Index(np.arange(nbins), name="x")
    pmf1d = pd.DataFrame({f"{tag}": F_kT}, index=idx)
    dpmf = pd.DataFrame({f"{tag}": dF_kT}, index=idx)
    ranges = pd.DataFrame({tag: centers}, index=idx)

    return dict(
        edges=[edges_1d],
        centers=centers,
        F_kT=F_kT,
        dF_kT=dF_kT,
        pmf=pmf1d,
        dpmf=dpmf,
        ranges=ranges,
    )


def pmf1d_from_weights(
    data,
    tag,
    *,
    wtag="ww",
    kT=kb * T,
    nbins=100,
    rang=None,
    energy_offset=None,
    energy_col=None,
):
    """
    Project onto a 1D reaction coordinate x using weights.

    If `energy_offset` is given, it is interpreted as a per-frame target-state
    energy offset in kJ/mol and applied as exp(-energy_offset / kT) before
    histogramming.
    """
    x = np.asarray(data[tag], float).ravel()
    w = np.asarray(data[wtag], float).ravel()
    energy = _get_frame_energy(
        data,
        energy_offset=energy_offset,
        energy_col=energy_col,
    )
    w = _apply_energy_offset_to_weights(w, energy, kT=kT)

    assert x.shape == w.shape, "x and weights must have same length"

    if rang is None:
        pad = 1e-12 * (x.max() - x.min() + 1.0)
        x_edges = np.linspace(x.min(), x.max() + pad, nbins + 1)
    else:
        xmin, xmax = rang
        x_edges = np.linspace(xmin, xmax, nbins + 1)

    H, xe = np.histogram(x, bins=x_edges, weights=w)
    H = H.astype(float)
    Hsum = H.sum()
    P = H / Hsum if Hsum > 0 else H

    with np.errstate(divide="ignore", invalid="ignore"):
        F = -np.log(P)
    finite = np.isfinite(F)
    if np.any(finite):
        F -= np.nanmin(F)
    F_kT = F * kT

    x_centers = 0.5 * (xe[:-1] + xe[1:])

    idx = pd.Index(np.arange(nbins), name="x")
    pmf1d = pd.DataFrame({f"{tag}": F_kT}, index=idx)
    ranges = pd.DataFrame({tag: x_centers})

    return dict(
        edges=xe,
        centers=x_centers,
        F_kT=F_kT,
        P=P,
        pmf=pmf1d,
        dpmf=None,
        ranges=ranges,
    )


# plotting


def dist1D(
    data: pd.DataFrame,
    ranges: pd.DataFrame,
    *,
    err: None,
    fmin=0.0,
    fmax=20.0,
    size: int = 1,
    label=label,
    minmax=minmax,
    tics=tics,
    colors=colors1d,
    lw=2,
    key=None,
    markers=None,
    tag="dist",
    mode="together",
    horizontal=None,
    vertical=None,
    save=None,
) -> None:

    if mode == "together":
        nplots = 1
        rows = 1
        cols = 1
    else:
        nplots = len(data)
        rows = int((nplots + 1) / 2)
        cols = 2

    if key is not None:
        xoff = 3
    else:
        xoff = 1

    fig, ax = plt.subplots(
        rows,
        cols,
        figsize=(cols * 5 * size + xoff, rows * 4 * size + 1),
        dpi=75,
        constrained_layout=True,
    )

    xlabel = "distance [nm]"
    ylabel = "[kJ/mol]"

    xmin = 5.0
    xmax = 10.0

    if tag is not None:
        if label is not None and tag in label:
            xlabel = label[tag]
        if tics is not None and tag in tics:
            xtics = tics[tag]
        else:
            xtics = None
        if minmax is not None and tag in minmax:
            xmin = minmax[tag][0]
            xmax = minmax[tag][1]
            if xtics is not None:
                xtics = [x for x in xtics if x >= xmin and x <= xmax]

    if nplots > 1:
        ax = ax.ravel()

    for i, d in enumerate(data):
        X = ranges[i][tag]
        Y = d[tag]

        if mode == "together":
            axi = ax
        else:
            axi = ax[i]

        if i < len(colors):
            linecolor = colors[i]
        else:
            linecolor = (0.5, 0.5, 0.5)

        if key is not None and i < len(key):
            keyname = key[i]
        else:
            keyname = ""

        axi.plot(X, Y, color=linecolor, label=keyname, linewidth=lw)
        axi.set_xlabel(xlabel)  # , fontsize=20)
        axi.set_ylabel(ylabel)  # , fontsize=20)
        axi.set_xlim(xmin, xmax)
        axi.set_ylim(fmin, fmax)

        if err is not None and err[i] is not None:
            axi.fill_between(X, Y - err[i][tag], Y + err[i][tag], alpha=0.3, color=linecolor)

        if xtics is not None:
            axi.set_xticks(xtics)

        if markers is not None:
            for m in markers:
                axi.plot(
                    m[tag], 1.0, "x", color=m["col"], markersize=int(12 * size), markeredgewidth=4
                )
                if len(m) > 3:
                    axi.annotate(
                        m["label"],
                        xy=(m[tag], 1.0),
                        xytext=(0, 16 * m["pos"] * size + 6 * size),
                        color=m["col"],
                        textcoords="offset points",
                        ha="center",
                        va="top",
                        fontsize=int(14 * size),
                    )

        if vertical is not None:
            axi.axvline(x=vertical, color="#808080", linestyle="--", linewidth=3)
        if horizontal is not None:
            axi.axhline(y=horizontal, color="#808080", linestyle="--", linewidth=3)

    if nplots > 1:
        for i in range(nplots, rows * cols):
            ax[i].remove()
    else:
        if key is not None:
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0)

    if save:
        fig.savefig(save, dpi=300)

    plt.show()


def _is_tiledata_dict(obj) -> bool:
    return isinstance(obj, dict) and "comb" in obj


def _nanmean_axis(a: np.ndarray, axis: int = 0) -> np.ndarray:
    a = np.asarray(a, float)
    mask = np.isfinite(a)
    n = np.sum(mask, axis=axis)
    s = np.sum(np.where(mask, a, 0.0), axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = s / n
    mean = np.where(n > 0, mean, np.nan)
    return mean


def _nansem_axis(a: np.ndarray, axis: int = 0) -> np.ndarray:
    a = np.asarray(a, float)
    mask = np.isfinite(a)
    n = np.sum(mask, axis=axis)

    s = np.sum(np.where(mask, a, 0.0), axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = s / n
    mean = np.where(n > 0, mean, np.nan)

    mean_exp = np.expand_dims(mean, axis=axis)
    dev = np.where(mask, a - mean_exp, 0.0)
    ss = np.sum(dev * dev, axis=axis)

    with np.errstate(divide="ignore", invalid="ignore"):
        var = ss / (n - 1)
    var = np.where(n > 1, var, np.nan)

    sd = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        sem = sd / np.sqrt(n)
    sem = np.where(n > 1, sem, np.nan)
    return sem


def _interp_to_grid(x_ref: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    if x.size == 0 or y.size == 0:
        return np.full(x_ref.shape, np.nan, dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    with np.errstate(invalid="ignore"):
        return np.interp(x_ref, x, y, left=np.nan, right=np.nan)


def average_pmf1d(pmf, ranges, tag, *, method="linear", kT=kb * T):
    """Average 1D PMFs on a common grid.

    Parameters
    ----------
    pmf : list[pd.DataFrame]
        Each dataframe has a column `tag` with free energies (kJ/mol).
    ranges : list[pd.DataFrame]
        Each dataframe has a column `tag` with bin centers.
    tag : str
        Column name for x / PMF.
    method : {"linear","boltzmann"}
        Linear averages free energies; boltzmann averages exp(-F/kT).
    kT : float
        Thermal energy in same units as free energies (kJ/mol).

    Returns
    -------
    dict with keys: pmf, dpmf, ranges
        dpmf contains SEM (per bin) across the input PMFs.
    """
    if len(pmf) != len(ranges):
        raise ValueError("average_pmf1d: pmf and ranges length mismatch")
    if len(pmf) == 0:
        raise ValueError("average_pmf1d: empty input")

    m = str(method).lower().strip()
    if m in {"linear", "fe", "free_energy"}:
        mode = "linear"
    elif m in {"boltzmann", "boltz", "exp"}:
        mode = "boltzmann"
    else:
        raise ValueError(f"average_pmf1d: unknown method {method!r}")

    x_ref = np.asarray(ranges[0][tag], float).ravel()
    if x_ref.size == 0:
        raise ValueError("average_pmf1d: empty grid")

    y_stack = []
    for p, r in zip(pmf, ranges):
        x = np.asarray(r[tag], float).ravel()
        y = np.asarray(p[tag], float).ravel()
        if x.shape != x_ref.shape or not np.allclose(x, x_ref, atol=1e-9, rtol=0.0):
            y = _interp_to_grid(x_ref, x, y)
        y_stack.append(y)

    Y = np.vstack(y_stack)  # (nrep, nbins)

    if mode == "linear":
        mean = _nanmean_axis(Y, axis=0)
        sem = _nansem_axis(Y, axis=0)
    else:
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            W = np.exp(-Y / float(kT))
        mean_w = _nanmean_axis(W, axis=0)
        sem_w = _nansem_axis(W, axis=0)

        mean = np.full(mean_w.shape, np.nan, dtype=float)
        sem = np.full(mean_w.shape, np.nan, dtype=float)

        ok = np.isfinite(mean_w) & (mean_w > 0.0)
        mean[ok] = -float(kT) * np.log(mean_w[ok])
        sem[ok] = float(kT) * sem_w[ok] / mean_w[ok]

    idx = pmf[0].index
    pmf_avg = pd.DataFrame({tag: mean}, index=idx)
    dpmf_avg = pd.DataFrame({tag: sem}, index=idx)
    ranges_avg = pd.DataFrame({tag: x_ref}, index=idx)

    return {"pmf": pmf_avg, "dpmf": dpmf_avg, "ranges": ranges_avg}


def plot1D_grouped(
    groups,
    tag="dist",
    *,
    average="boltzmann",
    usembar=False,
    minmax=minmax,
    tics=tics,
    label=label,
    colors=colors1d,
    key=None,
    kbT=kb * T,
    nbins=50,
    fmin=0.0,
    fmax=20.0,
    size=1.5,
    markers=None,
    offset=None,
    matchflat=None,
    matchzero=False,
    average_overlay=False,
    vertical=None,
    horizontal=None,
    energy_offset=None,
    energy_col=None,
    save=None,
):
    """Average replicate PMFs per group and plot group averages together."""
    if isinstance(groups, dict) and "comb" not in groups:
        group_names = list(groups.keys())
        group_list = list(groups.values())
        if isinstance(energy_offset, dict):
            missing = [k for k in group_names if k not in energy_offset]
            if missing:
                raise KeyError(f"Missing energy_offset groups: {missing}")
            energy_input = [energy_offset[k] for k in group_names]
        else:
            energy_input = energy_offset
    else:
        group_names = None
        group_list = list(groups) if isinstance(groups, (list, tuple)) else [groups]
        energy_input = energy_offset

    energy_groups = _normalize_energy_structure(group_list, energy_input)

    norm_groups = []
    norm_energy_groups = []
    for g, ge in zip(group_list, energy_groups):
        if _is_tiledata_dict(g):
            norm_groups.append([g])
            norm_energy_groups.append([ge])
        else:
            norm_groups.append(list(g))
            if ge is None:
                ge = [None] * len(g)
            elif not isinstance(ge, (list, tuple)) or len(ge) != len(g):
                raise ValueError("energy_offset must mirror grouped input data")
            norm_energy_groups.append(list(ge))

    for g in norm_groups:
        if not g:
            raise ValueError("plot1D_grouped: empty group")
        for d in g:
            if not _is_tiledata_dict(d):
                raise TypeError("plot1D_grouped: group entries must be tiledata dicts")

    n_groups = len(norm_groups)
    if group_names is None:
        if key is not None:
            if len(key) != n_groups:
                raise ValueError("plot1D_grouped: key length mismatch")
            group_names = list(key)
        else:
            group_names = [f"group{i+1}" for i in range(n_groups)]
    else:
        if key is not None:
            if len(key) != n_groups:
                raise ValueError("plot1D_grouped: key length mismatch")
            group_names = list(key)

    reps = []
    rep_group = []
    rep_energy = []
    for gi, (g, ge) in enumerate(zip(norm_groups, norm_energy_groups)):
        for d, eo in zip(g, ge):
            reps.append(d)
            rep_group.append(gi)
            rep_energy.append(eo)

    if isinstance(tag, list):
        taglist = list(tag)
        stag = taglist[0].rstrip("0123456789")
    else:
        taglist = None
        stag = str(tag).rstrip("0123456789")

    vals = []
    for d in reps:
        if taglist is None:
            vals.append(np.asarray(d["comb"][tag], float).ravel())
        else:
            for t in taglist:
                vals.append(np.asarray(d["comb"][t], float).ravel())

    x = np.concatenate(vals)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("plot1D_grouped: no finite values for common range")

    xmin = float(np.min(x))
    xmax = float(np.max(x))
    pad = 1e-12 * (xmax - xmin + 1.0)
    rang = (xmin, xmax + pad)

    pmf_rep = []
    ranges_rep = []
    err_rep = []
    for d, eo in zip(reps, rep_energy):
        res = _pmf1d_from_tiledata(
            d,
            tag,
            usembar=usembar,
            kT=kbT,
            nbins=nbins,
            rang=rang,
            energy_offset=eo,
            energy_col=energy_col,
        )
        pmf_rep.append(res["pmf"])
        ranges_rep.append(res["ranges"])
        err_rep.append(res["dpmf"])

    n_rep = len(pmf_rep)

    base_rep = [0.0] * n_rep
    base_grp = [0.0] * n_groups
    if offset is None:
        pass
    elif isinstance(offset, (float, int)):
        base_rep = [float(offset)] * n_rep
    else:
        off = list(offset)
        if len(off) == n_rep:
            base_rep = [float(off[i]) for i in range(n_rep)]
        elif len(off) == n_groups:
            base_grp = [float(off[i]) for i in range(n_groups)]
        else:
            base_rep = [float(off[i]) if i < len(off) else 0.0 for i in range(n_rep)]

    for p, o in zip(pmf_rep, base_rep):
        p[stag] = p[stag] + float(o)

    extra = [0.0] * n_rep
    if matchflat is not None and len(matchflat) == 2:
        mmin, mmax = float(matchflat[0]), float(matchflat[1])

        means = []
        for p, r in zip(pmf_rep, ranges_rep):
            mask = r[stag].between(mmin, mmax, inclusive="both")
            m = p[stag][mask].mean()
            means.append(float(m) if pd.notna(m) else 0.0)

        if matchzero:
            extra = [-m for m in means]
        else:
            mmax_val = max(means) if means else 0.0
            extra = [mmax_val - m for m in means]

    for p, o in zip(pmf_rep, extra):
        p[stag] = p[stag] + float(o)

    pmf_grp = []
    ranges_grp = []
    err_grp = []
    for gi in range(n_groups):
        idxs = [i for i, g in enumerate(rep_group) if g == gi]
        pmfs = [pmf_rep[i] for i in idxs]
        rngs = [ranges_rep[i] for i in idxs]

        avg = average_pmf1d(pmfs, rngs, stag, method=average, kT=kbT)
        if base_grp[gi] != 0.0:
            avg["pmf"][stag] = avg["pmf"][stag] + float(base_grp[gi])

        pmf_grp.append(avg["pmf"])
        ranges_grp.append(avg["ranges"])
        err_grp.append(avg["dpmf"])

    if average_overlay:
        pmf_plot = []
        ranges_plot = []
        err_plot = []
        key_plot = []
        colors_plot = []

        for gi in range(n_groups):
            idxs = [i for i, g in enumerate(rep_group) if g == gi]
            col = colors[gi] if gi < len(colors) else (0.5, 0.5, 0.5)

            for i in idxs:
                pmf_plot.append(pmf_rep[i])
                ranges_plot.append(ranges_rep[i])
                err_plot.append(None)
                key_plot.append("_nolegend_")
                colors_plot.append(col)

            pmf_plot.append(pmf_grp[gi])
            ranges_plot.append(ranges_grp[gi])
            err_plot.append(err_grp[gi])
            key_plot.append(group_names[gi])
            colors_plot.append(col)

        dist1D(
            pmf_plot,
            ranges_plot,
            err=err_plot,
            size=size,
            markers=markers,
            tag=stag,
            minmax=minmax,
            tics=tics,
            label=label,
            fmin=fmin,
            fmax=fmax,
            colors=colors_plot,
            key=key_plot,
            vertical=vertical,
            horizontal=horizontal,
            save=save,
        )
        return

    dist1D(
        pmf_grp,
        ranges_grp,
        err=err_grp,
        size=size,
        markers=markers,
        tag=stag,
        minmax=minmax,
        tics=tics,
        label=label,
        fmin=fmin,
        fmax=fmax,
        colors=colors,
        key=group_names,
        vertical=vertical,
        horizontal=horizontal,
        save=save,
    )


def plot1D_combined(
    df,
    tag="dist",
    *,
    usembar=False,
    minmax=minmax,
    tics=tics,
    label=label,
    colors=colors1d,
    key=None,
    kbT=kb * T,
    nbins=50,
    fmin=0.0,
    fmax=20.0,
    size=1.5,
    markers=None,
    offset=None,
    matchflat=None,
    matchzero=False,
    average=None,
    average_overlay=False,
    average_key="avg",
    vertical=None,
    horizontal=None,
    energy_offset=None,
    energy_col=None,
    save=None,
):
    """Plot 1D PMFs with optional per-frame target-state reweighting."""
    if isinstance(df, dict) and "comb" not in df:
        avg_mode = average if average is not None else "boltzmann"
        plot1D_grouped(
            df,
            tag,
            average=avg_mode,
            usembar=usembar,
            minmax=minmax,
            tics=tics,
            label=label,
            colors=colors,
            key=key,
            kbT=kbT,
            nbins=nbins,
            fmin=fmin,
            fmax=fmax,
            size=size,
            markers=markers,
            offset=offset,
            matchflat=matchflat,
            matchzero=matchzero,
            average_overlay=average_overlay,
            vertical=vertical,
            horizontal=horizontal,
            energy_offset=energy_offset,
            energy_col=energy_col,
            save=save,
        )
        return

    top = list(df) if isinstance(df, (list, tuple)) else [df]
    energy_top = _normalize_energy_structure(top, energy_offset)

    groups: list[list[dict]] = []
    group_energy = []
    is_group: list[bool] = []
    for it, eo in zip(top, energy_top):
        if _is_tiledata_dict(it):
            groups.append([it])
            group_energy.append(eo)
            is_group.append(False)
            continue
        if isinstance(it, (list, tuple)):
            g = list(it)
            if not g:
                raise ValueError("plot1D_combined: empty group")
            for d in g:
                if not _is_tiledata_dict(d):
                    raise TypeError("plot1D_combined: group entries must be tiledata dicts")
            if eo is None:
                eo = [None] * len(g)
            elif not isinstance(eo, (list, tuple)) or len(eo) != len(g):
                raise ValueError("energy_offset must mirror grouped input data")
            groups.append(g)
            group_energy.append(list(eo))
            is_group.append(True)
            continue
        raise TypeError("plot1D_combined: df must be tiledata dicts or lists of them")

    has_groups = any(is_group)
    if has_groups:
        avg_mode = average if average is not None else "boltzmann"

        n_items = len(groups)
        if key is not None and len(key) != n_items:
            raise ValueError("plot1D_combined: key length must match top-level entries")

        if isinstance(tag, list):
            taglist = list(tag)
            stag = taglist[0].rstrip("0123456789")
        else:
            taglist = None
            stag = str(tag).rstrip("0123456789")

        reps: list[dict] = []
        rep_item: list[int] = []
        rep_energy = []
        for gi, (g, ge) in enumerate(zip(groups, group_energy)):
            if is_group[gi]:
                for d, eo in zip(g, ge):
                    reps.append(d)
                    rep_item.append(gi)
                    rep_energy.append(eo)
            else:
                reps.append(g[0])
                rep_item.append(gi)
                rep_energy.append(ge)

        vals = []
        for d in reps:
            if taglist is None:
                vals.append(np.asarray(d["comb"][tag], float).ravel())
            else:
                for t in taglist:
                    vals.append(np.asarray(d["comb"][t], float).ravel())

        x = np.concatenate(vals)
        x = x[np.isfinite(x)]
        if x.size == 0:
            raise ValueError("plot1D_combined: no finite values for common range")

        xmin = float(np.min(x))
        xmax = float(np.max(x))
        pad = 1e-12 * (xmax - xmin + 1.0)
        rang = (xmin, xmax + pad)

        pmf_rep = []
        ranges_rep = []
        err_rep = []
        for d, eo in zip(reps, rep_energy):
            res = _pmf1d_from_tiledata(
                d,
                tag,
                usembar=usembar,
                kT=kbT,
                nbins=nbins,
                rang=rang,
                energy_offset=eo,
                energy_col=energy_col,
            )
            pmf_rep.append(res["pmf"])
            ranges_rep.append(res["ranges"])
            err_rep.append(res["dpmf"])

        n_rep = len(pmf_rep)
        base_rep = [0.0] * n_rep
        if offset is None:
            pass
        elif isinstance(offset, (float, int)):
            base_rep = [float(offset)] * n_rep
        else:
            off = list(offset)
            if len(off) == n_items:
                base_rep = [float(off[rep_item[i]]) for i in range(n_rep)]
            elif len(off) == n_rep:
                base_rep = [float(off[i]) for i in range(n_rep)]
            else:
                base_rep = [float(off[i]) if i < len(off) else 0.0 for i in range(n_rep)]

        for p, o in zip(pmf_rep, base_rep):
            p[stag] = p[stag] + float(o)

        extra = [0.0] * n_rep
        if matchflat is not None and len(matchflat) == 2:
            mmin, mmax = float(matchflat[0]), float(matchflat[1])
            means = []
            for p, r in zip(pmf_rep, ranges_rep):
                mask = r[stag].between(mmin, mmax, inclusive="both")
                m = p[stag][mask].mean()
                means.append(float(m) if pd.notna(m) else 0.0)

            if matchzero:
                extra = [-m for m in means]
            else:
                mmax_val = max(means) if means else 0.0
                extra = [mmax_val - m for m in means]

        for p, o in zip(pmf_rep, extra):
            p[stag] = p[stag] + float(o)

        pmf_plot = []
        ranges_plot = []
        err_plot = []
        colors_plot = []
        key_plot = [] if key is not None else None

        for gi in range(n_items):
            idxs = [i for i, g in enumerate(rep_item) if g == gi]
            col = colors[gi] if gi < len(colors) else (0.5, 0.5, 0.5)

            if is_group[gi]:
                if average_overlay:
                    for ri in idxs:
                        pmf_plot.append(pmf_rep[ri])
                        ranges_plot.append(ranges_rep[ri])
                        err_plot.append(None)
                        colors_plot.append(col)
                        if key_plot is not None:
                            key_plot.append("_nolegend_")

                avg = average_pmf1d(
                    [pmf_rep[i] for i in idxs],
                    [ranges_rep[i] for i in idxs],
                    stag,
                    method=avg_mode,
                    kT=kbT,
                )
                pmf_plot.append(avg["pmf"])
                ranges_plot.append(avg["ranges"])
                err_plot.append(avg["dpmf"])
                colors_plot.append(col)
                if key_plot is not None:
                    key_plot.append(key[gi])
            else:
                ri = idxs[0]
                pmf_plot.append(pmf_rep[ri])
                ranges_plot.append(ranges_rep[ri])
                err_plot.append(err_rep[ri])
                colors_plot.append(col)
                if key_plot is not None:
                    key_plot.append(key[gi])

        dist1D(
            pmf_plot,
            ranges_plot,
            err=err_plot,
            size=size,
            markers=markers,
            tag=stag,
            minmax=minmax,
            tics=tics,
            label=label,
            fmin=fmin,
            fmax=fmax,
            colors=colors_plot,
            key=key_plot,
            vertical=vertical,
            horizontal=horizontal,
            save=save,
        )
        return

    pmf = []
    ranges = []
    err = []

    dflist = top
    energy_list = list(energy_top)

    if isinstance(tag, list):
        stag = tag[0].rstrip("0123456789")
    else:
        stag = str(tag).rstrip("0123456789")

    rang = None
    if average is not None:
        vals = []
        if isinstance(tag, list):
            for d in dflist:
                for t in tag:
                    vals.append(np.asarray(d["comb"][t], float).ravel())
        else:
            for d in dflist:
                vals.append(np.asarray(d["comb"][tag], float).ravel())
        x = np.concatenate(vals)
        x = x[np.isfinite(x)]
        if x.size == 0:
            raise ValueError("plot1D_combined: no finite values for common range")
        xmin = float(np.min(x))
        xmax = float(np.max(x))
        pad = 1e-12 * (xmax - xmin + 1.0)
        rang = (xmin, xmax + pad)

    for d, eo in zip(dflist, energy_list):
        res = _pmf1d_from_tiledata(
            d,
            tag,
            usembar=usembar,
            kT=kbT,
            nbins=nbins,
            rang=rang,
            energy_offset=eo,
            energy_col=energy_col,
        )
        pmf.append(res["pmf"])
        ranges.append(res["ranges"])
        err.append(res["dpmf"])

    n = len(pmf)

    if offset is None:
        base = [0.0] * n
    elif isinstance(offset, (float, int)):
        base = [float(offset)] * n
    else:
        base = [float(offset[i]) if i < len(offset) else 0.0 for i in range(n)]

    extra = [0.0] * n
    if matchflat is not None and len(matchflat) == 2:
        mmin, mmax = float(matchflat[0]), float(matchflat[1])

        means = []
        for p, r in zip(pmf, ranges):
            mask = r[stag].between(mmin, mmax, inclusive="both")
            m = p[stag][mask].mean()
            means.append(float(m) if pd.notna(m) else 0.0)

        if matchzero:
            extra = [-m for m in means]
        else:
            mmax_val = max(means) if means else 0.0
            extra = [mmax_val - m for m in means]

    total = [b + e for b, e in zip(base, extra)]
    for p, o in zip(pmf, total):
        p[stag] = p[stag] + float(o)

    if average is not None:
        avg = average_pmf1d(pmf, ranges, stag, method=average, kT=kbT)
        if average_overlay:
            pmf_plot = pmf + [avg["pmf"]]
            ranges_plot = ranges + [avg["ranges"]]
            err_plot = [None] * len(pmf) + [avg["dpmf"]]

            key_plot = None
            if key is not None:
                key_plot = list(key) + [average_key]
        else:
            pmf_plot = [avg["pmf"]]
            ranges_plot = [avg["ranges"]]
            err_plot = [avg["dpmf"]]

            key_plot = None
            if key is not None:
                key_plot = [average_key]
        colors_plot = colors
    else:
        pmf_plot = pmf
        ranges_plot = ranges
        err_plot = err
        key_plot = key
        colors_plot = colors

    dist1D(
        pmf_plot,
        ranges_plot,
        err=err_plot,
        size=size,
        markers=markers,
        tag=stag,
        minmax=minmax,
        tics=tics,
        label=label,
        fmin=fmin,
        fmax=fmax,
        colors=colors_plot,
        key=key_plot,
        vertical=vertical,
        horizontal=horizontal,
        save=save,
    )


def plot_series(
    s, *, title=None, xlabel=None, ylabel=None, logx=False, logy=False, save=None, size=1
):
    fig, ax = plt.subplots(figsize=(4 * size, 3 * size))
    ax.plot(s.index, s.values)
    ax.set_title(title or "")
    ax.set_xlabel(xlabel or s.index.name or "x")
    ax.set_ylabel(ylabel or s.name or "value")
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=300)
    plt.show()


def plot_hist_overlap(
    frames: dict[str, pd.DataFrame],
    col: str,
    *,
    wcol: str | None = None,
    keys: Iterable[str] | None = None,
    bins: int = 60,
    rang: tuple[float, float] | None = None,
    density: bool = True,
    cmap: str = "viridis",
    alpha: float = 0.35,
    lw: float = 1.3,
    figsize: tuple[float, float] = (14.0, 4.2),
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = True,
    legend_max: int = 18,
    sort_keys: bool = True,
    save: str | None = None,
) -> None:
    """
    Overlay per-dict-entry histograms for a given dataframe column.

    Parameters
    ----------
    frames : dict[str, DataFrame]
        Mapping like {"run_6.20": df, ...}. Non-DataFrame entries are ignored.
    col : str
        Column to histogram (e.g., "gxdist").
    wcol : str | None
        Optional per-sample weights column (e.g., "ww").
    keys : iterable[str] | None
        Subset/order of keys to plot. Defaults to all DataFrame keys.
    bins : int
        Number of bins.
    rang : (float, float) | None
        Histogram x-range. If None, computed from all data.
    density : bool
        Plot probability density (recommended for overlap checks).
    cmap : str
        Matplotlib colormap name.
    alpha : float
        Line fill alpha.
    lw : float
        Line width.
    figsize : (float, float)
        Wide figure size.
    legend : bool
        Show legend (auto-limited via legend_max).
    legend_max : int
        Max legend entries (avoids huge legends for 50 windows).
    sort_keys : bool
        Sort keys (lexicographic) if keys is None.
    save : str | None
        Save figure path if given.
    """
    df_keys = [k for k, v in frames.items() if isinstance(v, pd.DataFrame)]
    if keys is None:
        use_keys = sorted(df_keys) if sort_keys else list(df_keys)
    else:
        use_keys = [k for k in keys if k in frames and isinstance(frames[k], pd.DataFrame)]

    if not use_keys:
        raise ValueError("No DataFrame entries to plot.")

    xs: list[np.ndarray] = []
    ws: list[np.ndarray | None] = []
    for k in use_keys:
        s = pd.to_numeric(frames[k][col], errors="coerce").to_numpy()
        mask = np.isfinite(s)
        s = s[mask]
        if s.size == 0:
            continue
        xs.append(s)
        if wcol is None:
            ws.append(None)
        else:
            w = pd.to_numeric(frames[k][wcol], errors="coerce").to_numpy()
            w = w[mask]
            w = w[np.isfinite(w)]
            if w.size != s.size:
                w = None
            ws.append(w)

    if not xs:
        raise ValueError(f"No finite data found for column {col!r}.")

    if rang is None:
        xmin = min(float(np.min(a)) for a in xs)
        xmax = max(float(np.max(a)) for a in xs)
        pad = 1e-12 * (xmax - xmin + 1.0)
        rang = (xmin, xmax + pad)

    edges = np.linspace(rang[0], rang[1], bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=figsize, dpi=100, constrained_layout=True)
    cm = plt.get_cmap(cmap)
    n = len(xs)
    colors = [cm(i / max(n - 1, 1)) for i in range(n)]

    shown = 0
    for i, (k, x, w) in enumerate(zip(use_keys, xs, ws)):
        h, _ = np.histogram(x, bins=edges, weights=w, density=density)
        ax.plot(centers, h, color=colors[i], lw=lw, alpha=0.95, label=k)
        ax.fill_between(centers, 0.0, h, color=colors[i], alpha=alpha, linewidth=0.0)
        shown += 1

    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(xlabel or col)
    ax.set_ylabel(ylabel or ("Density" if density else "Count"))
    ax.set_xlim(rang[0], rang[1])
    ax.grid(True, linestyle="--", alpha=0.35)

    if legend and shown <= legend_max:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    if save:
        fig.savefig(save, dpi=300)
    plt.show()
