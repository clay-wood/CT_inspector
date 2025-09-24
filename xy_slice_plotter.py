from pathlib import Path
from typing import Iterable, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm, ListedColormap
from ipywidgets import (
    IntSlider, FloatText, Dropdown, Button, Text, ToggleButton, Label,
    HBox, VBox, Output
)
from IPython.display import display

# --- optional SciPy for high-quality rotation ---
USE_SCIPY_ROTATE = True
try:
    from scipy.ndimage import rotate as nd_rotate
except Exception:
    USE_SCIPY_ROTATE = False

# --- local TIFF I/O helper (matches your project style) ---
try:
    # If this file lives next to your io_tiff.py, this will work directly.
    # Otherwise, adjust the import path or pass a NumPy array directly.
    from io_tiff import read_volume as _read_volume
except Exception:
    _read_volume = None


def _finite(a):
    a = np.asarray(a)
    return a[np.isfinite(a)]


def _robust_minmax(a, lp=0.1, up=99.9):
    vals = _finite(a)
    if vals.size == 0:
        return 0.0, 1.0
    # subsample for speed on huge arrays
    if vals.size > 2_000_000:
        idx = np.random.default_rng(0).choice(vals.size, size=2_000_000, replace=False)
        vals = vals[idx]
    return float(np.percentile(vals, lp)), float(np.percentile(vals, up))


def _hist_counts(vals, vmin, vmax, bins):
    vals = _finite(vals)
    if vals.size == 0 or not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        edges = np.linspace(0 if not np.isfinite(vmin) else vmin,
                            1 if not np.isfinite(vmax) else vmax,
                            bins + 1)
        return np.zeros(bins, dtype=int), edges
    return np.histogram(vals, bins=bins, range=(vmin, vmax))


def _make_step_xy(counts, edges):
    x = np.empty(2 * len(counts))
    y = np.empty(2 * len(counts))
    x[0::2] = edges[:-1]; x[1::2] = edges[1:]
    y[0::2] = counts;     y[1::2] = counts
    return x, y


def _compute_trinarize_params(volume, lo_ref, hi_ref):
    """Estimate mode +/- sigma from the (clipped) histogram of the whole volume."""
    vals = _finite(volume)
    if vals.size > 2_000_000:
        idx = np.random.default_rng(1).choice(vals.size, size=2_000_000, replace=False)
        vals = vals[idx]
    counts, edges = np.histogram(vals, bins=256, range=(lo_ref, hi_ref))
    centers = 0.5 * (edges[:-1] + edges[1:])
    if counts.max() <= 0:
        mu = float(np.nanmedian(vals)) if vals.size else 0.0
        sigma = float(np.nanstd(vals)) if vals.size else 1.0
        return mu, sigma, mu - sigma, mu + sigma

    p = int(np.argmax(counts))
    thr = 0.5 * counts[p]
    li = p
    while li > 0 and counts[li] >= thr:
        li -= 1
    ri = p
    nbin = counts.size
    while ri < nbin - 1 and counts[ri] >= thr:
        ri += 1
    li = max(li, 0); ri = min(ri, nbin - 1)
    if ri - li < 4:
        li = max(p - 2, 0); ri = min(p + 2, nbin - 1)

    xw = centers[li:ri+1]
    yw = counts[li:ri+1].astype(float)
    mask = yw > 0
    mu = centers[p]
    sigma = None
    if mask.sum() >= 3:
        try:
            ln_y = np.log(yw[mask])
            a, b, _ = np.polyfit(xw[mask], ln_y, 2)
            if a < 0:
                mu = -b / (2 * a)
                sigma = np.sqrt(-1 / (2 * a))
        except Exception:
            sigma = None
    if sigma is None or not np.isfinite(sigma):
        x_left = centers[max(li, 0)]
        x_right = centers[min(ri, nbin - 1)]
        width_hp = max(1e-12, (x_right - x_left))
        sigma = width_hp / (2 * np.sqrt(np.log(2.0)))
    lo_t, hi_t = mu - sigma, mu + sigma
    return float(mu), float(sigma), float(lo_t), float(hi_t)


def _trinarize_2d(arr2d, lo_t, hi_t):
    a = np.asarray(arr2d)
    out = np.full(a.shape, 0, dtype=np.int8)
    m = np.isfinite(a)
    out[m & (a < lo_t)] = -1
    out[m & (a > hi_t)] = +1
    return out


def _rotate2d(arr2d, deg):
    if abs(deg) < 1e-12:
        return arr2d
    if USE_SCIPY_ROTATE:
        # order=1 (bilinear) is a good quality/speed compromise; reshape=False keeps the same shape
        return nd_rotate(arr2d, angle=deg, reshape=False, order=1, mode="nearest", prefilter=False)
    # Simple near-neighbor fallback if SciPy not available
    # NOTE: this is a minimalist fallback; for production prefer SciPy.
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    h, w = arr2d.shape
    yy, xx = np.mgrid[0:h, 0:w]
    # center grid at image center
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    x0 = xx - cx
    y0 = yy - cy
    # inverse rotation mapping
    xr = c * x0 + s * y0
    yr = -s * x0 + c * y0
    xi = np.rint(xr + cx).astype(int)
    yi = np.rint(yr + cy).astype(int)
    out = np.full_like(arr2d, np.nan)
    m = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    out[m] = arr2d[yi[m], xi[m]]
    return out


def _ensure_volume(vol_or_src: Union[np.ndarray, str, Iterable[str]], *, memmap=False) -> np.ndarray:
    if isinstance(vol_or_src, np.ndarray):
        vol = vol_or_src
    else:
        if _read_volume is None:
            raise RuntimeError("io_tiff.read_volume not available; pass a NumPy volume instead.")
        vol = _read_volume(vol_or_src, memmap=memmap)
    if vol.ndim != 3:
        raise ValueError("Expected volume shape (Z, Y, X).")
    return vol


def create_interactive_xy_slice_plotter(
    vol_or_src: Union[np.ndarray, str, Iterable[str]],
    *,
    voxel_size: float = 1.0,
    units: str = "px",
    lower_pct: float = 0.1,
    upper_pct: float = 99.9,
    interpolation: str = "antialiased",   # crisp: "none"
    dpi: int = 110,
    memmap: bool = False,
    display_container: bool = True,
):
    """
    Interactive XY slice viewer with:
      - Z slider
      - Min/Max intensity boxes
      - Colormap presets (div & seq)
      - Histogram toggle (for the current slice)
      - Trinarize toggle (mode±sigma -> {-1,0,1})
      - Save filename + button
      - Reset button
      - Rotation entry (rounded to 0.1°) applied to the XY slice

    Returns: dict with "container", "fig", "axes", "widgets".
    """
    vol = _ensure_volume(vol_or_src, memmap=memmap)
    zlen, ylen, xlen = vol.shape

    # Initial robust min/max from the whole volume
    vmin0, vmax0 = _robust_minmax(vol, lower_pct, upper_pct)

    # Choose norm/cmap based on sign
    cmap_div = "RdBu_r"
    cmap_seq = "Greys_r"
    if vmin0 < 0 < vmax0:
        norm0 = TwoSlopeNorm(vcenter=0.0, vmin=vmin0, vmax=vmax0)
        cmap0 = cmap_div
    else:
        norm0 = Normalize(vmin=vmin0, vmax=vmax0)
        cmap0 = cmap_seq
    norm_holder = {"norm": norm0}

    # Widgets
    zslider  = IntSlider(description="Z:", min=0, max=zlen - 1, step=1, value=zlen // 2, continuous_update=True)
    min_box  = FloatText(value=vmin0, description="Min:")
    max_box  = FloatText(value=vmax0, description="Max:")
    cmap_dd  = Dropdown(options=[cmap_div, "inferno", cmap_seq], value=cmap0, description="Colormap:")
    hist_tog = ToggleButton(value=False, description="Histogram", tooltip="Show/Hide histogram")
    trin_tog = ToggleButton(value=False, description="Trinarize", tooltip="Show {-1,0,1} using mode±sigma")
    rot_box  = FloatText(value=0.0, step=0.1, description="Rotate (°):", tooltip="Positive is CCW; rounded to 0.1°")
    save_to  = Text(value="", placeholder="e.g., figs/xy_slice.png", description="Save to:")
    save_btn = Button(description="Save")
    reset_btn= Button(description="Reset")

    # Figure & axes
    out_fig = Output()
    with out_fig, plt.ioff():
        # 2 rows if histogram visible later; start compact
        fig = plt.figure(figsize=(8.5, 6.0), dpi=dpi, constrained_layout=True)
        gs  = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[4, 1])
        ax_img = fig.add_subplot(gs[0, 0])
        ax_hist= fig.add_subplot(gs[1, 0]); ax_hist.set_visible(False)

        try:
            fig.canvas.header_visible = False
        except Exception:
            pass
        display(fig.canvas)

    base_w, base_h = fig.get_size_inches()

    vx = vy = voxel_size
    extent_xy = (0, xlen * vx, 0, ylen * vy)

    # State for trinarize thresholds (computed on-demand from the volume)
    tri_params = {"mu": None, "sigma": None, "lo": None, "hi": None}
    saved_norm = {"norm": None}  # to restore after leaving trinarize mode

    # Histogram line handle
    HIST_BINS = 256
    hist_line = None

    # Initial image
    def _get_slice(z):
        return np.take(vol, indices=z, axis=0)

    im = ax_img.imshow(
        _get_slice(zslider.value), origin="lower", interpolation=interpolation,
        extent=extent_xy, cmap=cmap0, norm=norm_holder["norm"]
    )
    ax_img.set_xlabel(f"X [{units}]")
    ax_img.set_ylabel(f"Y [{units}]")
    cbar = fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)

    # --- helpers ---

    def _update_hist(data2d, vmin, vmax):
        nonlocal hist_line
        if not hist_tog.value:
            return
        counts, edges = _hist_counts(data2d.ravel(), vmin, vmax, HIST_BINS)
        x, y = _make_step_xy(counts, edges)
        if hist_line is None:
            (hist_line,) = ax_hist.plot(x, y, lw=1.5)
            ax_hist.set_title("XY histogram"); ax_hist.set_ylabel("Count")
        else:
            hist_line.set_data(x, y)
        ax_hist.set_xlim(edges[0], edges[-1])
        ax_hist.set_ylim(0, max(1, int(counts.max() * 1.05)))

    def _maybe_compute_trinarize():
        if tri_params["lo"] is None or tri_params["hi"] is None:
            mu, sigma, lo_t, hi_t = _compute_trinarize_params(vol, vmin0, vmax0)
            tri_params.update(mu=mu, sigma=sigma, lo=lo_t, hi=hi_t)

    def _round_deg_to_point1(value):
        # Clamp to [-180, 180] and round to nearest 0.1
        if not np.isfinite(value):
            return 0.0
        v = max(-180.0, min(180.0, float(value)))
        return np.round(v * 10.0) / 10.0

    def _current_image_data():
        z = int(zslider.value)
        sl = _get_slice(z)
        deg = _round_deg_to_point1(rot_box.value)
        if abs(deg) >= 1e-12:
            sl = _rotate2d(sl, deg)
        return sl

    def update(*_):
        vmin = float(min_box.value); vmax = float(max_box.value)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin, vmax = vmin0, vmax0
            min_box.value, max_box.value = vmin, vmax

        arr = _current_image_data()

        if trin_tog.value:
            _maybe_compute_trinarize()
            arr_tri = _trinarize_2d(arr, tri_params["lo"], tri_params["hi"])
            im.set_data(arr_tri)
            if not isinstance(im.norm, Normalize) or im.norm.vmin != -1.0 or im.norm.vmax != 1.0:
                saved_norm["norm"] = norm_holder["norm"]
                im.set_norm(Normalize(vmin=-1.0, vmax=+1.0))
                cbar.update_normal(im)
            # Discrete 3-color scheme mapped to {-1,0,1}
            im.set_cmap(ListedColormap(["#2c7bb6", "#f7f7f7", "#d7191c"]))
            _update_hist(arr, vmin, vmax)  # hist of underlying intensities
        else:
            im.set_cmap(cmap_dd.value)
            # Restore norm and colorbar if needed
            if saved_norm["norm"] is not None and im.norm.vmin == -1.0 and im.norm.vmax == 1.0:
                norm_holder["norm"] = saved_norm["norm"]
                saved_norm["norm"] = None
                im.set_norm(norm_holder["norm"])
                cbar.update_normal(im)
            # Update norm bounds
            if isinstance(norm_holder["norm"], TwoSlopeNorm):
                im.set_norm(TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax))
                norm_holder["norm"] = im.norm
            else:
                im.set_norm(Normalize(vmin=vmin, vmax=vmax))
                norm_holder["norm"] = im.norm
            im.set_data(arr)
            _update_hist(arr, vmin, vmax)

        fig.canvas.draw_idle()

    # --- callbacks ---
    def on_toggle_hist(change):
        show = bool(change["new"])
        ax_hist.set_visible(show)
        if show:
            fig.set_size_inches(base_w, max(base_h, base_h + max(2.6, 0.45 * base_h)))
        else:
            fig.set_size_inches(base_w, base_h)
        update()

    def on_toggle_trinarize(change):
        if change["new"]:
            _maybe_compute_trinarize()
        update()

    def on_rotate_change(change):
        # enforce 0.1° increments
        v = _round_deg_to_point1(change["new"])
        if v != change["new"]:
            rot_box.value = v
            return  # value change will trigger update again
        update()

    def on_save(_):
        path = save_to.value.strip() or "xy_slice.png"
        p = Path(path)
        if p.suffix == "":
            p = p.with_suffix(".png")
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), dpi=dpi, bbox_inches="tight")
        save_to.value = str(p)  # echo resolved path

    def on_reset(_):
        zslider.value = zlen // 2
        min_box.value, max_box.value = vmin0, vmax0
        cmap_dd.value = cmap0
        hist_tog.value = False
        trin_tog.value = False
        rot_box.value = 0.0
        save_to.value = ""

    # Wire up
    zslider.observe(update, names="value")
    for w in (min_box, max_box, cmap_dd):
        w.observe(update, names="value")
    hist_tog.observe(on_toggle_hist, names="value")
    trin_tog.observe(on_toggle_trinarize, names="value")
    rot_box.observe(on_rotate_change, names="value")
    save_btn.on_click(on_save)
    reset_btn.on_click(on_reset)

    # Initial draw
    update()

    # Layout
    controls = VBox([
        HBox([zslider, rot_box]),
        HBox([min_box, max_box, cmap_dd]),
        HBox([hist_tog, trin_tog, Label(" "), save_to, save_btn, reset_btn]),
    ])
    container = VBox([out_fig, controls])
    if display_container:
        display(container)

    # return {
    #     "container": container,
    #     "fig": fig,
    #     "axes": dict(img=ax_img, hist=ax_hist),
    #     "widgets": dict(
    #         z=zslider, vmin=min_box, vmax=max_box, cmap=cmap_dd,
    #         histogram=hist_tog, trinarize=trin_tog, rotate=rot_box,
    #         save_path=save_to, save_btn=save_btn, reset=reset_btn
    #     ),
    #     "colorbar": cbar,
    #     "trinarize_params": tri_params,
    # }