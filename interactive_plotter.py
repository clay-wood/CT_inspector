import os
from pathlib import Path
from typing import Iterable, Union, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm, ListedColormap
from ipywidgets import (
    IntSlider, FloatSlider, FloatText, Dropdown, Button, Text, ToggleButton, Label,
    HBox, VBox, Output
)
from IPython.display import display

# Optional SciPy for smooth 3D rotation
USE_SCIPY_ROTATE = True
try:
    from scipy.ndimage import rotate as nd_rotate
except Exception:
    USE_SCIPY_ROTATE = False
    nd_rotate = None

# Optional TIFF reader (your io_tiff.py)
try:
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
    x[0::2] = edges[:-1]
    x[1::2] = edges[1:]
    y[0::2] = counts
    y[1::2] = counts
    return x, y

# Simple nearest-neighbor 2D rotate (fallback)
def _rotate2d_nn(arr2d, deg):
    if abs(deg) < 1e-12:
        return arr2d
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    h, w = arr2d.shape
    yy, xx = np.mgrid[0:h, 0:w]
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

def rotate_volume_3d(vol: np.ndarray, deg_z: float, deg_y: float, deg_x: float) -> np.ndarray:
    out = vol
    eps = 1e-12
    if USE_SCIPY_ROTATE and nd_rotate is not None:
        if abs(deg_z) > eps:
            out = nd_rotate(out, angle=deg_z, axes=(1, 2), reshape=False, order=1, mode="nearest", prefilter=False)
        if abs(deg_y) > eps:
            out = nd_rotate(out, angle=deg_y, axes=(0, 2), reshape=False, order=1, mode="nearest", prefilter=False)
        if abs(deg_x) > eps:
            out = nd_rotate(out, angle=deg_x, axes=(0, 1), reshape=False, order=1, mode="nearest", prefilter=False)
        return out
    # Fallbacks (slice-wise nearest neighbor rotations)
    if abs(deg_z) > eps:
        out = np.stack([_rotate2d_nn(slc, deg_z) for slc in out], axis=0)  # rotate each XY slice
    if abs(deg_y) > eps:
        oy = out.shape[1]
        tmp = np.empty_like(out)
        for yy in range(oy):
            tmp[:, yy, :] = _rotate2d_nn(out[:, yy, :], deg_y)
        out = tmp
    if abs(deg_x) > eps:
        ox = out.shape[2]
        tmp = np.empty_like(out)
        for xx in range(ox):
            tmp[:, :, xx] = _rotate2d_nn(out[:, :, xx], deg_x)
        out = tmp
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

def interactive_slice_plotter(
    vol_or_src: Union[np.ndarray, str, Iterable[str]],
    *,
    voxel_size: float = 1.0,
    units: str = "px",
    lower_pct: float = 0.1,
    upper_pct: float = 99.9,
    interpolation: str = "antialiased",
    cmap_seq: str = "gray",
    cmap_div: str = "coolwarm",
    dpi: int = 110,
    display_container: bool = True,
    memmap: bool = False,
) -> Dict[str, object]:
    vol = _ensure_volume(vol_or_src, memmap=memmap)
    zlen, ylen, xlen = vol.shape

    vmin0, vmax0 = _robust_minmax(vol, lower_pct, upper_pct)
    if vmin0 < 0 < vmax0:
        norm0 = TwoSlopeNorm(vcenter=0.0, vmin=vmin0, vmax=vmax0)
        cmap0 = cmap_div
    else:
        norm0 = Normalize(vmin=vmin0, vmax=vmax0)
        cmap0 = cmap_seq
    norm_holder = {"norm": norm0}

    sZ, sY, sX = zlen // 2, ylen // 2, xlen // 2
    vx = vy = vz = float(voxel_size)
    extent_xy = (0, xlen * vx, 0, ylen * vy)
    extent_zy = (0, xlen * vx, 0, zlen * vz)
    extent_zx = (0, ylen * vy, 0, zlen * vz)

    rot_state = {"z": 0.0, "y": 0.0, "x": 0.0}
    rot_cache = {"vol": vol.copy(), "angles": (0.0, 0.0, 0.0)}

    out_fig = Output()
    with out_fig, plt.ioff():
        fig = plt.figure(figsize=(15, 5), dpi=dpi, constrained_layout=True)
        gs  = fig.add_gridspec(nrows=2, ncols=3, height_ratios=[3, 1])
        ax_xy = fig.add_subplot(gs[0, 0])
        ax_zy = fig.add_subplot(gs[0, 1])
        ax_zx = fig.add_subplot(gs[0, 2])
        ax_hxy = fig.add_subplot(gs[1, 0]); ax_hxy.set_visible(False)
        ax_hzy = fig.add_subplot(gs[1, 1]); ax_hzy.set_visible(False)
        ax_hzx = fig.add_subplot(gs[1, 2]); ax_hzx.set_visible(False)
        try:
            fig.canvas.header_visible = False
        except Exception:
            pass
        display(fig.canvas)

    im_xy = ax_xy.imshow(vol[sZ], cmap=cmap0, norm=norm0, origin="lower",
                         extent=extent_xy, interpolation=interpolation, aspect="equal")
    ax_xy.set_title("XY (Z slice)")
    ax_xy.set_xlabel(f"X [{units}]"); ax_xy.set_ylabel(f"Y [{units}]")

    im_zy = ax_zy.imshow(vol[:, :, sX], cmap=cmap0, norm=norm0, origin="lower",
                         extent=extent_zy, interpolation=interpolation, aspect="equal")
    ax_zy.set_title("ZY (X slice)")
    ax_zy.set_xlabel(f"X [{units}]"); ax_zy.set_ylabel(f"Z [{units}]")

    im_zx = ax_zx.imshow(vol[:, sY, :], cmap=cmap0, norm=norm0, origin="lower",
                         extent=extent_zx, interpolation=interpolation, aspect="equal")
    ax_zx.set_title("ZX (Y slice)")
    ax_zx.set_xlabel(f"Y [{units}]"); ax_zx.set_ylabel(f"Z [{units}]")

    (line_xy_y,) = ax_xy.plot([sX * vx, sX * vx], [0, ylen * vy], "w--", lw=1, alpha=0.9)
    (line_xy_x,) = ax_xy.plot([0, xlen * vx], [sY * vy, sY * vy], "w--", lw=1, alpha=0.9)
    (line_zy_x,) = ax_zy.plot([sX * vx, sX * vx], [0, zlen * vz], "w--", lw=1, alpha=0.9)
    (line_zy_z,) = ax_zy.plot([0, xlen * vx], [sZ * vz, sZ * vz], "w--", lw=1, alpha=0.9)
    (line_zx_y,) = ax_zx.plot([sY * vy, sY * vy], [0, zlen * vz], "w--", lw=1, alpha=0.9)
    (line_zx_z,) = ax_zx.plot([0, ylen * vy], [sZ * vz, sZ * vz], "w--", lw=1, alpha=0.9)

    cbar = fig.colorbar(im_xy, ax=[ax_xy, ax_zy, ax_zx], fraction=0.046, pad=0.04)
    cbar.set_label("Intensity")

    zslider = IntSlider(description="Z slice", min=0, max=zlen - 1, step=1, value=sZ, continuous_update=True)
    yslider = IntSlider(description="Y slice", min=0, max=ylen - 1, step=1, value=sY, continuous_update=True)
    xslider = IntSlider(description="X slice", min=0, max=xlen - 1, step=1, value=sX, continuous_update=True)

    min_box = FloatText(value=vmin0, description="Min:"); max_box = FloatText(value=vmax0, description="Max:")
    cmap_dd = Dropdown(options=[cmap_seq, "bone", "pink", "YlGnBu_r", "viridis", "inferno", cmap_div, "RdBu_r", "twilight", "twilight_shifted", "Spectral", "turbo"], value=cmap0, description="Colormap:")

    save_path = Text(value="", placeholder="e.g., figs/slices.png", description="Save to:")
    save_btn  = Button(description="Save", tooltip="Save current figure to the given path")
    hist_toggle = ToggleButton(value=False, description="Histograms", tooltip="Show/Hide histograms")
    trinarize_toggle = ToggleButton(value=False, description="Trinarize", tooltip="Toggle trinarized view")
    reset_btn = Button(description="Reset")

    rot_z = FloatSlider(description="Yaw (Z°)", min=-180.0, max=180.0, step=0.1, value=0.0, readout_format=".1f", continuous_update=False)
    rot_y = FloatSlider(description="Pitch (Y°)", min=-180.0, max=180.0, step=0.1, value=0.0, readout_format=".1f", continuous_update=False)
    rot_x = FloatSlider(description="Roll (X°)", min=-180.0, max=180.0, step=0.1, value=0.0, readout_format=".1f", continuous_update=False)
    rot_reset = Button(description="Reset Rotation")
    rot_label = Label(value="Rotate entire volume (0.1° resolution)")

    HIST_BINS = 256
    hist_lines = {"xy": None, "zy": None, "zx": None}

    trinarize_params = {"mu": None, "sigma": None, "lo": None, "hi": None}
    saved_norm_before_trin = {"norm": None}

    def _active_volume():
        a = (rot_state["z"], rot_state["y"], rot_state["x"])
        if a != rot_cache["angles"]:
            rot_cache["vol"] = rotate_volume_3d(vol, a[0], a[1], a[2])
            rot_cache["angles"] = a
        return rot_cache["vol"]

    def _slice_xy(z): return np.take(_active_volume(), indices=z, axis=0)
    def _slice_zy(y): return np.take(_active_volume(), indices=y, axis=1)
    def _slice_zx(x): return np.take(_active_volume(), indices=x, axis=2)

    def _update_hist_axis(axh, line, arr, vmin, vmax, title):
        counts, edges = _hist_counts(arr, vmin, vmax, HIST_BINS)
        xw, yw = _make_step_xy(counts, edges)
        if line is None:
            (line,) = axh.plot(xw, yw, lw=1.0)
        else:
            line.set_data(xw, yw)
        axh.set_title(title)
        axh.set_xlim(edges[0], edges[-1])
        axh.set_ylim(0, max(1, int(counts.max() * 1.05)))
        return line

    def _compute_trinarize_params():
        lo_r, hi_r = vmin0, vmax0
        vals = _finite(_active_volume())
        if vals.size > 2_000_000:
            idx = np.random.default_rng(1).choice(vals.size, size=2_000_000, replace=False)
            vals = vals[idx]
        counts, edges = np.histogram(vals, bins=256, range=(lo_r, hi_r))
        centers = 0.5 * (edges[:-1] + edges[1:])
        if counts.max() <= 0:
            mu = float(np.nanmedian(vals)) if vals.size else 0.0
            sigma = float(np.nanstd(vals)) if vals.size else 1.0
            return mu, sigma, mu - sigma, mu + sigma
        p = int(np.argmax(counts)); nbin = counts.size
        li = p
        while li > 0 and counts[li] > 0.5 * counts[p]: li -= 1
        ri = p
        while ri < nbin - 1 and counts[ri] > 0.5 * counts[p]: ri += 1
        mu = centers[p]; sigma = None
        mask = (centers >= centers[max(li, 0)]) & (centers <= centers[min(ri, nbin - 1)])
        try:
            ln_y = np.log(counts[mask] + 1e-12)
            a, b, _ = np.polyfit(centers[mask], ln_y, 2)
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

    def _trinarize_2d(arr2d):
        mu, sigma, lo_t, hi_t = trinarize_params["mu"], trinarize_params["sigma"], trinarize_params["lo"], trinarize_params["hi"]
        a = np.asarray(arr2d)
        out = np.full(a.shape, 0, dtype=np.int8)
        mask = np.isfinite(a)
        out[mask & (a < lo_t)] = -1
        out[mask & (a > hi_t)] = +1
        return out

    def update(*_):
        sliceZ, sliceY, sliceX = zslider.value, yslider.value, xslider.value
        vmin, vmax, cmap_name = float(min_box.value), float(max_box.value), cmap_dd.value

        if trinarize_toggle.value:
            im_xy.set_data(_trinarize_2d(_slice_xy(sliceZ)))
            im_zy.set_data(_trinarize_2d(_slice_zy(sliceY)))
            im_zx.set_data(_trinarize_2d(_slice_zx(sliceX)))
        else:
            im_xy.set_data(_slice_xy(sliceZ))
            im_zy.set_data(_slice_zy(sliceY))
            im_zx.set_data(_slice_zx(sliceX))

        line_xy_y.set_xdata([sliceX * vx, sliceX * vx]); line_xy_x.set_ydata([sliceY * vy, sliceY * vy])
        line_zy_x.set_xdata([sliceX * vx, sliceX * vx]); line_zy_z.set_ydata([sliceZ * vz, sliceZ * vz])
        line_zx_y.set_xdata([sliceY * vy, sliceY * vy]); line_zx_z.set_ydata([sliceZ * vz, sliceZ * vz])

        if not trinarize_toggle.value:
            if isinstance(norm_holder["norm"], TwoSlopeNorm):
                new_norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
            else:
                new_norm = Normalize(vmin=vmin, vmax=vmax)
            norm_holder["norm"] = new_norm
            im_xy.set_norm(new_norm); im_zy.set_norm(new_norm); im_zx.set_norm(new_norm)
            im_xy.set_cmap(cmap_name); im_zy.set_cmap(cmap_name); im_zx.set_cmap(cmap_name)
            cbar.update_normal(im_xy)

        if hist_toggle.value:
            hist_lines["xy"] = _update_hist_axis(ax_hxy, hist_lines["xy"], _slice_xy(sliceZ), vmin, vmax, "XY histogram")
            hist_lines["zy"] = _update_hist_axis(ax_hzy, hist_lines["zy"], _slice_zy(sliceY), vmin, vmax, "ZY histogram")
            hist_lines["zx"] = _update_hist_axis(ax_hzx, hist_lines["zx"], _slice_zx(sliceX), vmin, vmax, "ZX histogram")

        fig.canvas.draw_idle()

    def _set_xlim(ax, lim): ax.set_xlim(lim[0], lim[1])
    def _set_ylim(ax, lim): ax.set_ylim(lim[0], lim[1])
    sync_guard = {"on": False}
    def on_xy_limits_changed(_):
        if sync_guard["on"]: return
        sync_guard["on"] = True
        try:
            _set_xlim(ax_zy, ax_xy.get_xlim())
            _set_xlim(ax_zx, ax_xy.get_ylim())
        finally:
            sync_guard["on"] = False
    def on_zy_limits_changed(_):
        if sync_guard["on"]: return
        sync_guard["on"] = True
        try:
            _set_xlim(ax_xy, ax_zy.get_xlim())
            _set_ylim(ax_zx, ax_zy.get_ylim())
        finally:
            sync_guard["on"] = False
    def on_zx_limits_changed(_):
        if sync_guard["on"]: return
        sync_guard["on"] = True
        try:
            _set_ylim(ax_xy, ax_zx.get_xlim())
            _set_ylim(ax_zy, ax_zx.get_ylim())
        finally:
            sync_guard["on"] = False
    ax_xy.callbacks.connect("xlim_changed", on_xy_limits_changed)
    ax_xy.callbacks.connect("ylim_changed", on_xy_limits_changed)
    ax_zy.callbacks.connect("xlim_changed", on_zy_limits_changed)
    ax_zy.callbacks.connect("ylim_changed", on_zy_limits_changed)
    ax_zx.callbacks.connect("xlim_changed", on_zx_limits_changed)
    ax_zx.callbacks.connect("ylim_changed", on_zx_limits_changed)

    def on_save(_):
        path = save_path.value.strip()
        if path:
            Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=fig.dpi, bbox_inches="tight")

    def on_toggle_hist(change):
        show = bool(change["new"])
        ax_hxy.set_visible(show); ax_hzy.set_visible(show); ax_hzx.set_visible(show)
        if show:
            update()
        fig.canvas.draw_idle()

    def on_toggle_trinarize(change):
        if change["new"]:
            mu, sigma, lo_t, hi_t = _compute_trinarize_params()
            trinarize_params.update(mu=mu, sigma=sigma, lo=lo_t, hi=hi_t)
            saved_norm_before_trin["norm"] = norm_holder["norm"]
            trin_norm = Normalize(vmin=-1.0, vmax=+1.0)
            im_xy.set_norm(trin_norm); im_zy.set_norm(trin_norm); im_zx.set_norm(trin_norm)
            im_xy.set_cmap(ListedColormap(["#2c7bb6", "#f7f7f7", "#d7191c"]))
            im_zy.set_cmap(ListedColormap(["#2c7bb6", "#f7f7f7", "#d7191c"]))
            im_zx.set_cmap(ListedColormap(["#2c7bb6", "#f7f7f7", "#d7191c"]))
            cbar.update_normal(im_xy)
        else:
            old = saved_norm_before_trin["norm"]
            if old is not None:
                norm_holder["norm"] = old
                im_xy.set_norm(old); im_zy.set_norm(old); im_zx.set_norm(old)
                im_xy.set_cmap(cmap_dd.value); im_zy.set_cmap(cmap_dd.value); im_zx.set_cmap(cmap_dd.value)
                cbar.update_normal(im_xy)
        update()

    def on_reset(_):
        zslider.value, yslider.value, xslider.value = zlen // 2, ylen // 2, xlen // 2
        min_box.value, max_box.value = vmin0, vmax0
        cmap_dd.value = cmap0
        hist_toggle.value = False
        trinarize_toggle.value = False
        save_path.value = ""
        rot_z.value = 0.0; rot_y.value = 0.0; rot_x.value = 0.0

    def on_rot_change(_):
        rot_state["z"] = float(rot_z.value)
        rot_state["y"] = float(rot_y.value)
        rot_state["x"] = float(rot_x.value)
        update()

    def on_rot_reset(_):
        rot_z.value = 0.0; rot_y.value = 0.0; rot_x.value = 0.0

    save_btn.on_click(on_save)
    hist_toggle.observe(on_toggle_hist, names="value")
    trinarize_toggle.observe(on_toggle_trinarize, names="value")
    reset_btn.on_click(on_reset)
    # Wire up slice sliders + contrast + colormap
    zslider.observe(update, names="value")
    yslider.observe(update, names="value")
    xslider.observe(update, names="value")
    min_box.observe(update, names="value")
    max_box.observe(update, names="value")
    cmap_dd.observe(update, names="value")
    # Rotation events
    rot_z.observe(on_rot_change, names="value")
    rot_y.observe(on_rot_change, names="value")
    rot_x.observe(on_rot_change, names="value")
    rot_reset.on_click(on_rot_reset)

    controls_top = HBox([zslider, yslider, xslider])
    controls_mid = HBox([min_box, max_box, cmap_dd, hist_toggle, trinarize_toggle, reset_btn])
    controls_rot = HBox([rot_label, rot_z, rot_y, rot_x, rot_reset])
    controls_save = HBox([save_path, save_btn])

    container = VBox([controls_top, controls_mid, controls_rot, out_fig, controls_save])

    update()

    if display_container:
        display(container)

    return {
        "container": container,
        "fig": fig,
        "axes": dict(xy=ax_xy, zy=ax_zy, zx=ax_zx, hxy=ax_hxy, hzy=ax_hzy, hzx=ax_hzx),
        "widgets": dict(
            z=zslider, y=yslider, x=xslider, vmin=min_box, vmax=max_box,
            cmap=cmap_dd, save_path=save_path, save_btn=save_btn,
            hist_toggle=hist_toggle, trinarize_toggle=trinarize_toggle, reset=reset_btn,
            rot_z=rot_z, rot_y=rot_y, rot_x=rot_x, rot_reset=rot_reset
        ),
        "colorbar": cbar,
        "trinarize_params": trinarize_params,
        "rotation_state": rot_state,
    }