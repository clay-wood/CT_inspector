import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from ipywidgets import (
    IntSlider, FloatText, Dropdown, Button, Text, ToggleButton,
    HBox, VBox, Output
)
from IPython.display import display


def create_interactive_slice_plotter(
    vol: np.ndarray,
    *,
    voxel_size: float = 1.0,          # dz = dy = dx
    units: str = "px",
    cmap_seq: str = "Greys_r",
    cmap_div: str = "RdBu_r",
    lower_pct: float = 0.1,
    upper_pct: float = 99.9,
    dpi: int = 100,
    display_container: bool = True,
):
    """Interactive three-view slice viewer with histogram and trinarize toggles.
    Volume shape must be (Z, Y, X). NaNs are allowed."""

    assert vol.ndim == 3, "vol must be 3D (Z, Y, X)"
    zlen, ylen, xlen = vol.shape

    # ----- robust min/max (NaN-safe; subsample if huge) -----
    def robust_minmax(a, lp=lower_pct, up=upper_pct):
        a = np.asarray(a)
        finite = a[np.isfinite(a)]
        if finite.size == 0:
            return 0.0, 1.0
        if finite.size > 2_000_000:
            idx = np.random.default_rng(0).choice(finite.size, size=2_000_000, replace=False)
            finite = finite[idx]
        return float(np.percentile(finite, lp)), float(np.percentile(finite, up))

    sZ, sY, sX = zlen // 2, ylen // 2, xlen // 2
    vmin0, vmax0 = robust_minmax(vol)
    if vmin0 < 0 < vmax0:
        norm0 = TwoSlopeNorm(vcenter=0.0, vmin=vmin0, vmax=vmax0)
        cmap0 = cmap_div
    else:
        norm0 = Normalize(vmin=vmin0, vmax=vmax0)
        cmap0 = cmap_seq
    norm_holder = {"norm": norm0}

    # ----- lazy slicing -----
    def slice_xy(z): return np.take(vol, indices=z, axis=0)
    def slice_zy(y): return np.take(vol, indices=y, axis=1)
    def slice_zx(x): return np.take(vol, indices=x, axis=2)

    # ----- figure inside Output to avoid static duplicate -----
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

    base_w, base_h = fig.get_size_inches()
    vx = vy = vz = voxel_size
    extent_xy = (0, xlen*vx, 0, ylen*vy)
    extent_zy = (0, xlen*vx, 0, zlen*vz)
    extent_zx = (0, ylen*vy, 0, zlen*vz)

    # ----- initial images -----
    im_xy = ax_xy.imshow(slice_xy(sZ), origin="lower", interpolation="none",
                         extent=extent_xy, cmap=cmap0, norm=norm_holder["norm"])
    im_zy = ax_zy.imshow(slice_zy(sY), origin="lower", interpolation="none",
                         extent=extent_zy, cmap=cmap0, norm=norm_holder["norm"])
    im_zx = ax_zx.imshow(slice_zx(sX), origin="lower", interpolation="none",
                         extent=extent_zx, cmap=cmap0, norm=norm_holder["norm"])

    # crosshairs
    line_xy_y = ax_xy.axvline(sX*vx, c="k", alpha=0.25, ls="--")
    line_xy_x = ax_xy.axhline(sY*vy, c="k", alpha=0.25, ls="--")
    line_zy_x = ax_zy.axvline(sX*vx, c="k", alpha=0.25, ls="--")
    line_zy_z = ax_zy.axhline(sZ*vz, c="k", alpha=0.25, ls="--")
    line_zx_y = ax_zx.axvline(sY*vy, c="k", alpha=0.25, ls="--")
    line_zx_z = ax_zx.axhline(sZ*vz, c="k", alpha=0.25, ls="--")

    # labels & colorbar
    ax_xy.set_xlabel(f"X [{units}]"); ax_xy.set_ylabel(f"Y [{units}]")
    ax_zy.set_xlabel(f"X [{units}]"); ax_zy.set_ylabel(f"Z [{units}]")
    ax_zx.set_xlabel(f"Y [{units}]"); ax_zx.set_ylabel(f"Z [{units}]")
    cbar = fig.colorbar(im_xy, ax=[ax_xy, ax_zy, ax_zx], fraction=0.046, pad=0.04)
    cbar.set_label("Intensity")

    # ----- widgets (controls below) -----
    zslider = IntSlider(description="Z slice", min=0, max=zlen-1, step=1, value=sZ, continuous_update=True)
    yslider = IntSlider(description="Y slice", min=0, max=ylen-1, step=1, value=sY, continuous_update=True)
    xslider = IntSlider(description="X slice", min=0, max=xlen-1, step=1, value=sX, continuous_update=True)

    min_box = FloatText(value=vmin0, description="Min:")
    max_box = FloatText(value=vmax0, description="Max:")
    cmap_dd = Dropdown(options=[cmap_div, "inferno", cmap_seq], value=cmap0, description="Colormap:")

    # UPDATED save UI
    save_path = Text(value="", placeholder="e.g., figs/slices.png", description="Save to:")
    save_btn  = Button(description="Save", tooltip="Save current figure to the given path")

    hist_toggle = ToggleButton(value=False, description="Histograms", tooltip="Show/Hide histograms")
    trinarize_toggle = ToggleButton(value=False, description="Trinarize", tooltip="Toggle trinarized view")
    reset_btn = Button(description="Reset")

    state0 = dict(sZ=sZ, sY=sY, sX=sX, vmin=vmin0, vmax=vmax0, cmap=cmap0)

    # ----- histograms (NaN-safe) -----
    HIST_BINS = 256
    hist_lines = {"xy": None, "zy": None, "zx": None}

    def _finite(vals):
        vals = np.asarray(vals); return vals[np.isfinite(vals)]

    def _hist_counts(vals, vmin, vmax, bins):
        vals = _finite(vals)
        if vals.size == 0 or not np.isfinite(vmin) or not np.isfinite(vmax) or (vmin >= vmax):
            edges = np.linspace(0 if not np.isfinite(vmin) else vmin,
                                1 if not np.isfinite(vmax) else vmax,
                                bins + 1)
            return np.zeros(bins, dtype=int), edges
        return np.histogram(vals, bins=bins, range=(vmin, vmax))

    def _make_step_xy(counts, edges):
        x = np.empty(2*len(counts)); y = np.empty(2*len(counts))
        x[0::2] = edges[:-1]; x[1::2] = edges[1:]
        y[0::2] = counts;     y[1::2] = counts
        return x, y

    def _update_hist_axis(axh, line, data2d, vmin, vmax, title):
        counts, edges = _hist_counts(data2d.ravel(), vmin, vmax, HIST_BINS)
        x, y = _make_step_xy(counts, edges)
        if line is None:
            ln, = axh.plot(x, y, lw=1.5, color="blue")
            axh.set_title(title); axh.set_ylabel("Count")
        else:
            ln = line; ln.set_data(x, y)
        axh.set_xlim(edges[0], edges[-1])
        axh.set_ylim(0, max(1, int(counts.max()*1.05)))
        return ln

    def _update_all_histograms(sliceZ, sliceY, sliceX, vmin, vmax):
        if not hist_toggle.value:
            return
        hist_lines["xy"] = _update_hist_axis(ax_hxy, hist_lines["xy"], slice_xy(sliceZ), vmin, vmax, "XY histogram")
        hist_lines["zy"] = _update_hist_axis(ax_hzy, hist_lines["zy"], slice_zy(sliceY), vmin, vmax, "ZY histogram")
        hist_lines["zx"] = _update_hist_axis(ax_hzx, hist_lines["zx"], slice_zx(sliceX), vmin, vmax, "ZX histogram")

    # ----- trinarization thresholds (computed on-demand) -----
    trinarize_params = {"mu": None, "sigma": None, "lo": None, "hi": None}
    saved_norm_before_trin = {"norm": None}

    def _compute_trinarize_params():
        lo_r, hi_r = vmin0, vmax0
        vals = _finite(vol)
        if vals.size > 2_000_000:
            idx = np.random.default_rng(1).choice(vals.size, size=2_000_000, replace=False)
            vals = vals[idx]
        counts, edges = np.histogram(vals, bins=256, range=(lo_r, hi_r))
        centers = 0.5*(edges[:-1] + edges[1:])
        if counts.max() <= 0:
            mu = float(np.nanmedian(vals)) if vals.size else 0.0
            sigma = float(np.nanstd(vals)) if vals.size else 1.0
            return mu, sigma, mu - sigma, mu + sigma

        p = int(np.argmax(counts))
        peak_c = counts[p]
        thr = peak_c / np.sqrt(2.0)
        li = p
        while li > 0 and counts[li] >= thr:
            li -= 1
        ri = p
        nbin = counts.size
        while ri < nbin-1 and counts[ri] >= thr:
            ri += 1
        li = max(li, 0); ri = min(ri, nbin-1)
        if ri - li < 4:
            li = max(p-2, 0); ri = min(p+2, nbin-1)

        sel = slice(li, ri+1)
        xw = centers[sel]
        yw = counts[sel].astype(float)
        mask = yw > 0
        mu = centers[p]
        sigma = None
        if mask.sum() >= 3:
            ln_y = np.log(yw[mask])
            xw_m = xw[mask]
            try:
                a, b, _ = np.polyfit(xw_m, ln_y, 2)
                if a < 0:
                    mu = -b / (2*a)
                    sigma = np.sqrt(-1 / (2*a))
            except Exception:
                sigma = None
        if sigma is None or not np.isfinite(sigma):
            x_left = centers[max(li, 0)]
            x_right = centers[min(ri, nbin-1)]
            width_hp = max(1e-12, (x_right - x_left))
            sigma = width_hp / (2*np.sqrt(np.log(2.0)))
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

    # ----- main update -----
    def update():
        sliceZ, sliceY, sliceX = zslider.value, yslider.value, xslider.value
        vmin, vmax, cmap_name = float(min_box.value), float(max_box.value), cmap_dd.value

        if trinarize_toggle.value:
            im_xy.set_data(_trinarize_2d(slice_xy(sliceZ)))
            im_zy.set_data(_trinarize_2d(slice_zy(sliceY)))
            im_zx.set_data(_trinarize_2d(slice_zx(sliceX)))
        else:
            im_xy.set_data(slice_xy(sliceZ))
            im_zy.set_data(slice_zy(sliceY))
            im_zx.set_data(slice_zx(sliceX))

        line_xy_y.set_xdata([sliceX*vx, sliceX*vx]); line_xy_x.set_ydata([sliceY*vy, sliceY*vy])
        line_zy_x.set_xdata([sliceX*vx, sliceX*vx]); line_zy_z.set_ydata([sliceZ*vz, sliceZ*vz])
        line_zx_y.set_xdata([sliceY*vy, sliceY*vy]); line_zx_z.set_ydata([sliceZ*vz, sliceZ*vz])

        if not trinarize_toggle.value:
            if cmap_name != im_xy.get_cmap().name:
                im_xy.set_cmap(cmap_name); im_zy.set_cmap(cmap_name); im_zx.set_cmap(cmap_name)

            cur = norm_holder["norm"]
            want_div = (vmin < 0 < vmax); have_div = isinstance(cur, TwoSlopeNorm)
            if want_div and have_div:
                cur.vmin, cur.vmax, cur.vcenter = vmin, vmax, 0.0
            elif (not want_div) and (not have_div):
                cur.vmin, cur.vmax = vmin, vmax
            elif want_div and not have_div:
                new_norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
                im_xy.set_norm(new_norm); im_zy.set_norm(new_norm); im_zx.set_norm(new_norm)
                norm_holder["norm"] = new_norm
            else:
                new_norm = Normalize(vmin=vmin, vmax=vmax)
                im_xy.set_norm(new_norm); im_zy.set_norm(new_norm); im_zx.set_norm(new_norm)
                norm_holder["norm"] = new_norm

        cbar.update_normal(im_xy)
        _update_all_histograms(sliceZ, sliceY, sliceX, vmin, vmax)
        fig.canvas.draw_idle()

    # ----- linked zoom/pan -----
    sync_guard = {"on": False}
    def _set_xlim(ax, xlim): ax.set_xlim(xlim)
    def _set_ylim(ax, ylim): ax.set_ylim(ylim)

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

    # ----- controls -----
    def on_save(_):
        path = save_path.value.strip()
        if not path:
            path = "slices.png"
        p = Path(path)
        if p.suffix == "":
            p = p.with_suffix(".png")
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), dpi=dpi, bbox_inches="tight")
        # Reflect any normalization state (continuous or trinarized) automatically
        save_path.value = str(p)

    def on_toggle_hist(change):
        show = bool(change["new"])
        ax_hxy.set_visible(show); ax_hzy.set_visible(show); ax_hzx.set_visible(show)
        if show:
            fig.set_size_inches(base_w, max(base_h, base_h + max(3.0, 0.6*base_h)))
            _update_all_histograms(zslider.value, yslider.value, xslider.value,
                                   float(min_box.value), float(max_box.value))
        else:
            fig.set_size_inches(base_w, base_h)
        fig.canvas.draw_idle()

    def on_toggle_trinarize(change):
        if change["new"]:
            mu, sigma, lo_t, hi_t = _compute_trinarize_params()
            trinarize_params.update(mu=mu, sigma=sigma, lo=lo_t, hi=hi_t)
            saved_norm_before_trin["norm"] = norm_holder["norm"]
            trin_norm = Normalize(vmin=-1.0, vmax=+1.0)
            im_xy.set_norm(trin_norm); im_zy.set_norm(trin_norm); im_zx.set_norm(trin_norm)
            cbar.update_normal(im_xy)
        else:
            old = saved_norm_before_trin["norm"]
            if old is not None:
                im_xy.set_norm(old); im_zy.set_norm(old); im_zx.set_norm(old)
                norm_holder["norm"] = old
                cbar.update_normal(im_xy)
        update()

    def on_reset(_):
        zslider.value, yslider.value, xslider.value = state0["sZ"], state0["sY"], state0["sX"]
        min_box.value, max_box.value = state0["vmin"], state0["vmax"]
        cmap_dd.value = state0["cmap"]
        hist_toggle.value = False
        trinarize_toggle.value = False
        save_path.value = ""

    save_btn.on_click(on_save)
    hist_toggle.observe(on_toggle_hist, names="value")
    trinarize_toggle.observe(on_toggle_trinarize, names="value")
    reset_btn.on_click(on_reset)

    def _trigger(_=None): update()
    for w in (zslider, yslider, xslider, min_box, max_box, cmap_dd):
        w.observe(_trigger, names="value")

    update()

    # ----- layout: figure on top, controls below -----
    controls = VBox([
        HBox([zslider, yslider, xslider]),
        HBox([min_box, max_box, cmap_dd]),
        HBox([save_path, save_btn, hist_toggle, trinarize_toggle, reset_btn]),
    ])
    container = VBox([out_fig, controls])

    if display_container:
        display(container)

    return {
        "container": container,
        "fig": fig,
        "axes": dict(xy=ax_xy, zy=ax_zy, zx=ax_zx, hxy=ax_hxy, hzy=ax_hzy, hzx=ax_hzx),
        "widgets": dict(
            z=zslider, y=yslider, x=xslider, vmin=min_box, vmax=max_box,
            cmap=cmap_dd, save_path=save_path, save_btn=save_btn,
            hist_toggle=hist_toggle, trinarize_toggle=trinarize_toggle, reset=reset_btn
        ),
        "colorbar": cbar,
        "trinarize_params": trinarize_params,
    }