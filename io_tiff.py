import os
import re
import glob
from typing import Iterable, Tuple, Optional, Dict, Any, Union, List

import numpy as np
import tifffile as tiff


# ----------------------------
# Helpers
# ----------------------------

def _is_dir(p: str) -> bool:
    return os.path.isdir(p)

def _is_file(p: str) -> bool:
    return os.path.isfile(p)

def _natural_key(s: str):
    # Natural sort: file_2.tif < file_10.tif
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r'\d+|\D+', s)]

def _resolve_sequence(spec: Union[str, Iterable[str]]) -> List[str]:
    """
    Accepts: directory path, glob pattern, or explicit iterable of file paths.
    Returns a naturally-sorted list of TIFF files.
    """
    if isinstance(spec, str):
        if _is_dir(spec):
            paths = [os.path.join(spec, f) for f in os.listdir(spec)]
        else:
            # assume glob pattern or a single file path
            paths = glob.glob(spec)
            if not paths and _is_file(spec):
                paths = [spec]
    else:
        paths = list(spec)

    # filter tiffs
    paths = [p for p in paths if os.path.isfile(p) and p.lower().endswith((".tif", ".tiff"))]
    if not paths:
        raise FileNotFoundError("No TIFF files found for the given sequence spec.")

    paths.sort(key=_natural_key)
    return paths


# ----------------------------
# Metadata / shape without full load
# ----------------------------

def get_volume_info(spec: Union[str, Iterable[str]]) -> Dict[str, Any]:
    """
    Inspect a volume (multipage TIFF or a sequence of TIFFs) without loading it fully.
    Returns shape=(Z,Y,X), dtype, and best-effort voxel size/units if present in tags.
    """
    info: Dict[str, Any] = {"shape": None, "dtype": None, "voxel_size": None, "units": None, "reader": None}

    if isinstance(spec, str) and _is_file(spec) and spec.lower().endswith((".tif", ".tiff")):
        # multipage TIFF
        with tiff.TiffFile(spec) as tf:
            series = tf.series[0]
            shp = series.shape
            # Accept (Z, Y, X) or (Y, X) -> assume single slice
            if len(shp) == 2:
                zyx = (1, shp[0], shp[1])
            elif len(shp) >= 3:
                # Handle possible extra axes (e.g., samples)
                # We pick the last two dims as Y,X and use the remaining as Z-like
                y, x = shp[-2], shp[-1]
                z = int(np.prod(shp[:-2]))
                zyx = (z, y, x)
            else:
                raise ValueError(f"Unexpected TIFF series shape: {shp}")

            info["shape"] = zyx
            info["dtype"] = series.dtype
            info["reader"] = "multipage"

            # Try to extract pixel size (resolution is pixels per unit)
            try:
                page0 = tf.pages[0]
                tags = page0.tags
                xres = tags.get("XResolution", None)
                yres = tags.get("YResolution", None)
                runit = tags.get("ResolutionUnit", None)  # 1=None, 2=inch, 3=cm (TIFF spec)
                if xres and yres and runit:
                    # XResolution is (num, den) pixels per unit
                    def _to_float(r):
                        # tifffile may return fractions.Fraction or tuple
                        try:
                            return float(r)
                        except Exception:
                            return r[0] / r[1]
                    px_per_unit_x = _to_float(xres.value)
                    px_per_unit_y = _to_float(yres.value)
                    if runit.value == 2:  # inch
                        unit = "inch"
                        sx = 1.0 / px_per_unit_x
                        sy = 1.0 / px_per_unit_y
                    elif runit.value == 3:  # cm
                        unit = "cm"
                        sx = 1.0 / px_per_unit_x
                        sy = 1.0 / px_per_unit_y
                    else:
                        unit = "unknown"
                        sx = 1.0 / px_per_unit_x
                        sy = 1.0 / px_per_unit_y
                    # We cannot know Z from tags; assume isotropic if you need to
                    info["voxel_size"] = (None, sy, sx)
                    info["units"] = unit
            except Exception:
                pass

    else:
        # sequence of single-page TIFFs
        files = _resolve_sequence(spec)
        with tiff.TiffFile(files[0]) as tf0:
            arr0 = tf0.asarray()
            if arr0.ndim != 2:
                raise ValueError(f"Expected 2D pages in sequence; got shape {arr0.shape} in {files[0]}")
            y, x = arr0.shape
            info["shape"] = (len(files), y, x)
            info["dtype"] = arr0.dtype
            info["reader"] = "sequence"

            # Try resolution from first page
            try:
                page0 = tf0.pages[0]
                tags = page0.tags
                xres = tags.get("XResolution", None)
                yres = tags.get("YResolution", None)
                runit = tags.get("ResolutionUnit", None)
                if xres and yres and runit:
                    def _to_float(r):
                        try:
                            return float(r)
                        except Exception:
                            return r[0] / r[1]
                    px_per_unit_x = _to_float(xres.value)
                    px_per_unit_y = _to_float(yres.value)
                    if runit.value == 2:
                        unit = "inch"
                        sx = 1.0 / px_per_unit_x
                        sy = 1.0 / px_per_unit_y
                    elif runit.value == 3:
                        unit = "cm"
                        sx = 1.0 / px_per_unit_x
                        sy = 1.0 / px_per_unit_y
                    else:
                        unit = "unknown"
                        sx = 1.0 / px_per_unit_x
                        sy = 1.0 / px_per_unit_y
                    info["voxel_size"] = (None, sy, sx)
                    info["units"] = unit
            except Exception:
                pass

    return info


# ----------------------------
# Lazy slice access
# ----------------------------

def read_slice(spec: Union[str, Iterable[str]], z: int) -> np.ndarray:
    """
    Read a single Z-slice (2D array) lazily without loading the whole volume.
    Works for a multipage TIFF file or a sequence of TIFF files.
    """
    if isinstance(spec, str) and _is_file(spec) and spec.lower().endswith((".tif", ".tiff")):
        with tiff.TiffFile(spec) as tf:
            series = tf.series[0]
            shp = series.shape
            # Map z to the appropriate page index (flatten all non-spatial axes)
            if len(shp) == 2:
                if z != 0:
                    raise IndexError("Requested z>0 from a single-slice TIFF.")
                return series.asarray()
            # Flatten non YX dims
            total_z = int(np.prod(shp[:-2]))
            if not (0 <= z < total_z):
                raise IndexError(f"z out of range [0,{total_z-1}].")
            return series.asarray(key=z)  # tifffile supports key=index for series
    else:
        files = _resolve_sequence(spec)
        if not (0 <= z < len(files)):
            raise IndexError(f"z out of range [0,{len(files)-1}].")
        # Read only that file
        return tiff.imread(files[z])


# ----------------------------
# Full volume read (with optional memmap for multipage TIFF)
# ----------------------------

def read_volume(
    spec: Union[str, Iterable[str]],
    *,
    memmap: bool = False
) -> np.ndarray:
    """
    Load a full 3D volume (Z,Y,X).
    - If 'spec' is a multipage TIFF path: tries memmap when memmap=True.
      Memmap only works when the TIFF data are contiguous/native; falls back gracefully.
    - If 'spec' is a directory/glob/list: reads sequence and stacks along Z (no memmap across files).
    """
    # Multipage TIFF
    if isinstance(spec, str) and _is_file(spec) and spec.lower().endswith((".tif", ".tiff")):
        if memmap:
            # Try tifffile.memmap (works only for certain file layouts)
            try:
                mm = tiff.memmap(spec)  # may raise on non-contiguous/compressed data
                # Ensure (Z,Y,X)
                arr = np.asarray(mm)
                if arr.ndim == 2:  # single slice
                    arr = arr[np.newaxis, :, :]
                elif arr.ndim >= 3:
                    # Flatten non-spatial dims to Z
                    y, x = arr.shape[-2], arr.shape[-1]
                    z = int(np.prod(arr.shape[:-2]))
                    arr = arr.reshape((z, y, x))
                return arr
            except Exception:
                # Fallback to normal read
                pass
        # Regular read (will allocate)
        arr = tiff.imread(spec)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        elif arr.ndim >= 3:
            y, x = arr.shape[-2], arr.shape[-1]
            z = int(np.prod(arr.shape[:-2]))
            arr = arr.reshape((z, y, x))
        return arr

    # Sequence of single-page TIFFs
    files = _resolve_sequence(spec)
    # Read/stack; for very large sequences consider prealloc with shape from metadata
    with tiff.TiffFile(files[0]) as tf0:
        y, x = tf0.asarray().shape
    z = len(files)
    out = np.empty((z, y, x), dtype=tiff.TiffFile(files[0]).series[0].dtype)
    for i, f in enumerate(files):
        out[i] = tiff.imread(f)
    return out