# Data Visulaztion Utilities: TIFF I/O + Interactive 3-Slice Viewer

This package provides:
- **io_tiff**: reads a single multipage TIFF or a sequence of single-page TIFFs (directory, glob, or list), with lazy single-slice reads and optional `tifffile.memmap` for multipage files.
- **create_interactive_slice_plotter** An interactive 3-view slice viewer (XY, ZY, ZX) volumes with the following features: 
    - linked zoom/pan 
    - interactive sliders to scroll through CT slices
    - adjustable min/max for intensity limits
    - preset colormaps
    - toggle on/off histogram row for corresponding slices
    - toggle on/off "trinarize" so 
    - file name and "Save" button to save current view as a file
    - "Reset" button to return to original view

## Install
```bash
pip install tifffile ipywidgets ipympl matplotlib