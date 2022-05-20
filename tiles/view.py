"""
This module contains bokeh code to display a viewer for the tiles game.
To use this viewer run the following code in a jupyter notebook.

#############################
from bokeh.plotting import show
from bokeh.io import output_notebook

import numpy as np

from tiles.view import modify_document_factory

grids = [
    np.array([[0, 2, 2, 0], [0, 2, 2, 0], [0, 0, 3, 0], [0, 1, 0, 0], [0, 0, 0, 0]]),
    np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 2, 2, 0], [0, 2, 2, 0]]),
]

modify_document = modify_document_factory(grids=grids)

output_notebook()
show(modify_document) # If running this on a remote server, add the url to the Jupyter server with notebook_url='{IP}:{port}'
#############################


"""
from __future__ import annotations
import logging


import bokeh
from bokeh.core.validation.warnings import FIXED_SIZING_MODE
from bokeh.core.validation import silence
from bokeh.models import ColumnDataSource, Button
from bokeh.layouts import layout, row, column
from bokeh.plotting import figure
import numpy as np

logger = logging.getLogger(__loader__.name)


def get_colours_from_grid(grid: np.ndarray, colour_map: Dict[int, str]):
    """"""
    colours = np.flipud(grid).flatten()
    colours = [colour_map[value] for value in colours]

    return colours


silence(FIXED_SIZING_MODE)


def modify_document_factory(grids: List[np.ndarray], colour_palette: str = "Paired"):
    """Function factory that produces bokeh viewers based on input data.

    Parameters:

    Returns:
        modify_document a function that can be used to instantiate a bokeh server.
            See module documentation for more details.
    """

    def modify_document(doc):
        """"""
        nonlocal colour_palette

        def update_to_next_grid():
            nonlocal current_index
            current_index = min(current_index + 1, len(grids) - 1)
            source.data["colours"] = get_colours_from_grid(grid=grids[current_index], colour_map=colour_map)

        def update_to_previous_grid():
            nonlocal current_index
            current_index = max(current_index - 1, 0)
            source.data["colours"] = get_colours_from_grid(grid=grids[current_index], colour_map=colour_map)

        current_index = 0
        grid = grids[current_index]

        y_size, x_size = grid.shape
        x_range = [str(v) for v in range(x_size)]
        y_range = [str(v) for v in range(y_size)]
        x = np.array([list(range(x_size)) for _ in range(y_size)]).flatten().astype(str).tolist()
        y = np.array([[i] * x_size for i in range(y_size)]).flatten().astype(str).tolist()

        unique_values = np.unique(np.concatenate(grid))
        colour_palette = bokeh.palettes.all_palettes[colour_palette][len(unique_values)]
        colour_map = {value: colour for (value, colour) in zip(unique_values, colour_palette)}
        colour_map[0] = "white"

        colours = get_colours_from_grid(grid=grid, colour_map=colour_map)

        data = {"colours": colours, "x": x, "y": y}
        source = ColumnDataSource(data=data)

        p = figure(toolbar_location=None, x_range=x_range, y_range=y_range)
        p.rect("x", "y", color="colours", width=1, height=1, source=source)
        p.axis.visible = False

        next_button = Button(label="Next")
        previous_button = Button(label="Previous")
        widgets = [next_button, previous_button]

        next_button.on_click(update_to_next_grid)
        previous_button.on_click(update_to_previous_grid)

        l = layout(row([p, column(widgets)]))

        doc.add_root(l)

    return modify_document
