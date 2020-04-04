import io
import base64

from eelbrain import fmtxt
from eelbrain.plot._base import EelFigure
from IPython.core.display import display
from IPython.display import HTML
import matplotlib.figure


class Layout:
    """Display multiple plots in a multi-column layout

    Parameters
    ----------


    Examples
    --------
    Create a layout, add items and display them::

        layout = Layout()
        for i in range(3):
            p = plot.UTSStat(y[i], x)
            layout.add(p)
        # last item so that it gets displayed:
        layout

    If ``layout`` can not be the last item, use ``display(layout)``.

    Notes
    -----
    Based on https://stackoverflow.com/a/49566213/166700
    """
    def __init__(self, plots: list = None, border: int = 0):
        # string buffer for the HTML: initially some CSS; images to be appended
        options = [
            "display: inline-block;",
            "margin: 10px;",
        ]
        if border:
            options.append(f"border: {border}px solid #888888;")

        self.html = """
        <style>
        .floating-box {
        %s
        }
        </style>
        """ % '\n'.join(options)
        if plots:
            for plot in plots:
                self.add(plot)

    def add(self, obj):
        """Add a plot to the layout (matplotlib figure or eelbrain plot)"""
        fmtext_obj = fmtxt.asfmtext(obj)
        html = fmtxt.html(fmtext_obj)
        self.html += f'<div class="floating-box">{html}</div>'
        if isinstance(obj, matplotlib.figure.Figure):
            from matplotlib import pyplot
            pyplot.close(obj)
        elif isinstance(obj, EelFigure):
            obj.close()

    def linebreak(self):
        """Add a line break (use to limit the number of columns in the layout"""
        self.html += '<br />'

    def _ipython_display_(self):
        display(HTML(self.html))
