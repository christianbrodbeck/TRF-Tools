import io
import base64

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

    def add(self, p):
        """Add a plot to the layout (matplotlib figure or eelbrain plot)"""
        is_figure = isinstance(p, matplotlib.figure.Figure)
        figure = p if is_figure else p.figure

        # Create a PNG representation of the figure
        bio = io.BytesIO()  # bytes buffer for the plot
        figure.canvas.print_png(bio)  # make a png of the plot in the buffer

        # close the figure so it does not appear in the notebook
        if is_figure:
            from matplotlib import pyplot
            pyplot.close(figure)
        else:
            p.close()

        # encode the bytes as string using base 64
        image_base64 = base64.b64encode(bio.getvalue()).decode()
        self.html += f'<div class="floating-box"><img src="data:image/png;base64,{image_base64}\n"></div>'

    def linebreak(self):
        """Add a line break (use to limit the number of columns in the layout"""
        self.html += '<br />'

    def _ipython_display_(self):
        display(HTML(self.html))
