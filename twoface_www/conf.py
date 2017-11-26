# coding: utf-8

# Standard-library
from os import path

# Flask
from flask import Flask, render_template, g

# Third-party
import astropy.units as u
import numpy as np
from thejoker.sampler import JokerSamples
from twobody.celestial import RVOrbit, VelocityTrend1

from bokeh import events
from bokeh.plotting import figure, gridplot
from bokeh.models import ColumnDataSource, Whisker, TapTool, ResetTool, Circle
from bokeh.models.callbacks import CustomJS
from bokeh.embed import components

from twoface.config import TWOFACE_CACHE_PATH
from twoface.db import db_connect
from twoface.db import (JokerRun, AllStar, AllVisit, StarResult, Status,
                        AllVisitToAllStar)
from twoface.io import load_samples

app = Flask(__name__)

# ------------------------------------------------------------------------------
# Database

def get_db():
    """
    Opens a new database connection if there is none yet for the
    current application context.
    """
    if not hasattr(g, 'Session'):
        Session, engine = db_connect(database_path=app.config['DATABASE_PATH'],
                                     ensure_db_exists=False)
        g.Session = Session

    return g.Session

@app.teardown_appcontext
def shutdown_session(exception=None):
    if hasattr(g, 'Session'):
        g.Session.remove()

# ------------------------------------------------------------------------------
# Pages

@app.route('/')
def index():
    return render_template('pages/index.html')

@app.route('/plot/', defaults={'apogee_id': None})
@app.route('/plot/<string:apogee_id>')
def plot(apogee_id):

    # TODO: support different runs?
    run_name = 'apogee-jitter'

    if apogee_id is None:
        return render_template('errors/404.html', msg='No APOGEE ID specified!')

    session = get_db()
    star = session.query(AllStar).filter(AllStar.apogee_id == apogee_id).limit(1).one()
    data = star.apogeervdata()

    # Load samples
    samples = load_samples(path.join(TWOFACE_CACHE_PATH,
                                     '{0}.hdf5'.format(run_name)),
                           apogee_id)
    samples = JokerSamples(trend_cls=VelocityTrend1, **samples)

    # helpers:
    w = np.ptp(data.t.mjd)
    t_grid = np.linspace(data.t.mjd.min() - w*0.05,
                         data.t.mjd.max() + w*1.05,
                         1024)

    # Create the Bokeh plot of the data and orbit curves
    TOOLS = "pan,box_zoom,reset"
    data_plot = figure(x_axis_label='MJD', y_axis_label='RV [km/s]',
                       plot_width=700, plot_height=500,
                       x_range=[t_grid.min(), t_grid.max()],
                       tools=TOOLS)

    # Now plot orbits over the data:
    # TODO: control how many to plot?
    n_plot = 128
    all_data = []
    for i in range(n_plot):
        this_samples = dict()
        for k in samples.keys():
            this_samples[k] = samples[k][i]
        this_samples.pop('jitter', None)

        # get the trend parameters out
        trend_samples = dict()
        for k in samples.trend_cls.parameters:
            trend_samples[k] = this_samples.pop(k)
        trend = samples.trend_cls(t0=0., **trend_samples)

        orbit = RVOrbit(trend=trend, **this_samples)
        orbit_rv = orbit.generate_rv_curve(t_grid).to(u.km/u.s).value
        all_data.append(list(orbit_rv))

        data_plot.line(t_grid, orbit_rv, color='#aaaaaa',
                       line_alpha=0.25, line_width=1, name=str(i))

    # For any highlighted / selected points:
    highlighted = ColumnDataSource(data=dict(t=[], rv=[]))
    data_plot.line('t', 'rv', source=highlighted, alpha=1., color='#333333',
                   line_width=2)

    # For errorbars:
    upper = (data.rv + data.stddev).to(u.km/u.s).value
    lower = (data.rv - data.stddev).to(u.km/u.s).value
    source_error = ColumnDataSource(data=dict(base=data.t.mjd,
                                              lower=lower, upper=upper))

    data_plot.scatter(data.t.mjd, data.rv.to(u.km/u.s).value, color='#222222')
    data_plot.add_layout(
        Whisker(source=source_error, base="base", upper="upper", lower="lower",
                lower_head=None, upper_head=None)
    )

    # Plot the samples:
    samples_plot = figure(x_axis_label='period [day]',
                          y_axis_label='eccentricity',
                          x_axis_type="log", x_range=[8, 32768], y_range=[0, 1],
                          plot_width=500, plot_height=500,
                          tools=TOOLS)
    renderer = samples_plot.scatter(samples['P'].to(u.day).value,
                                    samples['ecc'].value,
                                    color='#222222', fill_alpha=0.5,
                                    line_alpha=0, line_width=5, size=10)

    renderer.nonselection_glyph = Circle(fill_alpha=0.05, fill_color="#222222",
                                         line_color=None)
    renderer.selection_glyph = Circle(fill_alpha=0.75, fill_color="#222222",
                                      line_color=None)

    # plot = data_plot
    plot = gridplot([[data_plot, samples_plot]])

    line_data = ColumnDataSource(data=dict(
        t_grid=list(t_grid),
        rv=list(all_data)))
    callback = CustomJS(args=dict(source=highlighted, line_data=line_data),
                        code="""
            var idx = cb_data.source.selected['1d'].indices[0];
            var d = source.data;

            if (idx !== undefined && idx !== null) {
                d['t'] = line_data.data['t_grid'];
                d['rv'] = line_data.data['rv'][idx];
                source.change.emit();
            }
        """)
    samples_plot.add_tools(TapTool(callback=callback))

    reset_callback = CustomJS(args=dict(source=highlighted), code="""
            var d = source.data;
            d['t'] = [];
            d['rv'] = [];
            source.change.emit();
    """)
    samples_plot.js_on_event(events.Reset, reset_callback)
    # samples_plot.add_tools(ResetTool(callback=reset_callback))

    script, div = components(plot)

    return render_template('pages/plot.html', apogee_id=apogee_id, star=star,
                           bokeh_script=script, bokeh_div=div)

# ------------------------------------------------------------------------------
# Error handlers.

@app.errorhandler(500)
def internal_error(error):
    # db_session.rollback()
    return render_template('errors/500.html'), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html', msg='Page not found.'), 404

# if not app.debug:
#     import logging
#     from logging import Formatter, FileHandler

#     file_handler = FileHandler('error.log')
#     file_handler.setFormatter(
#         Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
#     )
#     app.logger.setLevel(logging.INFO)
#     file_handler.setLevel(logging.INFO)
#     app.logger.addHandler(file_handler)
#     app.logger.info('errors')
