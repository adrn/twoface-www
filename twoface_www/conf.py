# coding: utf-8

# Standard-library
import json
from os import path

# Flask
from flask import Flask, render_template, g

# Third-party
import astropy.units as u
import numpy as np
import plotly
import plotly.graph_objs as go
from thejoker.sampler import JokerSamples
from twobody.celestial import RVOrbit, VelocityTrend1

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

    # Create the Plotly Data Structure

    graph_data = []

    w = np.ptp(data.t.mjd)
    t_grid = np.linspace(data.t.mjd.min() - w*0.05,
                         data.t.mjd.max() + w*1.05,
                         1024)

    # plot orbits over the data
    # TODO: how many to plot?
    n_plot = 128
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

        graph_data.append(
            go.Scatter(x=t_grid, y=orbit_rv,
                       mode='lines', customdata=i,
                       xaxis='x', yaxis='y',
                       line=dict(color='#555555', width=1),
                       opacity=0.25
                       )
        )

    graph_data.append(
        go.Scatter(x=data.t.mjd,
                   y=data.rv.to(u.km/u.s).value,
                   mode='markers',
                   error_y=dict(type='data',
                                array=data.stddev.to(u.km/u.s).value,
                                visible=True),
                   marker=dict(color='#222222'),
                   xaxis='x', yaxis='y')
    )

    graph_data.append(go.Scatter(
        x=samples['P'].to(u.day).value,
        y=samples['ecc'].value,
        mode='markers',
        xaxis='x2', yaxis='y2'))

    layout = dict(
        title='',
        height=512,
        xaxis=dict(
            anchor='y',
            title='time [MJD]',
            domain=[0, 0.6]
        ),
        yaxis=dict(
            anchor='x',
            title='radial velocity [km/s]'
        ),
        xaxis2=dict(
            anchor='y2',
            type='log',
            autorange=True,
            title='period [day]',
            domain=[0.7, 1.]
        ),
        yaxis2=dict(
            anchor='x2',
            title='eccentricity'
        ),
        showlegend=False
    )

    graph = dict(data=graph_data, layout=layout)

    # Convert the figures to JSON
    graph_json = json.dumps(graph, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('pages/plot.html', apogee_id=apogee_id,
                           star=star,
                           graph=graph_json)

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
