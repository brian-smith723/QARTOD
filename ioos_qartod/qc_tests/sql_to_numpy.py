import psycopg2
import psycopg2.extras
import numpy as np
from ioos_qartod.qc_tests import qc
import quantities as q
import pandas as pd
import StringIO
from pandas.io.excel import read_excel
from credentials import dsn, sheet_loc
from collections import OrderedDict
from datetime import datetime

conn = psycopg2.connect(dsn)
cur = conn.cursor()
cur.execute("""SELECT id, site_code FROM cbibs.d_station WHERE site_code != 'unknown'""")
station_ids = cur.fetchall()
station_lookup = dict(reversed(t) for t in station_ids)
cur.execute("SELECT id, actual_name FROM cbibs.d_variable")
var_ids = cur.fetchall()
var_lookup = dict(reversed(t) for t in var_ids)

def add_test_history(station_code, variable_name, test_id, time_range, test_args):
    var_id = var_lookup[variable_name]
    station_id = station_lookup[station_code]
    t_range = psycopg2.extras.DateTimeTZRange(time_range[0], time_range[1],
                                              '[]')
    now = datetime.utcnow()
    test_args_json = psycopg2.extras.Json(test_args)
    # test_args_json = psycopg2.extras.Json(test_args)
    #import ipdb; ipdb.set_trace()
    cur.execute("""INSERT INTO test_parameters (run_time, d_station_id, d_variable_id,
                    qartod_test_id, time_range, test_params) VALUES (%s, %s, %s,
                    %s, %s, %s)""", (now, station_id, var_id, test_id,
                                     t_range, test_args_json))

def insert_qartod_result(obs_ids, test_id, flags):
    """Insert results of the QARTOD test into the database.  Actually
       attempts to do an upsert"""
    rep_test_id = np.repeat(test_id, len(obs_ids))

    df = pd.DataFrame(OrderedDict([('f_observation_id', obs_ids),
                                  ('qartod_code', rep_test_id),
                                  ('qc_code_id', flags)]))
    # should run faster to insert many rows
    csv_obj = StringIO.StringIO(df.to_csv(None, index=False, header=False))
    cur.copy_from(csv_obj, 'qartod_staging', sep=',',
                  columns=['f_observation_id', 'qartod_code', 'qc_code_id'])

def upsert_results():
    """
    Run an upsert-like query in Postgres once all the QC results have
    been added
    """
    cur.execute('LOCK TABLE cbibs.j_qa_code_secondary_fob IN EXCLUSIVE MODE')
    # update any already existing records
    # replace any existing records with
    # if 2 (unknown) then use pre-existing QC value
    cur.execute('''UPDATE cbibs.j_qa_code_secondary_fob t
                          SET qc_code_id = CASE qs.qc_code_id WHEN 2 THEN
                                                t.qc_code_id
                                           ELSE
                                                qs.qc_code_id
                                           END
                          FROM qartod_staging qs
                          WHERE t.f_observation_id = qs.f_observation_id AND
                                t.qartod_code = qs.qartod_code
                          ''')
    cur.execute('''INSERT INTO cbibs.j_qa_code_secondary_fob
                          SELECT qs.f_observation_id, qs.qartod_code,
                                 qs.qc_code_id
                          FROM qartod_staging qs
                          LEFT JOIN cbibs.j_qa_code_secondary_fob t ON
                                    (qs.f_observation_id = t.f_observation_id
                                      AND
                                     qs.qartod_code = t.qartod_code)
                                WHERE t.f_observation_id IS NULL AND
                                      t.qartod_code IS NULL
                          ''')

# create staging/temp table to initially store the results in
cur.execute('''CREATE TEMPORARY TABLE qartod_staging
                (LIKE cbibs.j_qa_code_secondary_fob)''')

cur.execute('''CREATE TEMPORARY TABLE qartod_loc_staging
                (id integer, qc_result integer)''')

# read the sheet into a dict of DataFrames
qc_config = pd.read_excel(sheet_loc, None)
# TODO: could get rid of lon for most of these as they aren't location test
# preload prepare statement
cur.execute("""PREPARE get_obs AS SELECT o.id, measure_ts,
                obs_value, l.longitude lon, l.latitude lat
                FROM cbibs.f_observation o JOIN cbibs.d_location l
                ON (o.d_location_id = l.id)
                JOIN d_variable v ON v.id = o.d_variable_id
                JOIN d_station st ON st.id = o.d_station_id
                WHERE st.site_code = $1
                AND v.actual_name = $2
                ORDER BY measure_ts""")

# these can almost certainly become more efficient if we configure by site
# instead of test, as we don't have to naively refetch obs
def gross_range_adapter(config):
    # I wish I could inline here
    for _, conf in config.iterrows():
        # Pandas reads the string and may attempt to coerce to numeric
        site = str(conf['site_code'])
        var = conf['variable_name']
        kwargs = {
                'sensor_min': conf['sensor_min'],
                'sensor_max':conf['sensor_max'],
                'user_min': conf['user_min'],
                'user_max': conf['user_max']
        }

        # TODO: May want to add type checking as spreadsheets don't really
        # enforce types
        sensor_span = (kwargs['sensor_min'], kwargs['sensor_max'])
        if kwargs['user_min'] is not None and kwargs['user_max'] is not None:
            user_span = (kwargs['user_min'], kwargs['user_max'])
        else:
            user_span = None

        data = pd.read_sql("EXECUTE get_obs (%s, %s)", conn, params=(site, var),
                           index_col='id')
        #cur.execute("EXECUTE get_obs (%s, %s)", (site, var))
        tbounds = data.measure_ts.min(), data.measure_ts.max()
        res = qc.range_check(data['obs_value'], sensor_span, user_span)
        insert_qartod_result(data.index, qc.range_check.qartod_id, res)
        add_test_history(site, var, qc.range_check.qartod_id, tbounds,
                kwargs)

if "Gross Range" in qc_config:
    gross_range_adapter(qc_config['Gross Range'])

# TODO: Add remainder of test

def location_range_adapter(config):
    for _, conf in config.iterrows():
        site = str(conf['site_code'])
        ll_lat = conf['ll_lat']
        ll_lon = conf['ll_lon']
        ur_lat = conf['ur_lat']
        ur_lon = conf['ur_lon']
        bbox_arr = [[ll_lon, ur_lon], [ll_lat, ur_lat]]
        variable_group = conf['variable_group']

        data = pd.read_sql("""SELECT gl.id, l.longitude lon, l.latitude lat
                            FROM d_variable_group_location gl
                            JOIN d_variable_group vg ON vg.id = gl.d_variable_group_id
                            JOIN d_station st ON st.id = gl.d_station_id
                            JOIN d_location l ON l.id = gl.d_location_id
                            WHERE st.site_code = %s AND vg.group_name = %s""",
                        conn, params=(site, variable_group))
        res = qc.location_set_check(data['lon'], data['lat'], bbox_arr)
        df = pd.DataFrame(OrderedDict([('id', data['id']), ('qc_result', res)]))
        csv_obj = StringIO.StringIO(df.to_csv(None, index=False, header=False))
        # faster to copy_from?
        cur.copy_from(csv_obj, 'qartod_loc_staging', sep=',',
                      columns=['id', 'qc_result'])
        cur.execute("""UPDATE d_variable_group_location group_loc SET
                    location_test=t.qc_result FROM qartod_loc_staging t
                    WHERE t.id = group_loc.id""")

if "Location Test" in qc_config:
    location_range_adapter(qc_config['Location Test'])

def spike_check_adapter(config):
    for _, conf in config.iterrows():
        site = str(conf['site_code'])
        var = conf['variable_name']
        kwargs = {
                'low_thresh' : conf['low_threshold'],
                'high_thresh' : conf['high_threshold']
        }


        data = pd.read_sql("EXECUTE get_obs (%s, %s)", conn, params=(site, var),
                           index_col='id')
        tbounds = data.measure_ts.min(), data.measure_ts.max()
        res = qc.spike_check(data['obs_value'], **kwargs)
        insert_qartod_result(data.index, qc.spike_check.qartod_id, res)
        add_test_history(site, var, qc.spike_check.qartod_id, tbounds, kwargs)
if "Spike" in qc_config:
    spike_check_adapter(qc_config['Spike'])

def rate_of_change_check_adapter(config):
    for _, conf in config.iterrows():
        site = str(conf['site_code'])
        var = conf['variable_name']
        thresh = conf['thresh_val']

        data = pd.read_sql("EXECUTE get_obs (%s, %s)", conn, params=(site, var),
                           index_col='id')

        kwargs = {'thresh_val': thresh}
        tbounds = data.measure_ts.min(), data.measure_ts.max()
        res = qc.rate_of_change_check(data['obs_value'], **kwargs)
        insert_qartod_result(data.index, qc.rate_of_change_check.qartod_id, res)
        add_test_history(site, var, qc.rate_of_change_check.qartod_id, tbounds,
                kwargs)

if "Rate of Change" in qc_config:
    rate_of_change_check_adapter(qc_config["Rate of Change"])

def flat_line_check_adapter(config):
    for _, conf in config.iterrows():
        site = str(conf['site_code'])
        var = conf['variable_name']
        kwargs = {
                  'low_reps': conf['low_reps'],
                  'high_reps': conf['high_reps'],
                  'eps': conf['epsilon']
        }

        data = pd.read_sql("EXECUTE get_obs (%s, %s)", conn, params=(site, var),
                           index_col='id')
        tbounds = data.measure_ts.min(), data.measure_ts.max()
        res = qc.flat_line_check(data.index, **kwargs)
        insert_qartod_result(data.index, qc.flat_line_check.qartod_id, res)
        add_test_history(site, var, qc.flat_line_check.qartod_id, tbounds,
                         kwargs)

if "Flat Line" in qc_config:
    flat_line_check_adapter(qc_config["Flat Line"])

def lt_time_series_stuck_sensor_adapter(config):
    for _, conf in config.iterrows():
        site = str(conf['site_code'])
        var = conf['variable_name']
        kwargs = {
                'EPS': conf['EPS']
        }

        data = pd.read_sql("EXECUTE get_obs (%s, %s)", conn, params=(site, var),
                           index_col='id')
        tbounds = data.measure_ts.min(), data.measure_ts.max()
        res = qc.lt_time_series_stuck_sensor(data['obs_value'], **kwargs)
        insert_qartod_result(data.index, qc.lt_time_series_stuck_sensor.qartod_id, res)
        add_test_history(site, var, qc.lt_time_series_stuck_sensor.qartod_id, tbounds,
                kwargs)
if "LT Time Series Stuck Sensor" in qc_config:
    lt_time_series_stuck_sensor_adapter(qc_config['LT Time Series Stuck Sensor'])

def st_time_series_spike_adapter(config):
    for _, conf in config.iterrows():
        site = str(conf['site_code'])
        var = conf['variable_name']
        kwargs = {
                'N': conf['N'],
                'M': conf['M'],
                'P': conf['P']
        }
        data = pd.read_sql("EXECUTE get_obs (%s, %s)", conn, params=(site, var),
                           index_col='id')
        tbounds = data.measure_ts.min(), data.measure_ts.max()
        res = qc.st_time_series_spike(data['obs_value'], **kwargs)
        insert_qartod_result(data.index, qc.st_time_series_spike.qartod_id, res)
        add_test_history(site, var, qc.st_time_series_spike.qartod_id, tbounds,
                kwargs)
if "ST Time Series Spike" in qc_config:
    st_time_series_spike_adapter(qc_config["ST Time Series Spike"])

def st_time_series_segement_shift_adapter(config):
    for _, conf in config.iterrows():
        site = str(conf['site_code'])
        var = conf['variable_name']
        kwargs= {
                'P' : conf['P'],
                'm' : conf['m'],
                'n' : conf['n']
        }
        data = pd.read_sql("EXECUTE get_obs (%s, %s)", conn, params=(site, var),
                           index_col='id')
        tbounds = data.measure_ts.min(), data.measure_ts.max()
        res = qc.st_time_series_segment_shift(data['obs_value'], **kwargs)
        insert_qartod_result(data.index, qc.st_time_series_segment_shift.qartod_id, res)
        add_test_history(site, var, qc.st_time_series_segment_shift.qartod_id, tbounds,
                kwargs)
if "ST Time Segment Shift" in qc_config:
    st_time_series_segement_shift_adapter(qc_config["ST Time Segment Shift"])

def lt_time_series_rate_of_change_adapter(config):
    for _, conf in config.iterrows():
        site = str(conf['site_code'])
        var = conf['variable_name']
        kwargs ={
                'MAXHSDIFF': conf['MAXHSDIFF']
        }
        data = pd.read_sql("EXECUTE get_obs (%s, %s)", conn, params=(site, var),
                           index_col='id')
        tbounds = data.measure_ts.min(), data.measure_ts.max()
        res = qc.lt_time_series_rate_of_change(data['obs_value'], **kwargs)
        insert_qartod_result(data.index, qc.lt_time_series_rate_of_change.qartod_id, res)
        add_test_history(site, var, qc.lt_time_series_rate_of_change.qartod_id, tbounds,
                kwargs)
if "LT Time Series Rate of Change" in qc_config:
    lt_time_series_rate_of_change_adapter(qc_config["LT Time Series Rate of Change"])

# current_direction and current_velocity produce an empty array
# no data avaiable?

#def current_speed_adapter(config):
#    for _, conf in config.iterrows():
#        site = str(config['site_code'])
#        var = conf['variable_name']
#        SPDMAX = conf['SPDMAX']
#        data = pd.read_sql("EXECUTE get_obs (%s, %s)", conn, params=(site, var),
#                           index_col='id')
#
#        res = qc.current_speed(data['obs_val'], SPDMAX)
#        insert_qartod_result(data.index, qc.current_speed.qartod_id, res)
#if "Current Speed" in qc_config:
#    current_speed_adapter(qc_config["Current Speed"])
#
#def current_direction_adapter(config):
#    for _, conf in config.iterrows():
#        site = str(config['site_code'])
#        var = conf['variable_name']
#        data = pd.read_sql("EXECUTE get_obs (%s, %s)", conn, params=(site, var),
#                           index_col='id')
#
#        res = qc.current_direction(data['obs_val'])
#        insert_qartod_result(data['obs_val'], qc.current_direction.qartod_id, res)
#if "Current Direction" in qc_config:
#    current_speed_adapter(qc_config["Current Direction"])

upsert_results()
conn.commit()
