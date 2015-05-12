import psycopg2
import psycopg2.extras
import numpy as np
from ioos_qartod.qc_tests import qc
import quantities as q
import pandas as pd
import StringIO
from pandas.io.excel import read_excel
from credentials import dsn, sheet_loc

conn = psycopg2.connect(dsn)
cur = conn.cursor()
cur.execute("""SELECT id, site_name FROM cbibs.d_station WHERE site_code != 'unknown'""")
station_ids = cur.fetchall()
cur.execute("SELECT id, actual_name FROM cbibs.d_variable WHERE actual_name IN ('wind_speed', 'sea_water_salinity', 'sea_water_temperature','sea_surface_wave_significant_height')")
var_ids = cur.fetchall()


def insert_qartod_result(obs_ids, test_id, flags):
    """Insert results of the QARTOD test into the database.  Actually
       attempts to do an upsert"""
    rep_test_id = np.repeat(test_id, len(obs_ids))
    df = pd.DataFrame([obs_ids, rep_test_id, flags]).T
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
#import pdb
#pdb.set_trace()
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
        # TODO: May want to add type checking as spreadsheets don't really
        # enforce types
        sensor_span = (conf['sensor_min'], conf['sensor_max'])
        if conf['user_min'] is not None and conf['user_max'] is not None:
            user_span = (conf['user_min'], conf['user_max'])
        else:
            user_span = None
        data = pd.read_sql("EXECUTE get_obs (%s, %s)", conn, params=(site, var),
                           index_col='id')
        #cur.execute("EXECUTE get_obs (%s, %s)", (site, var))
        res = qc.range_check(data['obs_value'], sensor_span, user_span)
        insert_qartod_result(data.index, qc.range_check.qartod_id, res)

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
        bbox_arr=[[ll_lon, ur_lon], [ll_lat, ur_lat]]
        
        data = pd.read_sql("""SELECT o.id, measure_ts,
                obs_value, l.longitude lon, l.latitude lat
                FROM cbibs.f_observation o JOIN cbibs.d_location l
                ON (o.d_location_id = l.id)
                JOIN d_station st ON st.id = o.d_station_id
                WHERE st.site_code = %s
                ORDER BY measure_ts""" , conn, params=(site,))
                
        res = qc.location_set_check(data['lon'], data['lat'], bbox_arr)
        insert_qartod_result(data.index, qc.location_set_check.qartod_id, res)

if "Location Test" in qc_config:
    location_range_adapter(qc_config['Location Test'])



def spike_check_adapter(config):
    for _, conf in config.iterrows():
        site = str(conf['site_code'])
        var = conf['variable_name']
        low = conf['low_threshold']
        high = conf['high_threshold']

        data = pd.read_sql("EXECUTE get_obs (%s, %s)", conn, params=(site, var),
                           index_col='id')

        res = qc.spike_check(data['obs_value'], low, high)
        insert_qartod_result(data.index, qc.spike_check.qartod_id, res)

if "Spike" in qc_config:
    spike_check_adapter(qc_config['Spike'])

def rate_of_change_check_adapter(config):
    for _, conf in config.iterrows():
        site = str(conf['site_code'])
        var = conf['variable_name']
        thresh = conf['thresh_val']

        data = pd.read_sql("EXECUTE get_obs (%s, %s)", conn, params=(site, var),
                           index_col='id')

        res = qc.rate_of_change_check(data['obs_value'], thresh)
        insert_qartod_result(data.index, qc.rate_of_change_check.qartod_id, res)

if "Rate of Change" in qc_config:
    rate_of_change_check_adapter(qc_config["Rate of Change"])



def flat_line_check_adapter(config):
    for _, conf in config.iterrows():
        site = str(conf['site_code'])
        var = conf['variable_name']
        low = conf['low_reps']
        high = conf['high_reps']
        eps = conf['epsilon']
        data = pd.read_sql("EXECUTE get_obs (%s, %s)", conn, params=(site, var),
                           index_col='id')

        
        res = qc.flat_line_check(data.index, low, high, eps)
        insert_qartod_result(data.index, qc.flat_line_check.qartod_id, res)
        
if "Flat Line" in qc_config:
    flat_line_check_adapter(qc_config["Flat Line"])

def lt_time_series_stuck_sensor_adapter(config):
    for _, conf in config.iterrows():
        site = str(conf['site_code'])
        var = conf['variable_name']
        EPS= conf['EPS']

        data = pd.read_sql("EXECUTE get_obs (%s, %s)", conn, params=(site, var),
                           index_col='id')

        res = qc.lt_time_series_stuck_sensor(data['obs_value'], EPS)
        insert_qartod_result(data.index, qc.lt_time_series_stuck_sensor.qartod_id, res)
if "LT Time Series Stuck Sensor" in qc_config:
    lt_time_series_stuck_sensor_adapter(qc_config['LT Time Series Stuck Sensor'])

def st_time_series_spike_adapter(config):
    for _, conf in config.iterrows():
        site = str(conf['site_code'])
        var = conf['variable_name']
        N = conf['N']
        M = conf['M']
        P = conf['P']
        data = pd.read_sql("EXECUTE get_obs (%s, %s)", conn, params=(site, var),
                           index_col='id')

        res = qc.st_time_series_spike(data['obs_value'], N, M, P)
        insert_qartod_result(data.index, qc.st_time_series_spike.qartod_id, res)
if "ST Time Series Spike" in qc_config:
    st_time_series_spike_adapter(qc_config["ST Time Series Spike"])

def st_time_series_segement_shift_adapter(config):
    for _, conf in config.iterrows():
        site = str(conf['site_code'])
        var = conf['variable_name']
        P = conf['P']
        m = conf['m']
        n = conf['n']
        data = pd.read_sql("EXECUTE get_obs (%s, %s)", conn, params=(site, var),
                           index_col='id')
        res = qc.st_time_series_segment_shift(data['obs_value'], P, m, n)
        insert_qartod_result(data.index, qc.st_time_series_segment_shift.qartod_id, res)
if "ST Time Segment Shift" in qc_config:
    st_time_series_segement_shift_adapter(qc_config["ST Time Segment Shift"])
    
def lt_time_series_rate_of_change_adapter(config):
    for _, conf in config.iterrows():
        site = str(conf['site_code'])
        var = conf['variable_name']
        MAXHSDIFF = conf['MAXHSDIFF']
        data = pd.read_sql("EXECUTE get_obs (%s, %s)", conn, params=(site, var),
                           index_col='id')
        res = qc.lt_time_series_rate_of_change(data['obs_value'], MAXHSDIFF)
        insert_qartod_result(data.index, qc.lt_time_series_rate_of_change.qartod_id, res)
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
