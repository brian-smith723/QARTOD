import psycopg2
import psycopg2.extras
import numpy as np
from ioos_qartod.qc_tests import qc
import quantities as q
import pandas as pd
import StringIO
from credentials import dsn

conn = psycopg2.connect(dsn)
psycopg2.extras.register_hstore(conn)
cur = conn.cursor()
cur.execute("""SELECT id, site_name FROM cbibs.d_station WHERE site_code != 'unknown'""")
station_ids = cur.fetchall()
cur.execute("SELECT id, actual_name FROM cbibs.d_variable WHERE actual_name IN ('wind_speed', 'sea_water_salinity', 'sea_water_temperature')")
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
for station_id in station_ids:
    print(station_id[1])
    for var in var_ids:
        print(var[1])
        cur.execute("""select * from (select o.id, measure_ts,
                    obs_value, l.longitude lon, l.latitude lat
                    from cbibs.f_observation o JOIN cbibs.d_location l
                    ON (o.d_location_id = l.id) WHERE d_station_id = %s
                    AND d_variable_id = %s
                    ORDER BY measure_ts DESC LIMIT 50000) t ORDER BY measure_ts;""",
                    (station_id[0], var[0]))

        arr = np.fromiter(cur, dtype=[('id', 'i4'), ('timestamp', 'datetime64[us]'),
                                      ('obs_val', 'f8'),
                                      ('lon', 'f8'), ('lat', 'f8')])
        print('rate of change')
        rate_of_change_flags = qc.rate_of_change_check(arr['obs_val'], 3.5)
        insert_qartod_result(arr['id'], qc.rate_of_change_check.qartod_id,
                            rate_of_change_flags)
        print('location')
        loc_flags = qc.location_set_check(arr['lon'], arr['lat'],
                                            range_max=20.0 * q.kilometer)
        insert_qartod_result(arr['id'], qc.location_set_check.qartod_id, loc_flags)
        print('gross range')
        range_flags = qc.range_check(arr['obs_val'], (-2.5, 40))
        insert_qartod_result(arr['id'], qc.range_check.qartod_id, range_flags)
        print('spike')
        spike_flags = qc.spike_check(arr['obs_val'], 3.0, 8.0)
        insert_qartod_result(arr['id'], qc.spike_check.qartod_id, spike_flags)
        print('flat line')
        rep_flags = qc.flat_line_check(arr['obs_val'], 3, 5, 0.001)
        insert_qartod_result(arr['id'], qc.flat_line_check.qartod_id, rep_flags)

    final_flags = reduce(np.maximum,
                        [loc_flags, range_flags, spike_flags, rep_flags])

upsert_results()
conn.commit()
