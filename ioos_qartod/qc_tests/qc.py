import numpy as np
from numpy import nan
import pyproj
import quantities as q
import pandas as pd
import multiprocessing


class QCFlags:
    """Primary flags for QARTOD"""
    # don't subclass Enum since values don't fit nicely into a numpy array
    GOOD_DATA = 1
    UNKNOWN = 2
    SUSPECT = 3
    BAD_DATA = 4
    MISSING = 9


# Could also use a class based approach, but believe this to be a little
# simpler for simple tests which are run once and don't need to maintain state.
# This is a bit hardcoded, but if needed, one can always change the id attribute
# if their internal representation of QARTOD tests differ
def add_qartod_ident(qartod_id, qartod_test_name):
    """
    Adds attributes to the QARTOD functions corresponding to database fields.
    """
    def dec(fn):
        fn.qartod_id = qartod_id
        fn.qartod_test_name = qartod_test_name
        return fn
    return dec


# TODO: Consider refactoring this to use a decorator with something like
# functools so we keep the code more DRY
def set_prev_qc(flag_arr, prev_qc):
    """Takes previous QC flags and applies them to the start of the array
       where the flag values are not unknown"""
    cond = prev_qc != QCFlags.UNKNOWN
    flag_arr[cond] = prev_qc[cond]


@add_qartod_ident(3, 'Location Test')
def location_set_check(lon, lat, bbox_arr=[[-180, -90], [180, 90]],
                       range_max=None):
    """
    Checks that longitude and latitude are within reasonable bounds
    defaulting to lon = [-180, 180] and lat = [-90, 90].
    Optionally, check for a maximum range parameter in great circle distance
    defaulting to meters which can also use a unit from the quantities library
    """
    bbox = np.array(bbox_arr)
    if bbox.shape != (2, 2):
        # TODO: Use more specific Exception types
        raise ValueError('Invalid bounding box dimensions')
    if lon.shape != lat.shape:
        raise ValueError('Shape not the same')
    flag_arr = np.ones_like(lon, dtype='uint8')
    if range_max is not None:
        ellipsoid = pyproj.Geod(ellps='WGS84')
        _, _, dist = ellipsoid.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
        dist_m = np.insert(dist, 0, 0) * q.meter
        flag_arr[dist_m > range_max] = QCFlags.SUSPECT
    flag_arr[(lon < bbox[0][0]) | (lat < bbox[0][1]) |
             (lon > bbox[1][0]) | (lat > bbox[1][1]) |
             (np.isnan(lon)) | (np.isnan(lat))] = QCFlags.BAD_DATA
    return flag_arr


@add_qartod_ident(4, 'Gross Range Test')
def range_check(arr, sensor_span, user_span=None):
    """
    Given a 2-tuple of sensor minimum/maximum values, flag data outside of
    range as bad data.  Optionally also flag data which falls outside of a user
    defined range
    """
    flag_arr = np.ones_like(arr, dtype='uint8')
    if len(sensor_span) != 2:
        raise ValueError("Sensor range extent must be size two")
    # ensure coordinates are in proper order
    s_span_sorted = sorted(sensor_span)
    if user_span is not None:
        if len(user_span) != 2:
            raise ValueError("User defined range extent must be size two")
        u_span_sorted = sorted(user_span)
        if (u_span_sorted[0] < s_span_sorted[0] or
           u_span_sorted[1] > s_span_sorted[1]):
            raise ValueError("User span range may not exceed sensor bounds")
        # test timing
        flag_arr[(arr <= u_span_sorted[0]) |
                 (arr >= u_span_sorted[1])] = QCFlags.SUSPECT
    flag_arr[(arr <= s_span_sorted[0]) |
             (arr >= s_span_sorted[1])] = QCFlags.BAD_DATA
    return flag_arr


def _process_time_chunk(value_pairs):
    """Takes values and thresholds for climatologies
       and returns whether passing or not, or returns UNKNOWN
       if the threshold is None."""
    vals = value_pairs[0]
    threshold = value_pairs[1]
    if threshold is not None:
        return ((vals >= threshold[0]) &
                (vals <= threshold[1])).astype('i4')
    else:
        return pd.Series(np.repeat(QCFlags.UNKNOWN, len(vals)), vals.index,
                         dtype='i4')


@add_qartod_ident(5, 'Climatology Test')
def climatology_check(time_series, clim_table, group_function):
    """
    Takes a pandas time series, a dict of 2-tuples with (low, high) thresholds
    as values, and a grouping function to group the time series into bins which
    correspond to the climatology lookup table.  Flags data within
    the threshold as good data and data lying outside of it as bad.  Data for
    which climatology values do not exist (i.e. no entry to look up in the dict)
    will be flagged as Unknown/not evaluated.
    """
    grouped_data = time_series.groupby(group_function)
    vals = [(g, clim_table.get(grp_val)) for (grp_val, g) in grouped_data]
    # should speed up processing of climatologies
    pool = multiprocessing.Pool()
    chunks = pool.map(_process_time_chunk, vals)
    res = pd.concat(chunks)
    #replace 0s from boolean with suspect values
    res[res == 0] = QCFlags.SUSPECT
    return res


@add_qartod_ident(6, 'Spike Test')
def spike_check(arr, low_thresh, high_thresh, prev_qc=None):
    """
    Determine if there is a spike at data point n-1 by subtracting
    the midpoint of n and n-2 and taking the absolute value of this
    quantity, seeing if it exceeds a a low or high threshold.
    Values which do not exceed either threshold are flagged good,
    values which exceed the low threshold are flagged suspect,
    and values which exceed the high threshold are flagged bad.
    The flag is set at point n-1.
    """
    # subtract the average from point at index n-1 and get the absolute value.
    if low_thresh >= high_thresh:
        raise ValueError("Low theshold value must be less than high threshold "
                         "value")
    val = np.abs(np.convolve(arr, [-0.5, 1, -0.5], mode='same'))
    # first and last elements can't contain three points,
    # so set difference to zero so these will avoid getting spike flagged
    val[[0, -1]] = 0
    flag_arr = ((val < low_thresh) +
               ((val >= low_thresh) & (val < high_thresh)) * QCFlags.SUSPECT +
                (val >= high_thresh) * QCFlags.BAD_DATA)
    if prev_qc is not None:
        set_prev_qc(flag_arr, prev_qc)
    return flag_arr



@add_qartod_ident(7, 'Rate of Change Test')
def rate_of_change_check(arr, thresh_val, prev_qc=None):
    """
    Checks the first order difference of a series of values to see if
    there are any values exceeding a threshold.  These are then marked as
    suspect.  It is up to the test operator to determine an appropriate
    threshold value for the absolute difference not to exceed
    """
    flag_arr = np.ones_like(arr, dtype='uint8')
    exceed = np.insert(np.abs(np.diff(arr)) > thresh_val, 0, False)
    if prev_qc is not None:
        flag_arr[0] = prev_qc[0]
    else:
        flag_arr[0] = QCFlags.UNKNOWN

    flag_arr[exceed] = QCFlags.SUSPECT
    return flag_arr

@add_qartod_ident(8, 'Flat Line Test')
def flat_line_check(arr, low_reps, high_reps, eps, prev_qc=None):
    """
    Check for repeated consecutively repeated values
    within a tolerance eps
    """
    if any([not isinstance(d, int) for d in [low_reps, high_reps]]):
        raise TypeError("Both low and high repetitions must be type int")
    flag_arr = np.ones_like(arr, dtype='uint8')
    if low_reps >= high_reps:
        raise ValueError("Low reps must be less than high reps")
    it = np.nditer(arr)
    # consider moving loop to C for efficiency
    for elem in it:
        idx = it.iterindex
        # check if low repetitions threshold is hit
        cur_flag = QCFlags.GOOD_DATA
        if idx >= low_reps:
            is_suspect = np.all(np.abs(arr[idx - low_reps:idx] - elem) < eps)
            if is_suspect:
                cur_flag = QCFlags.SUSPECT
            # since high reps is strictly greater than low reps, check it
            if is_suspect and idx >= high_reps:
                is_bad = np.all(np.abs(arr[idx - high_reps:idx - low_reps]
                                       - elem) < eps)
                if is_bad:
                    cur_flag = QCFlags.BAD_DATA
        flag_arr[idx] = cur_flag
    if prev_qc is not None:
        set_prev_qc(flag_arr, prev_qc)
    return flag_arr


@add_qartod_ident(10, 'Attenuated Signal Test')
def attenuated_signal_check(arr, times, min_var_warn, min_var_fail,
                            time_range=(None, None), check_type='std',
                            prev_qc=None):
    """Check that the range or standard deviation is below a certain threshold
       over a certain time period"""
    flag_arr = np.empty(arr.shape, dtype='uint8')
    flag_arr.fill(QCFlags.UNKNOWN)

    if time_range[0] is not None:
        if time_range[1] is not None:
            time_idx = (times >= time_range[0]) & (times <= time_range[1])
        else:
            time_idx = times >= time_range[0]
    elif time_range[1] is not None:
        time_idx = times <= time_range[1]
    else:
        time_idx = np.ones_like(times, dtype='bool')

    if check_type == 'std':
        check_val = np.std(arr[time_idx])
    elif check_type == 'range':
        check_val = np.ptp(arr[time_idx])
    else:
        raise ValueError("Check type '{}' is not defined".format(check_type))

    # set previous QC values first so that selected segment does not get
    # overlapped by previous QC flags
    if prev_qc is not None:
        set_prev_qc(flag_arr, prev_qc)

    if check_val >= min_var_fail and check_val < min_var_warn:
        flag_arr[time_idx] = QCFlags.SUSPECT
    elif check_val < min_var_fail:
        flag_arr[time_idx] = QCFlags.BAD_DATA
    else:
        flag_arr[time_idx] = QCFlags.GOOD_DATA
    return flag_arr

#Brian
#waves
@add_qartod_ident(12, 'ST Time Series Gap')
# Similar to  timing/ gap test
# Test 9 NOT COMPLETED NO TEST WRITTEN
def st_time_seires_gap(arr, N, start=0, end=0):
    '''Checks for N consecutive missing data points. This defines the size of 
    an unacceptable gap in the times series.It is the maximum number of consecutive 
    missing data points allowed.  A counter (C2) increments from 0 (zero) as 
    consecutive data points are missed. At the end of a gap of missing data, 
    this counter is compared to N. If C2 > N, the test is failed and a suspect 
    flag is set. The counter (C2) is reset to 0 after a data point is encountered.'''
    # Checking for gaps assumes that there are no entries in the dataframe for
    # a specific time. I need to know the interval in which observations are taken
    # so I can compute how many points have been missed.... 
   
    #  If N is 'time', not 'integer'. E.g, if obs are taken every hour
    # and data will fail if 6 hours is missed, N = 6 * 3600000000000. Operator needs to define the 
    # value of time is, i.e. hours, seconds, days... 
    # time is diffed in nonseconds. 
    #assumes arr is a dataframe
    #arr['2014-06-19T21:00:00.000000000-0400':'2014-07-30T22:00:00.000000000-0400']
    #data_frame = arr[start : end]  
    # N needs to be in nanoseconds

    
    diff_time= np.diff(arr)
    #diff redeuces len by 1
    time_arr = np.insert(diff_time, diff_time[0],diff_time[0]) 
    time_arr = time_arr.astype(int)
    flags_arr = ((arr_abs < N ) * QCFlags.GOOD_DATA + (arr_abs > N) * QCFlags.BAD_DATA) 
    return flag_arr 
            
    

@add_qartod_ident(13, 'LT Time Series Stuck Sensor')
# Test 16
def lt_time_series_stuck_sensor(arr, EPS):
    '''When some sensors and/or data collection platforms (DCPs) fail, the result
    can be a continuously repeated observation of the same value. This test compares 
    the present observation (POn) to a number (REP_CNT_FAIL or REP_CNT_SUSPECT) of 
    previous observations. POn is flagged if it has the same value as previous 
    observations within a tolerance value EPS to allow for numerical round-off error.
    Note that historical flags are not changed.'''
    # cannot think of a better method, other than a for loop...
    # this function is very sensitive to EPS value as an argument.
    flag_arr = np.empty(arr.shape, dtype='uint8')
    flag_arr.fill(QCFlags.UNKNOWN)
    arr_copy = np.copy(arr)
    arr_copy = arr_copy.astype(float)
    diff_copy = np.diff(arr_copy)
    # Fail When the five most recent observations are equal, POn is flagged fail.
    # I decided to use a moving average...
    # I ignore the three most recent observations,ie P0n is flagged to suspect
    # 5 point moving average
    WINDOW = 5
    weights = np.repeat(1.0, WINDOW)/ WINDOW
    ma =np.convolve(arr, weights)[WINDOW-1:-(WINDOW-1)]
    diff_ma = np.diff(ma)
    diff_copy[ : len(diff_ma)]= diff_ma
    diff_copy = np.insert(diff_copy, diff_copy[0], diff_copy[0]) 
    arr_abs = abs(diff_copy)
    flag_arr = ((arr_abs < EPS ) * QCFlags.BAD_DATA + (arr_abs > EPS) * QCFlags.GOOD_DATA) 

    return flag_arr

@add_qartod_ident(14, "ST Time Series Acceleration")
# test 13 NO TEST WRITTEN NEED MORE DETAIL
def st_time_series_acceleration(arr, M, N):
    '''The in-situ systems that collect these time series data can accumulate
    accelerations in all directions from multiple sensors. Any acceleration 
    that exceeds a practical value should be replaced by an 
    interpolated/extrapolated value.'''
    G = 9.80
    M5 = 0
    A = G * M
    #arr must be dtype=float or nan will not work
    arr[arr<A]= nan
    # need to check if None > N
    nans = np.isnan(arrs).sum()
    if nans < N:
    #Any acceleration values exceeding M*G are replaced with an 
    #operator-defined interpolated/extrapolated values.
    #Not sure how that would work...
    # this function interpolates based on the data
    # need further information before I can finish this test.
        interp_arr = Seires(arr).interpolate()
    return inter_arr

@add_qartod_ident(15, "ST Time Series Spike")
# test 10
def st_time_series_spike(arr, N, M, P):
    '''The Spike Test checks for spikes in a time series. Spikes are defined as 
    points more than M times the standard deviation (SD) from the mean. After
    the ST time series is received, the mean (MEAN) and standard deviation (SD)
    must be determined. Counters M1 and M2 are set to 0. Once a spike has been
    identified, the spike is replaced with the average (AVG) of the previous point 
    (n-1) and the following point (n+1). The counter, M1, is incremented as spikes
    are identified. The algorithm should iterate over the time series multiple (P)
    times, recomputing the mean and standard deviation for each iteration. After 
    the Pth iteration, a final spike count, M2, is run. The counters M1 and M2 are 
    compared to the number of spikes allowed. The time series is rejected if
    it contains too many spikes (generally set to N% of all points) or if spikes 
    remain after P iterations (M2 > 0). '''

    flag_good = np.empty(arr.shape, dtype='uint8')
    flag_good.fill(QCFlags.GOOD_DATA)
    flag_bad =  np.empty(arr.shape, dtype='uint8')
    flag_bad.fill(QCFlags.BAD_DATA)
    recur = 0
    M1_list = []
    arr_copy = np.copy(arr)

    if type(N) == int:
        N= N/100.0
    
    def percent_outliers():
        outliers = M1_list[0]/float(len(arr))
        return  outliers
    
    while recur < P:
        M1 = 0
        std = np.std(arr_copy)     
        mean = np.mean(arr_copy)
        #should vectorize this one
        #I know how you hate for loops and enumerate
        for i,ob in enumerate(arr_copy):
            if abs(ob - mean)> M * std:
                try:
                    M1+=1
                    arr_copy[i] = (arr_copy[i-1] + arr_copy[i+1])/2
                except IndexError:
                    continue
        recur+=1
        M1_list.append(M1)
        M2 = M1

    if M2 > 0 or percent_outliers() > N:
        return flag_bad 
    
    elif percent_outliers() < N and M2 == 0:
        return flag_good
            
@add_qartod_ident(16, "ST Time Segment Shift")
#test 12
def st_time_series_segment_shift(arr,P, m=0, n=0):
    '''The time series is broken into n segments m points long. Segment means 
    are computed for each of the n segments. Each segment mean is compared to 
    neighboring segments. If the difference in the means of two consecutive 
    segments exceeds P, the ST time series data are rejected. The operator 
    defines n segments, m points, and P.

    The operator determines the number of segments (n) to be compared in the 
    time series and the length of each segment (m) to be compared in the time 
    series. Then, m or n can be computed by the other in conjunction with the
    length of the entire time series. The operator also defines the mean shift 
    (P) that is allowed in the time series. A mean value (MEAN [n]) is computed
    for each of the n segments. The means of consecutive segment are
    then compared. If the differences of the means exceed the allowed mean 
    shift (P) provided by the user, the entire time series is failed.'''
    # (above) enitre time series fails but also states if < P flag = 1
    # for all values of n-1...
    
    if n == 0  and m == 0:
        raise ValueError('n or m need a value')
    if m == 0:
        m = len(arr)/n
    if n == 0:
        n = len(arr)/m

    flag_good = np.empty(arr.shape, dtype='uint8')
    flag_good.fill(QCFlags.GOOD_DATA)
    flag_bad =  np.empty(arr.shape, dtype='uint8')
    flag_bad.fill(QCFlags.BAD_DATA)
    
    arr_copy = np.copy(arr)
    flag_arr = np.empty(arr.shape, dtype='uint8')
    # resizing removes elements to fit dimensions or adds copies of data
    arr_copy.resize(n,m)
    arr_diff=np.diff(np.mean(arr_copy,axis=1))
    arr_bool = arr_diff >= P
    if np.sum(arr_bool) >= 1:
         return flag_bad
    elif np.sum(arr_bool) < 1:
         return flag_good

@add_qartod_ident(17, "LT Time Series Mean and Standard Deviation")
#test 15
def lt_time_series_mean_and_standard_deviation(arr, N):
    '''Check that TSVAL value is within limits defined by the operator. Operator 
    defines the period over which the mean and standard deviation are calculated 
    and the number of allowable standard deviations (N).'''
     
    # See attenuated signal test 10

@add_qartod_ident(18,"LT Time Series Bulk Wave Parameters Max/Min/Acceptable Range")
# test 19
def lt_time_series_bulk_wave_parameters_max_min_acceptable_range(WVHGT, WVPD, WVDIR, WVSP, MINWH, MAXWH, MINWP, MAXWP, MINSV, MAXSV):
    '''The operator should establish maximum and minimum values for the bulk wave 
    parameters; wave height (WVHGT), period (WVPD), direction (WVDIR), and
    spreading (WVSP) (if provided). If the wave height fails this test, then no
    bulk wave parameters should be released. Otherwise, suspect flags are set.
    Operator supplies minimum wave height (MINWH), maximum wave height (MAXWH), 
    minimum wave period (MINWP), maximum wave period (MAXWP), minimum spreading 
    value (MINSV), and maximum spreading value (MAXSV).'''
    
    # See gross range test?
    # See current direction?
    # Looks like this a combination of the two.

   # if WVHGT < MINWH or WVHGT > MAXWH:
        # flag 4 for all parameters
   # if WVPD < MINWP or WVPD > MAXWP:
        #flag = 3.
   # if WVDIR < 0.0 or WVDIR > 360:
        # flag = 3.
   # if WVSP < MINSV or WVSP > MAXSV:
        #flag = 3.
@add_qartod_ident(19, "LT Time Series Rate of Change")
# test 20
def lt_time_series_rate_of_change(arr, MAXHSDIFF):
    '''This test is applied only to wave height Hs. The operator selects a 
    threshold value, MAXHSDIFF, and the two most recent observations Hs (n)
    and Hs(n-1) are checked to see if the rate of change is exceeded.'''
    
    arr_diff = np.insert(arr, arr[0], arr[0]) 
    arr_diff = np.diff(arr_diff)
    arr_abs = abs(arr_diff)
    flag_arr = ((arr_abs <= MAXHSDIFF) * QCFlags.GOOD_DATA + (arr_abs > MAXHSDIFF) * QCFlags.BAD_DATA)    
    return flag_arr

#Currents
@add_qartod_ident(20, "Current Speed")
# test 10
def current_speed(arr, SPDMAX):
    '''Current speed, CSPD(j), is typically provided as a positive value. This 
    test checks for unrealistically high current speed values and is applied to 
    each depth bin, i'''
    flag_arr = ((arr <= SPDMAX ) * QCFlags.GOOD_DATA + (arr > SPDMAX) * QCFlags.BAD_DATA) 

    return flag_arr

@add_qartod_ident(21, "Current Direction")
#test 11
def current_direction(arr):
    '''This test ensures that the current direction values fall between 0 and 360
    degrees, inclusive. In most systems, 0 is reported as NO current and 360
    degrees indicates a current to the north. This test is applied to each depth bin, i.'''

    arr_copy = np.copy(arr)
    flag_arr = np.ones_like(arr, dtype='uint8')
    flag_arr = np.logical_and(arr>=0.0, arr<=360.0)* QCFlags.GOOD_DATA 
    + np.logical_or(arr< 0.0, arr>360.0) * QCFlags.BAD_DATA
    
    return flag_arr

@add_qartod_ident(22, "Horizontal Velocity")
# test 12
def horizontal_velocity():
    '''Horizontal velocities, u(i) and v(i), may be represented as components 
    (East-West and North-South; Alongshore and Cross-Shore: Alongshelf and
    Cross-Shelf, Along-Isobath and Cross-Isobath, etc.) of the current speed
    and direction. This test ensures that speeds in the respective horizontal
    directions are valid. Maximum allowed values may differ in the orthogonal
    directions. This test is applied to each depth bin, i.'''

    #I don't fully understand this one. Will need to come back.
    # u(i) east == (0,90) (270,360) west == (90-180) (180, 360) 
    # v(i) north == (0,180) south ==( 180, 360)
    #  

@add_qartod_ident(22, "Vertical Velocity")
# test 13
def vertical_velocity():
    '''Vertical velocities, w(i), are reported by many ADCPs. They are calculated 
    just like the horizontal velocities, but along the vertical axis. This test is 
    applied to each depth bin, i.'''
    # Not sure either...
