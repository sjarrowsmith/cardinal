# cardinal.py
#
# A Python module for a 'PMCC-like' detector
#
# Stephen Arrowsmith (sarrowsmith@smu.edu)

import numpy as np
import pandas as pd
from array_analysis import *
from obspy.core import AttribDict
from pyproj import Geod
from numpy.linalg import inv
from matplotlib.dates import date2num, num2date, set_epoch
set_epoch('0000-12-31T00:00:00')    # Using the original matplotlib epoch
import matplotlib.pyplot as plt
import networkx as nx
import warnings, utm, io, dask, pickle, sqlite3, os, pdb
from dask.distributed import Client
from tqdm import tqdm
warnings.filterwarnings("ignore")

def polar_plot_families(families, r_axis='velocity'):
    '''
    Produces a summary polar plot of all families

    r_axis is the parameter to plot as a function of radius
    c_axis is the parameter to plot as a function of symbol color
    s_axis is the parameter to plot as a function of symbol size
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.grid(True)
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi/2.0)

    # Extracting detection parameters:
    start_time = families[0,:] - np.min(families[0,:])
    theta = families[4,:]*np.pi/180.
    mean_freq = np.mean((families[2,:],families[3,:]), axis=0)
    bandwidth = families[4,:] - families[3,:]
    mean_vel = families[6,:]
    n_pixels = families[9,:]

    hours = []
    for mlabtime in families[0,:]:
        mlabdatetime = num2date(mlabtime)
        hours.append(mlabdatetime.hour + mlabdatetime.minute/60)

    if r_axis == 'velocity':
        cm = ax.scatter(theta, mean_vel, s=bandwidth/4, c=mean_freq, vmin=0, vmax=2)
        plt.colorbar(cm)
        ax.set_rlim([0.0,0.8])
        plt.title('r-axis: trace velocity, color: frequency, size: Bandwidth')
    elif r_axis == 'hour':
        cm = ax.scatter(theta, np.array(hours), s=bandwidth/4, c=start_time)
        plt.colorbar(cm)
        plt.title('r-axis: Hour of day, color: Start time (days), size: Bandwidth')
    plt.show()

def read_families_from_db(dbname):
    '''
    Reads families from a SQLite3 database
    '''

    conn = sqlite3.connect(dbname)
    c = conn.cursor()

    detections = c.execute('select start_time, end_time, min_freq, max_freq, mean_baz, std_baz, mean_vel, std_vel, max_sem, n_pixels from detect').fetchall()
    conn.close()

    detections = np.array(detections).transpose()

    return detections

def write_families_to_db(dbname, families):
    '''
    Writes a set of families (or detections) to a SQLite3 database
    '''

    if not(os.path.exists(dbname)):
        conn = sqlite3.connect(dbname)
        c = conn.cursor()
        c.execute('''CREATE TABLE detect (id integer, start_time real, end_time real, min_freq real, max_freq real, mean_baz real, std_baz real, mean_vel real, std_vel real, max_sem real, n_pixels integer)''')
        conn.commit(); conn.close()

    if families is not None:
        conn = sqlite3.connect(dbname)
        c = conn.cursor()
        max_id = c.execute('SELECT max(id) from detect').fetchall()[0][0]
        
        if max_id is None:
            max_id = 0
        else:
            max_id = max_id + 1
        
        for i in range(0, families.shape[1]):
            c.execute('INSERT INTO detect VALUES (' + \
                    str(i+max_id) + ',' + str(families[0,i]) + ',' + \
                    str(families[1,i]) + ',' + str(families[2,i]) + ',' + \
                    str(families[3,i]) + ',' + str(families[4,i]) + ',' + \
                    str(families[5,i]) + ',' + str(families[6,i]) + ',' + \
                    str(families[7,i]) + ',' + str(families[8,i]) + ',' + \
                    str(families[9,i]) + ')')
        conn.commit(); conn.close()

def save_sliding_window_multifreq(fname, st, f_bands, T, B, V, S):
    '''
    Saves the inputs and results of Cardinal processing to a file
    '''

    pickle.dump([st,f_bands,T,B,V,S], open(fname, 'wb'))

def load_sliding_window_multifreq(fname):
    '''
    Loads the inputs and results of Cardinal processing from a file
    '''

    results = pickle.load(open(fname, 'rb'))
    st = results[0]; f_bands = results[1]; T = results[2]; B = results[3]; V = results[4]; S = results[5]

    return st, f_bands, T, B, V, S

def plot_sliding_window_multifreq(st, element, f_bands, T, B, V, S, title= None, event_window=None, bandpass=None,
                                  semblance_threshold=0.7, clim_baz=None, clim_vtr=[0,1],
                                  plot_trace_vel=False, log_freq=False, cmap_cyclic='twilight', cmap_sequential='pink_r',
                                  twin_plot=None, f_lim=None, plot_real_amplitude=False, amplitude_units='Pa',
                                  ix=None, pixels_in_families=None, figsize=(9,5), fname_plot=None):
    '''
    Plots the results of sliding-window array processing that span multiple frequency bands

    Required inputs:
    - st contains the ObsPy Stream of the data
    - element is a string that contains the name of the element to plot the time series data for
    - f_bands is a Pandas dataframe containing the frequency band information (see pmcc_fbands)
    - T is a NumPy array of times of length NT (the number of time windows with array processing results)
    - B is a NumPy array of backazimuths of length [NB, NT] where NB is the number of frequency bands
    - V is a NumPy array of phase/trace velocities of length [NB, NT]
    - S is a NumPy array of semblances of length [NB, NT]

    Optional parameters:
    - title is a string that adds a user defined title to the plot and, if bandpass is used, appends the bandpass frequencies
    - event_window is a tuple that takes the distance (km) between source and receiver, the UTC time that the event
      occurred, and the vmin and vmax celerity then plots predicted arrival time window 
      e.g. (15, UTCDateTime('2018-01-01 14:00:00'), .34, .26)
    - bandpass is a list that takes a min and max frequency to be filtered
    - semblance_threshold is the minimum value of semblance to plot B or V values for (default=0.7)
    - clim_baz is a list that puts an optional color limit for backazimuth, e.g., clim_baz=[10,30]
    - clim_vtr is a list that puts an optional color limit for phase/trace velocity
    - plot_trace_vel is a Boolean that defines whether to plot trace velocity (if False then semblance is plotted instead)
    - log_freq is a Boolean that defines whether to plot array processing results on a log frequency scale
    - cmap_cyclic is the Matplotlib colormap used to plot backazimuth (try 'twilight_shifted' for a different look)
    - f_lim can provide limits for plotting the frequency axis
    - plot_real_amplitude should be set to True to plot the amplitudes, otherwise they are normalized
    - amplitude_units is the units to display on the y-axis for the waveform
    - ix is the indices of frequencies, times where semblance > threshold
    - pixels_in_families is a Numpy array of all unique pixel ID's that are in families
    - fname_plot is an optional filename to save the plot to
    '''

    S_filt = S.copy()

    
    if (pixels_in_families is not None) and (ix is not None):
        # Set all semblances to zero where the pixel is not in a family:
        x = np.zeros(S.shape)
        x[ix[0][pixels_in_families],ix[1][pixels_in_families]] = 1   # Makes a mask where 1 means plot value
        S_filt[x == 0] = 0

    tr = st.select(station=element)[0]
    start_time_string = str(tr.stats.starttime).split('.')[0].replace('T',' ')
    fig, ax = plt.subplots(figsize=figsize)
    ax1 = plt.subplot(3,1,1)
    t_tr = np.arange(0, tr.stats.npts*tr.stats.delta, tr.stats.delta)
    
    if title is not None:
        plt.title(title+ ": Bandpass " + str(bandpass))
    
    #creating time window
    if event_window is not None:
        dist= event_window[0]
        event_hour=float(str(event_window[1]).split('T')[1].split(":")[0])
        event_minute=float(str(event_window[1]).split('T')[1].split(":")[1])
        event_second=float((str(event_window[1]).split('T')[1].split(":")[2])[:2])
        event_time=(60*60*event_hour)+(60*event_minute)+(event_second)
        vmin=event_window[2]
        vmax=event_window[3]
        arrival_1= (dist/vmax) + event_time
        arrival_2= (dist/vmin) + event_time
        plt.axvline(event_time, c= 'r')
        plt.axvline(arrival_1, c= 'g')
        plt.axvline(arrival_2, c= 'g')
    
    
    if (bandpass is None):
        pass
    else:
        tr.filter('bandpass', freqmin=min(bandpass), freqmax=max(bandpass))
    
    t_tr, y = fix_lengths(t_tr, tr.data)

    if plot_real_amplitude:
        plt.plot(t_tr, y, 'k-')
        plt.ylabel(amplitude_units)
    else:
        plt.plot(t_tr, y/np.max(np.abs(y)), 'k-')
        ax1.tick_params(labelleft=False)
    ax1.tick_params(labelbottom=False)

    ax2 = plt.subplot(3,1,2, sharex=ax1)
    ax2.tick_params(labelbottom=False)
    ix = np.where(S_filt < semblance_threshold)
    B_plt = B.copy()
    B_plt[ix] = None
    
    #if clim_baz[0]>clim_baz[1]:
    #    B_plt=B_plt.copy()
    #    np.where((B_plt > 0) & (B_plt<clim_baz[0]), B_plt+360, B_plt)
    #else:
    #    pass
    
    t_plot = np.hstack((T,T[len(T)-1]+np.diff(T)[0]))
    f_plot = np.hstack((f_bands['fmin'].values, f_bands['fmax'].values[len(f_bands['fmax'])-1]))
    
    #pcm1 = plt.pcolormesh(T, f_bands['fcenter'].values, B_plt, cmap=plt.get_cmap(cmap_cyclic), shading='nearest')
    pcm1 = plt.pcolormesh(t_plot, f_plot, B_plt, cmap=plt.get_cmap(cmap_cyclic), shading='flat')
    if clim_baz is not None and clim_baz[0]<clim_baz[1]:
        plt.clim([clim_baz[0], clim_baz[1]])
    elif clim_baz is not None and clim_baz[0]>clim_baz[1]:
        plt.clim([clim_baz[0], clim_baz[1]+360])
    plt.ylabel('Freq. (Hz)')
    if log_freq:
        plt.yscale('log')

    if plot_trace_vel:
        ax3 = plt.subplot(3,1,3, sharex=ax1, sharey=ax2)
        V_plt = V.copy()
        V_plt[ix] = None
        #pcm2 = plt.pcolormesh(T, f_bands['fcenter'].values, V_plt, cmap=plt.get_cmap('pink_r'), shading='nearest')
        pcm2 = plt.pcolormesh(t_plot, f_plot, V_plt, cmap=plt.get_cmap(cmap_sequential), shading='flat')
        if clim_vtr is not None:
            plt.clim([clim_vtr[0], clim_vtr[1]])
        plt.ylabel('Freq. (Hz)')
        plt.xlabel('Time (s) after ' + start_time_string)
        if log_freq:
            plt.yscale('log')
    else:
        # Plotting semblance
        ax3 = plt.subplot(3,1,3, sharex=ax1, sharey=ax2)
        #pcm2 = plt.pcolormesh(T, f_bands['fcenter'].values, S, cmap=plt.get_cmap('pink_r'), shading='nearest')
        pcm2 = plt.pcolormesh(t_plot, f_plot, S, cmap=plt.get_cmap(cmap_sequential), shading='flat')
        plt.clim([0,1])
        plt.ylabel('Freq. (Hz)')
        plt.xlabel('Time (s) after ' + start_time_string)
        if log_freq:
            plt.yscale('log')
    
    if twin_plot is not None:
        plt.xlim(twin_plot)
    else:
        plt.xlim([t_tr[0], t_tr[len(t_tr)-1]])
    
    if f_lim is not None:
        plt.ylim(f_lim)

    
    # Manually adding colorbars:
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.86, 0.386, 0.02, 0.22])
    fig.colorbar(pcm1, cax=cbar_ax)
    cbar_ax.set_ylabel('Azimuth (deg.)')
    cbar_ax.locator_params(nbins=6)
    
    if clim_baz is not None:
        if clim_baz[0]>clim_baz[1]:
            tick_labels= np.array(cbar_ax.get_yticks())
            tick_labels= tick_labels.astype(int)
            new_labels=(np.where((tick_labels > 360), tick_labels-360, tick_labels))
            cbar_ax.set_yticklabels(new_labels)
    
    cbar_ax = fig.add_axes([0.86, 0.113, 0.02, 0.22])
    fig.colorbar(pcm2, cax=cbar_ax)
    cbar_ax.locator_params(nbins=5)
    if plot_trace_vel:
        cbar_ax.set_ylabel('Velocity (km/s)')
    else:
        cbar_ax.set_ylabel('Semblance')
    
    if fname_plot is not None:
        plt.savefig(fname_plot)

def fix_lengths(t, y):

    t_out = t.copy()
    y_out = y.copy()

    if len(t_out) > len(y_out):
        t_out = t_out[0:len(y_out)]
    elif len(y_out) > len(t_out):
        y_out = y_out[0:len(y_out)]

    return t_out, y_out

def plot_sliding_window(st, element, T, B, V, C=None, v_min=0, v_max=5., 
                        semblance_threshold=None, twin_plot=None, clim=[0,1], figsize=(9,5)):
    '''
    Plots the results of sliding-window array processing
    
    Inputs:
    st - ObsPy Stream object containing array data
    element - Name of the element to plot the time series data for
    T - Times of array processing estimates (center of time windows) (s)
    B - Backazimuths
    V - Trace velocities (km/s)
    C - Optional color of points (e.g., Semblance, F-statistic, Correlation)
    v_max - Maximum trace velocity for y-axis on trace velocity
    twin_plot - List containing the start and end times (in seconds) to plot
    '''
    
    tr = st.select(station=element)[0]

    fig, ax = plt.subplots(figsize=figsize)

    ax1 = plt.subplot(3,1,1)
    t_tr = np.arange(0, tr.stats.npts*tr.stats.delta, tr.stats.delta)
    plt.plot(t_tr, tr.data/np.max(np.abs(tr.data)), 'k-')
    ax1.tick_params(labelbottom=False)

    ax2 = plt.subplot(3,1,2, sharex=ax1)
    if C is not None:
        if semblance_threshold is not None:
            ix2 = np.where(C < semblance_threshold)
            plt.scatter(T[ix2], B[ix2], s=0.05, c='lightgray')
            ix = np.where(C >= semblance_threshold)
            plt.scatter(T[ix], B[ix], s=4, c=C[ix], vmin=clim[0], vmax=clim[1], cmap=plt.get_cmap('hot_r'))
        else:
            plt.scatter(T, B, s=4, c=C, vmin=clim[0], vmax=clim[1], cmap=plt.get_cmap('hot_r'))
    else:
        plt.plot(T, B, 'k.')
    ax2.set_ylim([0,360])
    ax2.set_ylabel('Backazimuth')
    if twin_plot is not None:
        plt.xlim(twin_plot)
    ax2.tick_params(labelbottom=False)

    ax3 = plt.subplot(3,1,3, sharex=ax1)
    if C is not None:
        if semblance_threshold is not None:
            ix2 = np.where(C < semblance_threshold)
            plt.scatter(T[ix2], V[ix2], s=0.05, c='lightgray')
            ix = np.where(C >= semblance_threshold)
            plt.scatter(T[ix], V[ix], s=4, c=C[ix], vmin=clim[0], vmax=clim[1], cmap=plt.get_cmap('hot_r'))
        else:
            plt.scatter(T, V, s=4, c=C, vmin=clim[0], vmax=clim[1], cmap=plt.get_cmap('hot_r'))
    else:
        plt.plot(T, V, 'k.')
    ax3.set_ylim([v_min,v_max])
    ax3.set_ylabel('Phase vel.')
    ax3.set_xlabel('Time (s) after ' + str(tr.stats.starttime).split('.')[0].replace('T', ' '))
    
    plt.xlim([t_tr[0], t_tr[len(t_tr)-1]])

    ax1.get_yaxis().set_ticks([])

def make_custom_fbands(f_min=0.01, f_max=50, win_min=3, win_max=200, overlap=0.1, type='third_octave'):
    '''
    Makes a set of custom frequency bands and time windows for processing

    Input parameters:
    f_min is the minimum frequency
    f_max is the maximum frequency
    win_min is the minimum time window (to be used for maximum frequency)
    win_max is the maximum time window (to be used for minimum frequency)
    type is the type of filter band to use
    '''

    m, b = np.polyfit([1/f_min, 1/f_max], [win_max, win_min], 1)

    column_names = ['band', 'fmin', 'fcenter', 'fmax', 'win', 'step']
    f_bands = pd.DataFrame(columns = column_names)

    if type == 'third_octave':
        i = 0
        while f_min * np.cbrt(2) <= f_max:
            i = i + 1
            fmin = f_min
            fmax = f_min * np.cbrt(2)
            fcenter = (fmin + fmax)/2
            win = m * (1/fcenter) + b
            step = win * overlap
            f_min = fmax
            f_bands = f_bands.append({'band': i, 'fmin': fmin, 'fcenter': fcenter, 'fmax': fmax, 'win': win, 'step': step}, ignore_index=True)
    elif type == 'octave':
        i = 0
        while f_min * 2 <= f_max:
            i = i + 1
            fmin = f_min
            fmax = f_min * 2
            fcenter = (fmin + fmax)/2
            win = m * (1/fcenter) + b
            step = win * overlap
            f_min = fmax
            f_bands = f_bands.append({'band': i, 'fmin': fmin, 'fcenter': fcenter, 'fmax': fmax, 'win': win, 'step': step}, ignore_index=True)
    elif type == 'decade':
        i = 0
        while f_min * 10 <= f_max:
            i = i + 1
            fmin = f_min
            fmax = f_min * 10
            fcenter = (fmin + fmax)/2
            win = m * (1/fcenter) + b
            step = win * overlap
            f_min = fmax
            f_bands = f_bands.append({'band': i, 'fmin': fmin, 'fcenter': fcenter, 'fmax': fmax, 'win': win, 'step': step}, ignore_index=True)
    
    return f_bands

def pmcc_fbands():
    '''
    Returns the PMCC array processing time-window and frequency-band
    configurations, as reported in Matoza et al. (2013)

    Note the variables are:
    band = Band number
    fmin = Minimum frequency of band (Hz)
    fmax = Maximum frequency of band (Hz)
    win = Window length (s)
    step = Time step (s) (10% of window length for PMCC)
    '''

    pmcc_str = '''band fmin fcenter fmax win step
    1 0.0100 0.0126 0.0151 200.0000 20.0000
    2 0.0151 0.0190 0.0229 142.1606 14.2161
    3 0.0229 0.0288 0.0347 103.9404 10.3940
    4 0.0347 0.0436 0.0524 78.6846 7.8685
    5 0.0524 0.0659 0.0794 61.9956 6.1996
    6 0.0794 0.0997 0.1201 50.9676 5.0968
    7 0.1201 0.1509 0.1818 43.6803 4.3680
    8 0.1818 0.2284 0.2751 38.8648 3.8865
    9 0.2751 0.3457 0.4163 35.6828 3.5683
    10 0.4163 0.5231 0.6300 33.5801 3.3580
    11 0.6300 0.7916 0.9533 32.1907 3.2191
    12 0.9533 1.1980 1.4427 31.2725 3.1273
    13 1.4427 1.8130 2.1833 30.6658 3.0666
    14 2.1833 2.7436 3.3040 30.2649 3.0265
    15 3.3040 4.1520 5.0000 30.0000 3.0000'''

    data = io.StringIO(pmcc_str)
    fbands = pd.read_csv(data, delim_whitespace=True)

    return fbands

def extend_pmcc_fbands(f_bands, fmax):
    '''
    Extends the PMCC frequency-band configuration up to fmax
    
    f_bands is a Pandas dataframe, generated using pmcc_fbands

    Uses the same logarithmic frequency band spacing, and window length
    spacing rules, where window length is linearly proportional to period

    Returns an updated Pandas dataframe, f_bands
    '''

    fmin = f_bands['fmin'].values
    fbwidth = f_bands['fmax'].values - f_bands['fmin'].values
    m, b = np.polyfit(fmin, fbwidth, 1)

    m_win, b_win = np.polyfit(1/f_bands['fcenter'].values, f_bands['win'].values, 1)

    f_max_moving = 5; band_ix = 15
    while f_max_moving <= fmax:
        f_min = f_max_moving
        f_max = f_min + (m*f_min + b)
        if f_max > fmax:
            break
        f_cen = (f_min+f_max)/2
        band_ix = band_ix + 1
        win = m_win*(1/f_cen) + b_win
        step = win*0.1
        f_bands = f_bands.append({'band': band_ix, 'fmin': f_min, 'fcenter': f_cen, 'fmax': f_max, 'win': win, 'step': step}, ignore_index=True)
        f_max_moving = f_max
    
    return f_bands

def convert_to_slowness(baz, vel):
    '''
    Converts a backazimuth and phase/trace velocity to a slowness vector
    '''
    
    sl_y = np.abs(np.sqrt((1/vel**2)/((np.tan(np.deg2rad(baz)))**2+1)))
    sl_x = np.abs(sl_y * np.tan(np.deg2rad(baz)))
    if baz > 180:
        sl_x = -sl_x
    if (baz > 90) and (baz < 270):
        sl_y = -sl_y
    
    return sl_x, sl_y

def adjust_times_for_slowness(st, X, sl_x, sl_y):
    '''
    Adjusts start times in a Stream, st, for array coordinates given in X and a
    given slowness vector defined by sl_x, sl_y
    '''
    
    for i in range(0, X.shape[0]):
        t_shift = X[i,0]*sl_x + X[i,1]*sl_y
        st[i].stats.starttime = st[i].stats.starttime + t_shift
    
    return st

def compute_all_distances_for_pixel(i, ix, T_ix, F_ix, B_ix, sigma_t=2, sigma_f=2, sigma_b=10):
    '''
    Computes the distances between the i'th pixel and all other pixels that
    exceed the threshold
    '''
    
    d1 = ((ix[1][i]-ix[1])**2)/(sigma_t**2)
    d2 = ((ix[0][i]-ix[0])**2)/(sigma_f**2)
    d3 = ((B_ix[i]-B_ix)**2)/(sigma_b**2)
    
    distances = np.sqrt(d1 + d2 + d3)
    
    return distances

def make_families(T, B, V, S, f_bands, ref_time, threshold=0.6, dist_thres=1, min_pixels=6,
                  sigma_t=2, sigma_f=2, sigma_b=10, n_forward=200):
    '''
    Makes families by applying a semblance threshold and clustering resultant pixels based
    on weighted distance

    Inputs:
    T - array of times corresponding to array processing results
    B - estimated backazimuths for all times and frequencies
    V - estimated phase velocities
    S - estimated semblances
    f_bands - Pandas dataframe containing frequency band information
    ref_time - A Matplotlib datenumber containing the reference time for T
    threshold - semblance threshold for detection
    dist_thres - Threshold weighted distance for clustering pixels
    min_pixels - Threshold minimum number of pixels for a family
    sigma_t - Standard deviation in time indices for clustering
    sigma_f - Standard deviation in frequency indices for clustering
    sigma_b - Standard deviation in backazimuth for clustering
    n_forward - Number of pixels after a pixel (in time) to compute weighted distances

    Outputs:
    ix - Indices of frequencies, times where semblance > threshold
    pixels_in_families - Numpy array of all unique pixel ID's that are in families
    families - List of families (each contains unique pixel ID's in that family)
    
    Note: ix and pixels_in_families are provided for plotting purposes, such that:
    x = np.zeros(S.shape)
    x[ix[0][pixels_in_families],ix[1][pixels_in_families]] = 1   # Makes a mask where 1 means plot value
    '''
    
    # Extracting pixels, and associated parameters, above the semblance threshold:
    ix = np.where(S>=threshold)
    S_ix = S[ix]; B_ix = B[ix]; V_ix = V[ix]
    F_ix = f_bands['fcenter'].values[ix[0]]; T_ix = T[ix[1]]
    pixel_ids = np.arange(0, len(ix[0]))    # A list of unique pixel ID's

    # Splitting ix into an index on frequency (ix_f) and an index on time (ix_t):
    ix_f = ix[0]; ix_t = ix[1]

    # Sorting the pixel vectors by time:
    ix_sort_time = np.argsort(ix_t)
    ix_f = ix_f[ix_sort_time]; ix_t = ix_t[ix_sort_time]
    B_ix = B_ix[ix_sort_time]; T_ix = T_ix[ix_sort_time]
    F_ix = F_ix[ix_sort_time]; S_ix = S_ix[ix_sort_time]
    V_ix = V_ix[ix_sort_time]

    # Looping over each individual pixel in the pixel vectors:
    for i in range(0, len(B_ix)):

        # Checking for nearby pixels within a distance threshold:
        d1 = (ix_f[i] - ix_f[i+1:i+n_forward])**2 / sigma_f**2
        d2 = (ix_t[i] - ix_t[i+1:i+n_forward])**2 / sigma_t**2
        d3 = (B_ix[i] - B_ix[i+1:i+n_forward])**2 / sigma_b**2
        d = np.sqrt(d1+d2+d3)                                        # The distance to each i+1 -> i+n_forward pixel
        ixd = i + 1 + np.where(d <= dist_thres)[0]                   # The index of each associated pixel

        # Saving all pairs of pixels that are associated:
        assoc = np.vstack((np.tile(i, ixd.shape), ixd)).transpose()

        if i == 0:
            assoc_all = assoc
        else:
            assoc_all = np.vstack((assoc_all, assoc))

    # Applying the networkx library to group pairs into families:
    G = nx.Graph()
    G.add_edges_from(assoc_all)
    components = nx.connected_components(G)
    families = []
    for c in components:
        if len(list(c)) > min_pixels:
            families.append(list(c))

    # Post-processing families:
    pixels_in_families = np.empty(0, dtype=int)   # List of all pixels contained in families
    starttime = []; endtime = []; minfreq = []; maxfreq = []
    meanbaz = []; stdbaz = []; meanvel = []; stdvel = []
    maxsemb = []; npixels = []
    for family in families:
        pixels_in_families = np.hstack((pixels_in_families, family))
        starttime.append(ref_time + np.min(T_ix[family])/86400.)
        endtime.append(ref_time + np.max(T_ix[family])/86400.)
        minfreq.append(np.min(F_ix[family]))
        maxfreq.append(np.max(F_ix[family]))
        meanbaz.append(np.mean(B_ix[family]))
        stdbaz.append(np.std(B_ix[family]))
        meanvel.append(np.mean(V_ix[family]))
        stdvel.append(np.std(V_ix[family]))
        maxsemb.append(np.max(S_ix[family]))
        npixels.append(len(family))
    detections = np.array([starttime,endtime,minfreq,maxfreq,meanbaz,stdbaz,meanvel,stdvel,maxsemb,npixels])
    
    # Recombining ix as a tuple for plotting:
    ix = (ix_f, ix_t)
    
    return ix, pixels_in_families, detections

def df_families(ref_time, families):
    '''
    Returns a Pandas dataframe of families, sorted by start time, and with time in seconds
    for direct comparison with results plotted with plot_sliding_window_multifreq
    '''

    families_df = families.copy()
    families_df[0,:] = (families_df[0,:] - ref_time)*86400
    families_df[1,:] = (families_df[1,:] - ref_time)*86400

    df = pd.DataFrame(data=families_df.transpose(),
                    columns=['start_time','end_time','min_freq','max_freq',
                            'mean_baz','std_baz','mean_vel','std_vel','max_semb','n_pixels'])
    df = df.sort_values('start_time')

    return df

def sliding_time_array_fk_multifreq_tblock(st, element, f_bands, t_start=None, t_end=None, n_workers=1,
                                    sll_x=-3.6, slm_x=3.6, sll_y=-3.6, slm_y=3.6, sl_s=0.18, sl_corr=[0.,0.]):
    '''
    Runs sliding_time_array_fk_multifreq by parsing different time blocks to different threads, rather than
    different frequencies. This should be faster when processing large amounts of data.
    '''

    if t_end is None:
        t_end = st[0].stats.npts * st[0].stats.delta
    if t_start is None:
        t_start = 0
    
    t_dur = t_end - t_start
    t_block = int(t_dur/n_workers)

    # Making lists of start and end times to run as separate threads:
    t_start_times = []; t_end_times = []
    t_starti = t_start; t_endi = t_starti + t_block
    while t_endi + t_block <= t_end:
        t_start_times.append(t_starti)
        t_end_times.append(t_endi)
        t_starti = t_starti + t_block
        t_endi = t_endi + t_block
    t_start_times.append(t_starti)
    t_end_times.append(t_end-1)

    # Running sliding_time_array_fk_multifreq for each start/end block:
    client = Client(threads_per_worker=1, n_workers=n_workers)
    dask_all = []
    for i in range(0, len(t_start_times)):
        dask_out = dask.delayed(sliding_time_array_fk_multifreq)(st, element, f_bands, t_start = t_start_times[i],
                                                                 t_end = t_end_times[i], sll_x=sll_x, slm_x=slm_x,
                                                                 sll_y=sll_y, slm_y=slm_y, sl_s=sl_s, sl_corr=sl_corr)
        dask_all.append(dask_out)
    out = dask.compute(*dask_all)
    client.close()

    # Rearranging the output from all threads:
    for i in range(0, len(out)):
        if i == 0:
            T = out[i][0]; B = out[i][1]; V = out[i][2]; S = out[i][3]
        else:
            T = np.hstack((T, out[i][0]))
            B = np.hstack((B, out[i][1]))
            V = np.hstack((V, out[i][2]))
            S = np.hstack((S, out[i][3]))
    
    return T, B, V, S

def sliding_time_array_fk_multifreq(st, element, f_bands, t_start=None, t_end=None, n_workers=1,
                                    sll_x=-3.6, slm_x=3.6, sll_y=-3.6, slm_y=3.6, sl_s=0.18, sl_corr=[0.,0.],
                                    use_geographic_coords=True):
    '''
    Processes st with sliding window FK analysis in multiple frequency bands.

    Required inputs:
    - st contains the ObsPy Stream of the data
    - f_bands is a Pandas dataframe containing the frequency band information

    Optional parameters:
    - t_start is the start time (in seconds after the start of the ObsPy Stream) to process (None processes the whole Stream)
    - t_end is the end time (in seconds after the start of the ObsPy Stream) to process (None processes the whole Stream)
    - n_workers is the number of threads to use to do the computation (Using dask library if n_workers > 1)
    - Remaining parameters define the slowness plane
    '''
    
    if n_workers > 1:
        client = Client(threads_per_worker=1, n_workers=n_workers)
    
    tr = st.select(station=element)[0]    # Trace of reference element

    # Defining t_start, t_end:
    if (t_start == None) and (t_end == None):
        t_start = 1
        t_end = (tr.stats.npts * tr.stats.delta)-1

    # Processing each frequency band with sliding window FK processing:
    T_all = []; B_all = []; V_all = []; S_all = []; dask_all = []
    for f_band in f_bands['band'].values:
        win_len = f_bands[f_bands['band'] == f_band]['win'].values[0]
        frqlow = f_bands[f_bands['band'] == f_band]['fmin'].values[0]
        frqhigh = f_bands[f_bands['band'] == f_band]['fmax'].values[0]
        win_frac = f_bands[f_bands['band'] == f_band]['step'].values[0]/f_bands[f_bands['band'] == f_band]['win'].values[0]

        if n_workers == 1:
            T, B, V, S = sliding_time_array_fk(st, element, tstart=t_start, tend=t_end, win_len=win_len, win_frac=win_frac, 
                                               frqlow=frqlow, frqhigh=frqhigh,
                                               sll_x=sll_x, slm_x=slm_x, sll_y=sll_y, slm_y=slm_y, sl_s=sl_s,
                                               sl_corr=sl_corr, use_geographic_coords=use_geographic_coords)
            T_all.append(T); B_all.append(B); V_all.append(V); S_all.append(S)
        else:
            dask_out = dask.delayed(sliding_time_array_fk)(st, element, tstart=t_start, tend=t_end, 
                                                           win_len=win_len, win_frac=win_frac, 
                                                           frqlow=frqlow, frqhigh=frqhigh,
                                                           sll_x=sll_x, slm_x=slm_x, sll_y=sll_y, 
                                                           slm_y=slm_y, sl_s=sl_s, sl_corr=sl_corr,
                                                           use_geographic_coords=use_geographic_coords)
            dask_all.append(dask_out)

    if n_workers > 1:
        # Organizing output from distributed process:
        out = dask.compute(*dask_all)
        for out_i in out:
            T_all.append(out_i[0]); B_all.append(out_i[1]); V_all.append(out_i[2]); S_all.append(out_i[3])

    # Extracting the time vector corresponding to the maximum number of values:
    N = []
    for T in T_all:
        N.append(len(T))
    T = T_all[np.argmax(N)]    # Times for f-band with highest number of DOA estimates
    
    # Re-sampling array processing results to produce time/frequency matrices:
    NF = len(f_bands)
    NT = len(T)
    B = np.zeros((NF, NT))
    V = np.zeros((NF, NT))
    S = np.zeros((NF, NT))
    for i in range(0, NF):
        T_i = T_all[i]; B_i = B_all[i]; V_i = V_all[i]; S_i = S_all[i]
        for j in range(0, NT):
            ix = np.argmin(np.abs(T[j] - T_i))
            B[i,j] = B_i[ix]
            V[i,j] = V_i[ix]
            S[i,j] = S_i[ix]
    
    if n_workers > 1:
        client.close()

    return T, B, V, S

def sliding_time_array_fk(st, element, tstart=None, tend=None, win_len=20, win_frac=0.5, frqlow=0.5, frqhigh=4,
                          sll_x=-3.6, slm_x=3.6, sll_y=-3.6, slm_y=3.6, sl_s=0.18, sl_corr=[0.,0.],
                          normalize_waveforms=True, use_geographic_coords=True):
    '''
    Processes st with sliding window FK analysis. Default parameters are suitable for most
    regional infrasound arrays.

    Returns a numpy array with timestamp, relative power, absolute power, backazimuth, slowness
    '''

    tr = st.select(station=element)[0]    # Trace of reference element

    # Defining t_start, t_end:
    if (tstart == None) and (tend == None):
        tstart = 1
        tend = (tr.stats.npts * tr.stats.delta)-1
    
    if use_geographic_coords:
        for st_i in st:
            st_i.stats.coordinates = AttribDict({
                'latitude': st_i.stats.sac.stla,
                'elevation': 0.,
                'longitude': st_i.stats.sac.stlo})

    kwargs = dict(
            # slowness grid: X min, X max, Y min, Y max, Slow Step
            sll_x=sll_x, slm_x=slm_x, sll_y=sll_y, slm_y=slm_y, sl_s=sl_s,
            # sliding window properties
            win_len=win_len, win_frac=win_frac,
            # frequency properties
            frqlow=frqlow, frqhigh=frqhigh, prewhiten=0,
            # restrict output
            semb_thres=-1e9, vel_thres=-1e9, timestamp='mlabday',
            stime=tr.stats.starttime+tstart, etime=tr.stats.starttime+tend, verbose=False,
            sl_corr=sl_corr, normalize_waveforms=normalize_waveforms
        )

    slid_fk = array_processing(st, **kwargs)

    # Convert times to seconds after start time of reference element and adjusting to center of time window:
    T = ((slid_fk[:,0] - date2num(tr.stats.starttime.datetime))*86400) + win_len/2

    # Convert backazimuths to degrees from North:
    B = slid_fk[:,3] % 360.

    # Convert slowness to phase velocity:
    V = 1/slid_fk[:,4]

    # Semblance:
    S = slid_fk[:,1]

    return T, B, V, S

def sliding_time_array_lsq(st, X, tstart, tend, twin, overlap):
    '''
    Performs sliding time-window array processing using the least-squares array processing
    method in array_lsq
    
    Inputs:
    st - ObsPy Stream object containing array data
    X - array coordinates
    tstart - Start time for processing (in seconds after Stream start-time)
    tend - End time for processing (in seconds after Stream start-time)
    twin - Time window for array processing (s)
    overlap - Overlap for array processing (s)
    
    Outputs:
    T - Times of array processing estimates (center of time windows) (s)
    V - Phase velocities
    B - Backazimuths
    
    Stephen Arrowsmith (sarrowsmith@smu.edu)
    '''

    time_start = st[0].stats.starttime + tstart
    time_end = time_start+tend

    time_start_i = time_start
    time_end_i = time_start_i+twin

    t = tstart; T = []; V = []; B = []
    while time_end_i < time_end:
        st_win = st.slice(time_start_i, time_end_i)
        vel, baz = array_lsq(st_win, X)
        T.append(t + twin/2); V.append(vel); B.append(baz)
        t = t + overlap
        time_start_i = time_start_i + overlap
        time_end_i = time_end_i + overlap
    T = np.array(T); V = np.array(V); B = np.array(B)
    
    return T, V, B

def array_lsq(st, X):
    '''
    Performs pairwise cross-correlation on each trace in st, and least-squares inversion
    for the slowness vector corresponding to the best-fitting plane wave
    
    Inputs:
    st - ObsPy Stream object containing array data (in a time window suitable for array processing)
    X - [Nx2] NumPy array of array coordinates (in km relative to a reference point)
    
    Outputs:
    baz - Backazimuth (in degrees from North)
    vel - Phase velocity (in km/s)
    
    Notes:
    This function requires that all traces in st begin at the same time (within the sampling interval)
    
    Stephen Arrowsmith (sarrowsmith@smu.edu)
    '''
    
    # Initializing empty arrays for array distances and delay times:
    N = len(st)           # Number of elements
    M = int(N*(N-1)/2)    # Number of pairs of elements
    R = np.zeros((M,2))   # Array to hold relative coordinates between elements
    tau = np.zeros((M,1)) # Array to hold delay times

    k = 0
    for i in range(0,N):
        for j in range(i+1,N):

            tr1 = st[i]; tr2 = st[j]
            C = np.correlate(tr1.data, tr2.data, mode='full')
            lags = np.arange(-np.floor(len(C)/2), np.floor(len(C)/2)+1, 1)*tr1.stats.delta

            # Computing lag corresponding to maximum correlation:
            ix = np.argmax(C); tau[k] = lags[ix]

            # Computing vector of distances between array coordinates:
            R[k,:] = X[i,:] - X[j,:]

            k = k + 1
    
    # Performing least squares inversion:
    R = np.matrix(R); tau = np.matrix(tau)
    u = (inv(np.transpose(R)*R)*np.transpose(R))*tau
    v = 1/np.sqrt(u[0]**2 + u[1]**2)
    azimut = 180 * math.atan2(u[0], u[1]) / math.pi
    baz = (azimut % -360 + 180) % 360
    
    return float(v), float(baz)

def gc_backzimuth(st, evlo, evla):
    '''
    Computes the Great-Circle backazimuth and epicentral distance for an array in st given a known event location
    
    Inputs:
    st - ObsPy Stream object containing array data
    evlo - A float containing the event longitude
    evla - A float containing the event latitude
    
    Outputs:
    baz - Backazimuth (degrees from North)
    dist - Great-circle distance (km)
    '''
    
    g = Geod(ellps='sphere')
    
    a12,a21,dist = g.inv(evlo,evla,st[0].stats.sac.stlo,st[0].stats.sac.stla); dist = dist/1000.
    
    return a21%360., dist

def plot_array_coords(X, stnm, units='km'):
    '''
    Plots the array coordinates for a given array with coordinates X and element names stnm
    '''
    
    if units == 'm':
        X = X*1000
    plt.plot(X[:,0], X[:,1], '.')
    for i in range(0, len(stnm)):
        plt.text(X[i,0], X[i,1], stnm[i])
    if units == 'km':
        plt.xlabel('X (km)')
        plt.ylabel('Y (km)')
    elif units == 'm':
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
    else:
        print('Unrecognized units (Options are "km" and "m")')

def get_array_coords(st, ref_station):
    '''
    Returns the array coordinates for an array, in km with respect to the reference array provided
    
    Inputs:
    st - ObsPy Stream object containing array data
    ref_station - A String containing the name of the reference station
    
    Outputs:
    X - [Nx2] NumPy array of array coordinates in km
    stnm - [Nx1] list of element names
    
    Stephen Arrowsmith (sarrowsmith@smu.edu)
    '''
    
    X = np.zeros((len(st), 2))
    stnm = []
    for i in range(0, len(st)):
        E, N, _, _ = utm.from_latlon(st[i].stats.sac.stla, st[i].stats.sac.stlo)
        X[i,0] = E; X[i,1] = N
        stnm.append(st[i].stats.station)

    # Adjusting to the reference station, and converting to km:
    ref_station_ix = np.where(np.array(stnm) == ref_station)[0][0]    # index of reference station
    X[:,0] = (X[:,0] - X[ref_station_ix,0])
    X[:,1] = (X[:,1] - X[ref_station_ix,1])
    X = X/1000.
    
    return X, stnm

def compute_tshifts(X, baz, vel):
    '''
    Computes the set of time shifts required to align the waveforms for an array and DOA
    
    Inputs:
    X - [Nx2] NumPy array of array coordinates in km
    baz - Backazimuth of plane wave
    vel - Phase velocity of plane wave
    
    Outputs:
    t_shifts - [Nx1] List of time shifts (in seconds)
    '''
    
    # Compute slowness vector:
    sl_y = np.abs(np.sqrt((1/vel**2)/((np.tan(np.deg2rad(baz)))**2+1)))
    sl_x = np.abs(sl_y * np.tan(np.deg2rad(baz)))
    if baz > 180:
        sl_x = -sl_x
    if (baz > 90) and (baz < 270):
        sl_y = -sl_y

    # Computes the time shifts for the slowness vector defined by sl_x, sl_y:
    t_shifts = []
    for i in range(0, X.shape[0]):
        t_shift = X[i,0]*sl_x + X[i,1]*sl_y
        t_shifts.append(t_shift)
        #st_copy[i].stats.starttime = st_copy[i].stats.starttime + t_shift
    
    return t_shifts

def plot_beam(st, int_shifts, ref_station, twin=None, trace_spacing=2., return_beam=False):
    '''
    Plots array beam and waveforms on each array element
    
    Inputs:
    st - ObsPy Stream object containing array data 
    int_shifts - List or NumPy array containing integer time shifts for each element
    ref_station - A String describing the reference station (which all shifts are relative to)
    twin - A list defining the time window for plotting (in seconds from start), e.g., twin=[0,100]
    trace_spacing - A float that controls vertical spacing between traces
    return_beam - Boolean to optionally return the array beam
    
    Notes:
    This function requires that all traces in st begin at the same time (within the sampling interval)
    
    Stephen Arrowsmith (sarrowsmith@smu.edu)
    '''
    
    tr_ref = st.select(station=ref_station)[0]
    
    ix = 0
    for tr in st:

        # Plotting data:
        if int_shifts[ix] < 0:    # Need to truncate data before plotting
            ix_start = np.abs(int_shifts[ix])
            data = tr.data[ix_start::]
        elif int_shifts[ix] > 0:  # Need to add zeros to the beginning before plotting:
            data = np.concatenate((np.zeros((np.abs(int_shifts[ix]))), tr.data))
        else:
            data = tr.data

        # Computing stack:
        if ix == 0:
            stack = data
        else:
            diff = len(stack)-len(data)
            if diff > 0:
                data = np.concatenate((data, np.zeros(diff)))
            elif diff < 0:
                stack = np.concatenate((stack, np.zeros(np.abs(diff))))
            stack = data + stack
        
        # Normalizing data prior to plotting
        data = data/(np.max(np.abs(data)))

        try:
            t_data = np.arange(0, len(data)*tr_ref.stats.delta, tr_ref.stats.delta)
            t_data = t_data[0:len(data)]
            plt.plot(t_data, ix*trace_spacing + data, 'k')
        except:
            pdb.set_trace()
        
        ix = ix + 1

    beam = stack/len(st)
    
    stack = stack/np.max(np.abs(stack))
    plt.plot(np.arange(0, len(data)*tr_ref.stats.delta, tr_ref.stats.delta), ix*trace_spacing + stack, 'r')
    plt.xlabel('Time (s) after ' + str(tr_ref.stats.starttime).split('.')[0].replace('T', ' '))
    plt.gca().get_yaxis().set_ticks([])
    
    if twin is not None:
        plt.gca().set_xlim([twin[0],twin[1]])
    
    if return_beam:
        return beam

def add_beam_to_stream(st, beam, ref_station):
    '''
    Adds the beam channel to an ObsPy Stream using the time of the reference station
    
    Inputs:
    st - ObsPy Stream object containing array data
    beam - NumPy array containing beam data
    ref_station - String containing the name of the reference station
    
    Outputs:
    st - ObsPy Stream object including beam channel with station name = 'BEAM'
    '''
    
    # Obtain trace for reference station:
    tr = st.select(station=ref_station)[0].copy()
    tr.data = beam[0:len(tr.data)]
    tr.stats.station = 'BEAM'
    st.append(tr)
    
    return st

def plotFK(st, startTime, endTime, frqlow, frqhigh,
           sll_x=-3.6, slm_x=3.6, sll_y=-3.6, slm_y=3.6, sl_s=0.18,
           plot=True, normalize=True, sl_corr=[0.,0.], show_peak=False,
           cmap='viridis'):
    '''
    Computes and displays an FK plot for an ObsPy Stream object, st, given
    a start time and end time (as UTCDateTime objects) and a frequency band
    defined by frqlow and frqhigh. The slowness grid is defined as optional
    parameters (in s/km).

    This function implements code directly from ObsPy, which has been optimized,
    for simply plotting the FK spectrum

    It includes the option to normalize the data in the time window before running FK

    It also includes the option to apply a slowness correction, defined by sl_corr
    '''

    stream = st.copy()
    stream = stream.trim(startTime, endTime)

    if normalize:
        for st_i in stream:
            st_i.data = st_i.data/np.max(np.abs(st_i.data))
    
    for st_i in stream:
        st_i.stats.coordinates = AttribDict({
            'latitude': st_i.stats.sac.stla,
            'elevation': st_i.stats.sac.stel,
            'longitude': st_i.stats.sac.stlo})

    verbose = False
    coordsys = 'lonlat'
    method = 0

    prewhiten = 0

    grdpts_x = int(((slm_x - sll_x) / sl_s + 0.5) + 1)
    grdpts_y = int(((slm_y - sll_y) / sl_s + 0.5) + 1)

    geometry = get_geometry(stream, coordsys=coordsys, verbose=verbose)

    time_shift_table = get_timeshift(geometry, sll_x, sll_y,
                                     sl_s, grdpts_x, grdpts_y)
    nstat = len(stream)
    fs = stream[0].stats.sampling_rate
    nsamp = stream[0].stats.npts

    # generate plan for rfftr
    nfft = next_pow_2(nsamp)
    deltaf = fs / float(nfft)
    nlow = int(frqlow / float(deltaf) + 0.5)
    nhigh = int(frqhigh / float(deltaf) + 0.5)
    nlow = max(1, nlow)  # avoid using the offset
    nhigh = min(nfft // 2 - 1, nhigh)  # avoid using nyquist
    nf = nhigh - nlow + 1  # include upper and lower frequency

    # to speed up the routine a bit we estimate all steering vectors in advance
    steer = np.empty((nf, grdpts_x, grdpts_y, nstat), dtype=np.complex128)
    clibsignal.calcSteer(nstat, grdpts_x, grdpts_y, nf, nlow,
                         deltaf, time_shift_table, steer)
    _r = np.empty((nf, nstat, nstat), dtype=np.complex128)
    ft = np.empty((nstat, nf), dtype=np.complex128)

    # 0.22 matches 0.2 of historical C bbfk.c
    tap = cosine_taper(nsamp, p=0.22)
    relpow_map = np.empty((grdpts_x, grdpts_y), dtype=np.float64)
    abspow_map = np.empty((grdpts_x, grdpts_y), dtype=np.float64)

    for i, tr in enumerate(stream):
        dat = tr.data
        dat = (dat - dat.mean()) * tap
        ft[i, :] = np.fft.rfft(dat, nfft)[nlow:nlow + nf]

    ft = np.ascontiguousarray(ft, np.complex128)
    relpow_map.fill(0.)
    abspow_map.fill(0.)

    # computing the covariances of the signal at different receivers
    dpow = 0.
    for i in range(nstat):
        for j in range(i, nstat):
            _r[:, i, j] = ft[i, :] * ft[j, :].conj()
            if i != j:
                _r[:, j, i] = _r[:, i, j].conjugate()
            else:
                dpow += np.abs(_r[:, i, j].sum())
    dpow *= nstat

    clibsignal.generalizedBeamformer(
        relpow_map, abspow_map, steer, _r, nstat, prewhiten,
        grdpts_x, grdpts_y, nf, dpow, method)

    ix, iy = np.unravel_index(relpow_map.argmax(), relpow_map.shape)

    # here we compute baz, slow
    slow_x = sll_x + ix * sl_s
    slow_y = sll_y + iy * sl_s

    # ---------
    slow_x = slow_x - sl_corr[0]
    slow_y = slow_y - sl_corr[1]
    #print(slow_x, slow_y)
    # ---------

    slow = np.sqrt(slow_x ** 2 + slow_y ** 2)
    if slow < 1e-8:
        slow = 1e-8
    azimut = 180 * math.atan2(slow_x, slow_y) / math.pi
    baz = azimut % -360 + 180

    if plot:
        plt.pcolormesh(np.arange(sll_x, slm_x + sl_s, sl_s)+sl_corr[0],
                       np.arange(sll_x, slm_x + sl_s, sl_s)+sl_corr[1],
                       np.flipud(np.fliplr(relpow_map.transpose())),
                       cmap=cmap)
        plt.xlim(sll_x,slm_x)
        plt.ylim(sll_y,slm_y)
        plt.plot(0, 0, 'w+')
        if show_peak:
            plt.plot(-slow_x, -slow_y, 'w*')
        plt.xlabel('Slowness (x) s/km')
        plt.ylabel('Slowness (y) s/km')
        plt.title('Peak semblance at ' + str(round(baz % 360., 2)) + ' degrees ' + str(round(1/slow, 2)) + ' km/s')

    # only flipping left-right, when using imshow to plot the matrix is takes points top to bottom
    # points are now starting at top-left in row major
    return np.fliplr(relpow_map.transpose()), baz % 360, 1. / slow