# Cardinal: Seismic and Geoacoustic Array Processing
#
# Cardinal Algorithm:
# 1 - Segmentor
# 2 - Adaptive Array
# 3 - Array Processor
# 4 - Aggregator
#
# *Miro Ronac Giannone (mronacgiannone@smu.edu, mnronac@sandia.gov), Stephen Arrowsmith (sarrowsmith@smu.edu), Jonathan Reiter (jyreiter@smu.edu)
# *Contact for errors/bugs
# (https://github.com/sjarrowsmith/cardinal.git)

import itertools, warnings, dask, pickle, os, sqlite3, io, utm, random, cartopy, re, pywt, bisect, logging, psutil, gc

# Hide non-critical warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
logging.getLogger('distributed.nanny').setLevel(logging.CRITICAL)

# Import packages as
import numpy as np
import pandas as pd
import networkx as nx
import cartopy.crs as ccrs
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import datetime as datetime_module

# Import functions from packages
from obspy import *
from pyproj import Geod
from obspy.core import *
from scipy import signal
from array_analysis import *
from numpy.linalg import inv
from datetime import datetime
from collections import Counter 
from scipy.stats import linregress
from scipy.spatial import KDTree
from pisces.tables.css3 import Wfdisc
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
from scipy.integrate import cumulative_trapezoid
from matplotlib.collections import LineCollection
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize

# Set matplotlib epoch
from matplotlib.dates import date2num, num2date, set_epoch
set_epoch('0000-12-31T00:00:00') # Using the original matplotlib epoch

# Machine Learning Packages
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler, QuantileTransformer

import tensorflow as tf
from keras.utils import *
from tensorflow.keras import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------'
'---------------------------------------------------------------------------'
'----------------------------------------------'
############### ############### ###############
############## Global Variables ###############
############### ############### ###############

# Define Geod for Great Circle computations
g = Geod(ellps='sphere')

# Directories for RC ConvFormer model and scalers (seismic input, infrasound input, frequency bands, seismic RC, infrasound RC)
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, 'Model_and_Scalers')
RC_ConvFormer_dir = os.path.join(model_dir, 'RC_Model.keras')
X_infrasound_scaler_dir = os.path.join(model_dir, 'X_infrasound_scaler.pkl')
X_seismic_scaler_dir = os.path.join(model_dir, 'X_seismic_scaler.pkl')
F_scaler_dir = os.path.join(model_dir, 'F_scaler.pkl')
y_infrasound_scaler_dir = os.path.join(model_dir, 'y_infrasound_scaler.pkl')
y_seismic_scaler_dir = os.path.join(model_dir, 'y_seismic_scaler.pkl')
'----------------------------------------------'
'---------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------'
'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------'
'---------------------------------------------------------------------------'
'----------------------------------------------'
############### ############### ###############
############# Ancillary Functions #############
############### ############### ###############

def read_wfdisc(wfdisc_file):
    '''---------------------------------------------------------------------------------------------------------
    Reads wfdidc file and returns ObsPy stream (uses pisces 0.4.3)
    Make sure wfdisc and mwf files are in the same directory
    
    Input:
        wfdisc_file (str): path to wfdisc file

    Output:
        st: ObsPy stream object
    ---------------------------------------------------------------------------------------------------------'''
    tr_list = []
    with open(wfdisc_file, 'r') as f:
        for row in f:
            sta = row[0:6].strip()
            chan = row[7:15].strip()
            time = float(row[16:33].strip())
            wfid = int(row[34:42].strip())
            chanid = int(row[43:51].strip())
            jdate = int(row[52:60].strip())
            endtime = float(row[61:78].strip())
            nsamp = int(row[79:87].strip())
            samprate = float(row[88:99].strip())
            calib = float(row[100:116].strip())
            calper = row[117:133].strip()
            instype = row[134:140].strip()
            segtype = row[141:142].strip()
            datatype = row[143:145].strip()
            clip = row[146:147].strip()
            dir = row[148:212].strip()
            dir = os.path.dirname(wfdisc_file) + '/'
            dfile = row[213:245].strip()
            foff = int(row[246:256].strip())
            commid = int(row[257:265].strip())
            #--- Changed lddate computation
            # lddate = UTCDateTime(row[266:283].strip())
            jdate_str = str(jdate)
            lddate = UTCDateTime(jdate_str[0:4] + '-' + jdate_str[4:])
            wf = Wfdisc(
                calib, calper, chan, chanid, clip, commid, datatype,
                dfile, dir, endtime, foff, instype, jdate, lddate,
                nsamp, samprate, segtype, sta, time, wfid
                )
            tr = wf.to_trace()
            tr_list.append(tr)
            # break
    st = Stream(tr_list)

    return st

'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def fix_lengths(t, y):
    '''---------------------------------------------------------------------------------------------------------
    Fixes differences in lengths between time vector and data
    
    Input:
        t (array): sequence of time points corresponding to the data values in a time-series
        y (array): data values from time-series

    Output:
        t_out (array): time vector with same length as data
        y_out (array): data values with same length as time vector
    ---------------------------------------------------------------------------------------------------------'''
    t_out = t.copy()
    y_out = y.copy()

    if len(t_out) > len(y_out):
        t_out = t_out[0:len(y_out)]
    elif len(y_out) > len(t_out):
        y_out = y_out[0:len(y_out)]

    return t_out, y_out

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def data_time_window(t, st_data, t_start, t_end):
    '''---------------------------------------------------------------------------------------------------------
    Extracts array data within defined time window [t_start, t_end]

    Inputs:
        t (array): sequence of time points corresponding to the data values in a time-series
        st_data: ObsPy stream object or ObsPy trace object or array data (must be NumPy array where each row contains trace data)
        t_start (int/float): start time of time window (in s)
        t_end (int/float): end time of time window (in s)
    ---------------------------------------------------------------------------------------------------------'''
    ix = np.where((t_start <= t) & (t < t_end))[0]
    if type(st_data) != np.ndarray:
        data = []
        if len(np.array(st_data).shape) > 1:
            for i in range(len(st_data)):
                data.append(st_data[i].data[ix])
        else:
            data.append(st_data.data[ix])
        data = np.array(data)
    elif type(st_data) == np.ndarray:
        data = np.zeros((st_data.shape[0],len(ix)))
        for i in range(st_data.shape[0]):
            data[i,:] = st_data[i,ix]
    else:
        raise Exception('st_data must be ObsPy stream/trace object or NumPy array.')
    t_data = t[ix] - np.min(t[ix])
    
    return t_data, data

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def get_SNR(data_s, data_n):
    '''---------------------------------------------------------------------------------------------------------
    Computes signal-to-noise ratio

    Inputs:
        data_s (array): signal data
        data_n (array): noise data

    Outputs:
        snr (int/float): signal-to-noise ratio between data_s and data_n computed using root-mean-square
    ---------------------------------------------------------------------------------------------------------'''
    rms1 = np.sqrt(np.mean(data_s**2))
    rms2 = np.sqrt(np.mean(data_n**2))
    snr = 10*np.log((rms1/rms2)**2)
    
    return snr

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def power_spectrum(tr, t_start=None, t_end=None, window='hann', nperseg=2**8, noverlap_percent=50):
    '''---------------------------------------------------------------------------------------------------------
    Computes a power spectrum for a given trace and start/end times using SciPy's signal.welch() function

    Inputs:
        tr: ObsyPy trace object
        t_start (int/float): start time of time window (in s)
        t_end (int/float): end time of time window (in s)
        window (str): desired window to use (defaults to hann)
        nperseg (int): length of each segment (defaults to 256 - same as scipy.welch())
        noverlap_percent (int): percent overlap between segments (defaults to 50% same as scipy.signal.welch())
    ---------------------------------------------------------------------------------------------------------'''
    noverlap = (nperseg) * (noverlap_percent/100)
    if t_start is not None:
        if t_end == None:
            raise Exception('End time must be specified if start time is defined.')
        t = np.arange(0, tr.stats.npts*tr.stats.delta, tr.stats.delta)
        _, data = data_time_window(t, tr, t_start, t_end)
        f, Pxx = signal.welch(data, tr.stats.sampling_rate, window=window, nperseg=nperseg, noverlap=noverlap)  
    else:
        data = tr.data
        f, Pxx = signal.welch(data, tr.stats.sampling_rate, window=window, nperseg=nperseg, noverlap=noverlap)

    return f, Pxx

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def get_array_coords(st, ref_station, units='m'):
    '''---------------------------------------------------------------------------------------------------------
    Calculates array coordinates relative to a reference station
    
    Input:
        st: ObsPy Stream object
        ref_station (str): name of reference station
        units (str): scale of array - options are "m" for meters or "km" for kilometers
    
    Output:
        X (array): array coordinates in the specified scale ("m" or "km")
        stnm (list): list of element/station names
    ---------------------------------------------------------------------------------------------------------'''
    X = np.zeros((len(st), 2))
    stnm = []
    for i in range(0, len(st)):
        E, N, _, _ = utm.from_latlon(st[i].stats.sac.stla, st[i].stats.sac.stlo)
        X[i,0] = E; X[i,1] = N
        stnm.append(st[i].stats.station)
    #-----------------------------------------------------------------------------------------------------------------#
    # Adjusting to the reference station, and converting to km:
    ref_station_ix = np.where(np.array(stnm) == ref_station)[0][0]    # index of reference station
    X[:,0] = (X[:,0] - X[ref_station_ix,0])
    X[:,1] = (X[:,1] - X[ref_station_ix,1])
    if units == 'km':
        X = X/1000

    return X, stnm 

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def convert_to_slowness(baz, vel):
    '''---------------------------------------------------------------------------------------------------------
    Converts back azimuth - trace velocity pair to a 2-d slowness vector
    
    Input:
        baz (float/int): back azimuth
        vel (float/int): trace velocity

    Output:
        sl_x (float): slowness along the x-axis
        sl_y (float): slowness along the y-axis
    ---------------------------------------------------------------------------------------------------------'''
    sl_y = np.abs(np.sqrt((1/vel**2)/((np.tan(np.deg2rad(baz)))**2+1)))
    sl_x = np.abs(sl_y * np.tan(np.deg2rad(baz)))
    if baz > 180:
        sl_x = -sl_x
    if (baz > 90) and (baz < 270):
        sl_y = -sl_y
    
    return sl_x, sl_y

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def adjust_times_for_slowness(st, X, sl_x, sl_y):
    '''---------------------------------------------------------------------------------------------------------
    Adjusts start times in an ObsPy stream given a slowness vector
    
    Input:
        st: Obspy stream object
        X (array): array coordinates relative to a reference station
        sl_x (float): slowness along the x-axis
        sl_y (float): slowness along the y-axis

    Output:
        st: ObsPy stream with adjusted start times
        t_shifts (array): time shifts corresponding to slowness vector
    ---------------------------------------------------------------------------------------------------------'''
    st_sc = st.copy(); t_shifts = []
    for i in range(0, X.shape[0]):
        t_shift = X[i,0]*sl_x + X[i,1]*sl_y
        st_sc[i].stats.starttime = st_sc[i].stats.starttime + t_shift
        t_shifts.append(t_shift)
    t_shifts = np.array(t_shifts)

    return st_sc, t_shifts

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def get_slowness_vector_time_shifts(st, ref_station, baz, vel, units='m'):
    '''---------------------------------------------------------------------------------------------------------
    Get time shifts for a back azimuth - trace velocity pair
    
    Input:
        st: Obspy stream object
        ref_station (str): name of reference station
        baz (float/int): back azimuth
        tr_vel (float/int): trace velocity

    Output:
        t_shifts (array): time shifts corresponding to back azimuth - trace velocity pair
    ---------------------------------------------------------------------------------------------------------'''
    # Computing array coordinates
    X, _ = get_array_coords(st, ref_station, units=units)
    #-----------------------------------------------------------------------------------------------------------------#
    # Computing the slowness vector for a specified backazimuth and trace velocity
    sl_x, sl_y = convert_to_slowness(baz, vel)
    #-----------------------------------------------------------------------------------------------------------------#
    # Computing time shifts for the slowness vector defined by sl_x, sl_y
    _, t_shifts = adjust_times_for_slowness(st, X, sl_x, sl_y)

    return t_shifts
    
'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def gc_backazimuth(st, evlo, evla):
    '''---------------------------------------------------------------------------------------------------------
    Computes Great-Circle back azimuth and epicentral distance for an array in st given a known event location
    
    Input:
        st: ObsPy Stream object
        evlo (float): event longitude
        evla (float): event latitude

    Output:
        baz (float): back azimuth (degrees from North)
        dist (float): Great-circle distance (km)
    ---------------------------------------------------------------------------------------------------------'''
    _, a21, dist = g.inv(evlo,evla,st[0].stats.sac.stlo,st[0].stats.sac.stla); dist = dist/1000.
    
    return a21%360., dist

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def add_beam_to_stream(st, beam):
    '''---------------------------------------------------------------------------------------------------------
    Adds beamformed data channel to an ObsPy stream using the time of the reference station 
    
    Input:
        st: ObsPy stream object
        beam (array): NumPy array containing beam data
        ref_station (str): name of reference station

    Output:
        st: ObsPy stream object including beam channel with station name = 'Beam'
    ---------------------------------------------------------------------------------------------------------'''
    # Obtain trace for reference station:
    st_beam = st.copy()
    tr = st_beam.select(station=st_beam[0].stats.station)[0].copy()
    tr.data = beam[0:len(tr.data)]
    tr.stats.station = 'Beam'
    st_beam.append(tr)
    
    return st_beam

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def get_geometry(stream, coordsys='lonlat', return_center=False):
    '''---------------------------------------------------------------------------------------------------------
    Method to calculate the array geometry and the center coordinates in km (*ObsPy code modified for sac files)

    Input:
        stream: ObsPy stream object
        coordsys (str): choose which stream attributes to use for coordinates - options are "lonlat" or "xy"
        return_center (boolean): returns the center coordinates as extra tuple if True
    
    Output:
        geometry (array): geometry of the stations as 2d NumPy array
    ---------------------------------------------------------------------------------------------------------'''
    nstat = len(stream)
    center_lat = 0.
    center_lon = 0.
    center_h = 0.
    geometry = np.empty((nstat, 3))

    if isinstance(stream, Stream):
        for i, tr in enumerate(stream):
            if coordsys == 'lonlat':
                try:
                    geometry[i, 0] = tr.stats.sac.stlo
                    geometry[i, 1] = tr.stats.sac.stla
                    geometry[i, 2] = tr.stats.sac.stel
                except:
                    geometry[i, 0] = tr.stats.coordinates.stlo
                    geometry[i, 1] = tr.stats.coordinates.stla
                    geometry[i, 2] = tr.stats.coordinates.stel                    
            elif coordsys == 'xy':
                try:
                    geometry[i, 0] = tr.stats.sac.x
                    geometry[i, 1] = tr.stats.sac.y
                    geometry[i, 2] = tr.stats.sac.z
                except:
                    geometry[i, 0] = tr.stats.coordinates.x
                    geometry[i, 1] = tr.stats.coordinates.y
                    geometry[i, 2] = tr.stats.coordinates.z                  
    elif isinstance(stream, np.ndarray):
        geometry = stream.copy()
    else:
        raise TypeError('only Stream or numpy.ndarray allowed')

    if coordsys == 'lonlat':
        center_lon = geometry[:, 0].mean()
        center_lat = geometry[:, 1].mean()
        center_h = geometry[:, 2].mean()
        for i in np.arange(nstat):
            x, y = util_geo_km(center_lon, center_lat, geometry[i, 0],
                               geometry[i, 1])
            geometry[i, 0] = x
            geometry[i, 1] = y
            geometry[i, 2] -= center_h
    elif coordsys == 'xy':
        geometry[:, 0] -= geometry[:, 0].mean()
        geometry[:, 1] -= geometry[:, 1].mean()
        geometry[:, 2] -= geometry[:, 2].mean()
    else:
        raise ValueError("Coordsys must be one of 'lonlat', 'xy'")

    if return_center:
        return np.c_[geometry.T,
                     np.array((center_lon, center_lat, center_h))].T
    else:
        return geometry

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'


def make_synthetic_array(num_stns=10, small_arr_extent=0.2, large_arr_extent=5, randomize=True):
    '''---------------------------------------------------------------------------------------------------------
    Generates synthetic multi-scale array with elements randomly distributed across 2 different scaled grids

    Input:
        num_stns (int): number of stations contained in the synthetic array
        small_arr_extent (float/int): extent of small-scale grid (central elements)
        large_arr_extent (float/int): extent of large-scale grid (outer elements)
        randomize (boolean): whether to randomly shuffle array elements
    
    Output:
        new_geometry (array): geometry of synthetic array (in m)
        relative_geometry (array): geometry of synthetic array relative to a randomly selected reference element (in m)
        new_arr_labels (list): list of element names (numeric values = central elements and alphabetic values = outer elements)
    ---------------------------------------------------------------------------------------------------------'''
    # Make small array
    x_extent_small = small_arr_extent/2; y_extent_small = x_extent_small
    x_small = [random.uniform(-x_extent_small, x_extent_small) for _ in range(num_stns//2)]
    y_small = [random.uniform(-y_extent_small, y_extent_small) for _ in range(num_stns//2)]
    geometry_small = np.hstack((x_small, y_small)).reshape(len(x_small),2)
    #-----------------------------------------------------------------------------------------------------------------#
    # Make large array
    x_extent_large = large_arr_extent/2; y_extent_large = x_extent_large
    x_large = [random.uniform(-x_extent_large, x_extent_large) for _ in range(num_stns//2)]
    y_large = [random.uniform(-y_extent_large, y_extent_large) for _ in range(num_stns//2)]
    geometry_large = np.hstack((x_large, y_large)).reshape(len(x_large),2)
    #-----------------------------------------------------------------------------------------------------------------#
    # Combine arrays
    geometry = np.vstack((geometry_small, geometry_large))
    synthetic_arr_labels = np.array(['S1', 'S2', 'S3', 'S4', 'S5', 'SA', 'SB', 'SC', 'SD', 'SE']) # numeric = small...alphabetic = large
    rand_num_stns = random.sample(range(3, len(synthetic_arr_labels)-2), 1)[0] # randomly choose number of elements to incorporate
    rand_num_stns_idxs = random.sample(range(len(synthetic_arr_labels)), rand_num_stns)
    #-----------------------------------------------------------------------------------------------------------------#
    # Construct subarray
    if randomize == True:
        new_geometry = geometry[rand_num_stns_idxs,:]
        new_arr_labels = synthetic_arr_labels[rand_num_stns_idxs]
    else:
        new_geometry = geometry.copy()
        new_arr_labels = synthetic_arr_labels.copy()
    #-----------------------------------------------------------------------------------------------------------------#
    # Relative coordinates based on randomly chosen reference station
    rand_stn_idx = random.sample(range(new_geometry.shape[0]), 1)[0]
    ref_station = new_arr_labels[rand_stn_idx]
    ref_station_ix = np.where(new_arr_labels == ref_station)[0][0]
    relative_geometry = np.zeros((new_geometry.shape))
    relative_geometry[:,0] = (new_geometry[:,0] - new_geometry[ref_station_ix,0])
    relative_geometry[:,1] = (new_geometry[:,1] - new_geometry[ref_station_ix,1])
    new_geometry *= 1000; relative_geometry *= 1000 # convert to meters
    #-----------------------------------------------------------------------------------------------------------------#
    # Return geometry, relative geometry, and labels
    return new_geometry, relative_geometry, new_arr_labels

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

def find_optimal_clusters(linked):
    '''---------------------------------------------------------------------------------------------------------
    Finds optimal number of clusters from dendrogram to be used for Adaptive Array

    Input:
        linked (array): hierarchical clustering encoded as a linkage matrix

    Output:
        optimal_clusters (int): suggested number of clusters
        distances (float): distance to cut dendrogram
    ---------------------------------------------------------------------------------------------------------'''
    # Get distances at which clusters are merged
    distances = linked[:, 2]

    # Calculate successive differences (gaps)
    diffs = np.diff(distances)

    # Find index of largest gap
    jump_index = np.argmax(diffs)

    # Estimate the number of clusters: total merges - index
    optimal_clusters = len(distances) - jump_index

    return optimal_clusters, distances[jump_index]

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def element_pair_distances(st, plot=False, grid_linewidth=0.5, return_params=False, figsize=(12,6)):
    '''---------------------------------------------------------------------------------------------------------
    Calculates distance (in km) between each pair of elements in an array and plots dendrogram.
    Used to determine number of clusters for adaptive array.

    Input:
        st: ObsPy stream object
        plot (boolean): whether to plot each element-pair distance
        grid_linewidth (float): if plot, defines the linewidth of the grid
        return_params (boolean): specifies whether outputs are returned (defaults to False)
        figsize (tuple): if plot, defines the size of the figure

    Output:
        stn_dist_sort (array): element-pair distances sorted from min to max (in km)
        pairs (list): list of station pairs
    ---------------------------------------------------------------------------------------------------------'''
    # X = get_geometry(st)
    X, stnm = get_array_coords(st, st[0].stats.station, units='km')
    if X.shape[1] == 3:
        X = X[:,:-1]
    M = int(len(st) * (len(st) - 1) / 2)
    G = np.zeros((M, 2))
    stn_dist = np.zeros((M, 1)) # interstation distance between element pairs
    k = 0
    for i in range(0, len(st)):
        for j in range(i+1, len(st)):
            G[k,:] = X[i,:] - X[j,:]
            stn_dist[k,:] = np.linalg.norm(G[k,:])
            k += 1
    #--------------------------------------------------------------------------------#
    # Sorting data based on interstation distance
    ix = np.argsort(stn_dist, axis=0)
    stn_dist_sort = stn_dist[ix].flatten()
    #--------------------------------------------------------------------------------#
    x = np.arange(1, M+1, 1) # element pairs
    pair = []
    k = 0
    for i in range(0, len(st)):
        for j in range(i+1, len(st)):
            pair.append(st[i].stats.station)
            pair.append(st[j].stats.station)
            k += 1
    pairs = np.array(pair)
    pairs = np.array_split(pairs, M)
    pairs = np.flipud(pairs)
    #--------------------------------------------------------------------------------#
    if plot:
        fig = plt.figure(figsize=figsize)
        # Plot dendrogram
        ax1 = fig.add_subplot(1,2,1)
        linked = linkage(pdist(X), method='ward')
        k, cut_distance = find_optimal_clusters(linked)
        dendrogram(linked,
                   labels=stnm,
                   orientation='top',
                   color_threshold=cut_distance,
                   distance_sort='ascending',
                   show_leaf_counts=True,
                   leaf_font_size=8.5)
        plt.axhline(cut_distance, color='red', linestyle='--')
        plt.grid(linewidth=grid_linewidth)
        plt.xlabel("Sensor"); plt.ylabel('Distance [km]')
        plt.title("Hierarchical Clustering Dendrogram")
        print(f"Suggested number of clusters: {k}")
        print(f"Distance to cut dendrogram: {cut_distance:.2f}")
        # Plot element pair distances first
        ax2 = fig.add_subplot(1,2,2, sharey=ax1)
        ax2.plot(x.astype(int), stn_dist_sort, 'o', color='k')
        plt.xlabel('Element Pair')
        plt.grid(linewidth=grid_linewidth)
        plt.title('Element Pair Distance')
    if return_params == True:
        return stn_dist_sort, pairs

'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def pad_geometry(geometry, max_stns=10):
    '''---------------------------------------------------------------------------------------------------------
    Center pads geometry using zeros

    Input:
        geometry (array): geometry of array
        max_stns (int): length of padded geometry

    Output:
        geometry (array): center padded geometry with length = max_stns
    ---------------------------------------------------------------------------------------------------------'''
    # Center Pad
    pad_len = max_stns - len(geometry)
    left_pad = pad_len // 2
    right_pad = pad_len - left_pad
    geometry = np.pad(geometry, ((left_pad, right_pad),(0,0)), mode='constant')

    return geometry

'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def get_subarray_geometry(subarray, stnm, X):
    '''---------------------------------------------------------------------------------------------------------
    Retrieves subarray characteristics from original array, including station names, relative coordinates, and aperture (max_dist)

    Input:
        subarray (list): station names in subarray
        stnm (list): station names in entire array
        X (array): relative coordinates for elements in entire array

    Output:
        subarray (list): station names in subarray
        subarray_X (array): relative coordinates for elements in subarray
        max_dist (float): subarray aperture
    ---------------------------------------------------------------------------------------------------------'''
    # Retrieving relative coordinates for subarray
    stnm_new = np.array(stnm).reshape(len(X),1)
    stnm_X = np.hstack((stnm_new,X))
    subarray_idx = np.zeros((len(subarray),1))
    for idx in range(len(subarray)):    
        idx_tmp = np.where((subarray[idx] == stnm_X[:,0]))[0][0]
        subarray_idx[idx] = idx_tmp
    subarray_X = np.zeros((len(subarray_idx),2))
    for ii, idx in enumerate(subarray_idx):
        subarray_X[ii,:] = X[int(idx[0]),:]
    #-----------------------------------------------------------------------------------------------------------------#
    # Calculating max distance between all interstation pairs in subarray
    dists = []
    for ii in range(0, len(subarray_X)):
        for jj in range(ii+1, len(subarray_X)):
            dists.append(np.sqrt((subarray_X[jj,0] - subarray_X[ii,0])**2 + (subarray_X[jj,1] - subarray_X[ii,1])**2))
    dists = np.array(dists)
    max_dist = dists.max()

    return subarray, subarray_X, max_dist

'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def array_response(x, y, c_app=280, c_steps=50, freqmin=1, freqmax=2, freqsteps=50, px_0=0, py_0=0):
    '''---------------------------------------------------------------------------------------------------------
    Calculate array response on a square slowness grid for an arbitrary array of N elements

    Input:
        x (array): x-points in array
        y (array): y-points in array
        c_app (float/int): apparent velocity used to construct extent of slowness grid
        c_steps (int): define resolution of slowness grid
        freqmin (int): minimum frequency (Hz)
        freqmax (int): maximum frequency (Hz)
        freqsteps (int): frequency resolution
        px_0, py_0 (float/int): coordinates which define slowness correction

    Output:
        resp_norm[::-1] (array): response function map
        p_x (array): x-component slowness
        p_y (array): y-component slowness
        resp.max(): array gain
    ---------------------------------------------------------------------------------------------------------'''
    # Construct slowness square grid
    s_max = 1 / c_app 
    px = np.linspace(-s_max, s_max, c_steps)
    py = np.linspace(-s_max, s_max, c_steps)
    px, py = np.meshgrid(px, py)
    #-----------------------------------------------------------------------------------------------------------------#
    # Calculate each part
    i = 1j
    omega = 2 * np.pi * np.linspace(freqmin, freqmax, freqsteps)
    p_r_product = ((px[..., np.newaxis] + px_0) * np.array(x) + (py[..., np.newaxis] + py_0) * np.array(y))
    complex = -i * omega * p_r_product[..., np.newaxis]
    #-----------------------------------------------------------------------------------------------------------------#
    # Compile
    resp = np.sum(np.abs(np.sum(np.exp(complex), 2))**2, 2)
    resp_norm = resp / resp.max()
    
    return resp_norm[::-1], px, py, resp.max()

'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def scale_padded_dataset(X_train, X_test):
    '''---------------------------------------------------------------------------------------------------------
    Scales dataset using a Quantile Transform (normal distribution) fit on the training data

    Input:
        X_train (array): input feature vector training dataset
        X_test (array): input feature vector testing dataset

    Output:
        X_train_scaled (array): train set scaled
        X_test_scaled (array): test set scaled by scaler fit to train set (to prevent data leakage)
        train_scaler: Quantile Transform scaler fit to training dataset
    ---------------------------------------------------------------------------------------------------------'''
    # Create nonzero masks
    train_nonzero_mask = X_train != 0
    test_nonzero_mask = X_test != 0
    train_nonzero_values = X_train[train_nonzero_mask]
    test_nonzero_values = X_test[test_nonzero_mask]
    #-----------------------------------------------------------------------------------------------------------------#
    # Fit to train set and transform train/test set
    train_scaler = QuantileTransformer(output_distribution='normal').fit(train_nonzero_values.reshape(-1,1))
    train_nonzero_values_scaled = train_scaler.transform(train_nonzero_values.reshape(-1,1)).flatten()
    test_nonzero_values_scaled = train_scaler.transform(test_nonzero_values.reshape(-1,1)).flatten()
    #-----------------------------------------------------------------------------------------------------------------#
    # Reconstruct original arrays with scaled nonzero values
    X_train_scaled = np.zeros_like(X_train, dtype=float)
    X_test_scaled = np.zeros_like(X_test, dtype=float)
    X_train_scaled[train_nonzero_mask] = train_nonzero_values_scaled
    X_test_scaled[test_nonzero_mask] = test_nonzero_values_scaled
    
    return X_train_scaled, X_test_scaled, train_scaler

'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def append_location_info(st, array_coords_filepath):
    '''---------------------------------------------------------------------------------------------------------
    Appends location info (lat/lon/elev) to each trace in ObsPy stream

    Input:
        st: ObsPy stream object
        array_coords_filepath (str): filepath to site file containing 4 columns - station names, station lats, station lons, and station elevations

    Output:
        st: ObsPy stream object with sac files providing station location information
    ---------------------------------------------------------------------------------------------------------'''
    # Appending location information
    try:
        df = pd.read_table(array_coords_filepath, header=None, sep='\s+', names=['Stn', 'Lat', 'Lon', 'Elev']) # array coordinates
    except:
        df = pd.read_table(array_coords_filepath, header=None, sep='\s+', names=['Stn', 'Lat', 'Lon']) # array coordinates
    if len(df.columns) == 4:
        for tr in st:
            index = np.where(df['Stn'] == tr.stats.station)[0]
            sacAttrib = AttribDict({'stla': df['Lat'][index],
                                    'stlo': df['Lon'][index],
                                'stel': df['Elev'][index]})
            tr.stats.sac = sacAttrib
        for tr in st:
            try:
                lat = (df[df['Stn'] == tr.stats.station]['Lat']).values[0]
                lon = (df[df['Stn'] == tr.stats.station]['Lon']).values[0]
                elev = (df[df['Stn'] == tr.stats.station]['Elev']).values[0]
            except:
                lat = (df[df['Stn'] == tr.id]['Lat']).values[0]
                lon = (df[df['Stn'] == tr.id]['Lon']).values[0]
                elev = (df[df['Stn'] == tr.id]['Elev']).values[0]
            tr.stats.sac.stla = lat
            tr.stats.sac.stlo = lon
            tr.stats.sac.stel = elev
    elif len(df.columns) == 3:
        for tr in st:
            index = np.where(df['Stn'] == tr.stats.station)[0]
            sacAttrib = AttribDict({'stla': df['Lat'][index],
                                    'stlo': df['Lon'][index]})
            tr.stats.sac = sacAttrib
        for tr in st:
            try:
                lat = (df[df['Stn'] == tr.stats.station]['Lat']).values[0]
                lon = (df[df['Stn'] == tr.stats.station]['Lon']).values[0]
            except:
                lat = (df[df['Stn'] == tr.id]['Lat']).values[0]
                lon = (df[df['Stn'] == tr.id]['Lon']).values[0]
            tr.stats.sac.stla = lat
            tr.stats.sac.stlo = lon
    else:
        raise Exception('Array coordinates table must have either 3 or 4 columns, depending if elevation is provided.')
    
    return st

'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def dbscan_outliers(X, eps=1.0, min_samples=2):
    '''---------------------------------------------------------------------------------------------------------
    Detects station outliers within a subarray using DBSCAN.

    Input:
        X (array): geometry of subarray 
        eps (float): maximum distance for two points to be considered neighbors
        min_samples (int): minimum number of points to form a dense region (cluster)

    Returns:
        outlier_indices (list): indices of noise samples
        num_outliers (int): number of diagnosed outliers
        labels (array): cluster labels for each point in the dataset
    ---------------------------------------------------------------------------------------------------------'''
    # Ensure inputs are numpy arrays
    X = np.array(X)
    #-----------------------------------------------------------------------------------------------------------------#
    # Apply DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_
    #-----------------------------------------------------------------------------------------------------------------#
    # Identify outliers (label == -1)
    outlier_indices = np.where(labels == -1)[0]
    num_outliers = len(outlier_indices)
    
    return outlier_indices, num_outliers, labels # return outlier indices, num outliers, and labels (-1 label means outlier)

'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def find_closest(value, array):
    '''---------------------------------------------------------------------------------------------------------
    Finds the value in array closest to the input value

    Input:
        value (int/float): target value
        array (array): array to search for index to closest value

    Returns:
        closest_value (int/float): value in array closest to the input value
    ---------------------------------------------------------------------------------------------------------'''
    pos = bisect.bisect_left(array, value)
    # Compare neighbors to find closest value
    if pos == 0:
        return array[0]
    elif pos == len(array):
        return array[-1]
    else:
        before = array[pos-1]
        after = array[pos]
    closest_value = before if abs(before-value) <= abs(after-value) else after

    return closest_value

'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def map_time_to_sample(target_time, time):
    '''---------------------------------------------------------------------------------------------------------
    Maps a target time to the closest sample index in the time-series

    Input:
        target_time (int/float): the time value to map
        time (array): the time vector of the waveform

    Returns:
        idx (int): the index of the closest sample
    ---------------------------------------------------------------------------------------------------------'''
    idx = np.argmin(abs(time-target_time))

    return idx

'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def norm_xcorr(t_data, data1, data2):
    '''---------------------------------------------------------------------------------------------------------
    Compute normalized correlation coefficients and lag times associated with two input data streams

    Input:
        t_data (int/float): time vector associated with both data
        data1 (array): waveform data
        data2 (array): other array of waveform data

    Returns:
        lags (array): array of lag times
        x_corr (array): array of normalized cross-correlation coefficients
    ---------------------------------------------------------------------------------------------------------'''
    a = data1; b = data2
    a = (a - np.mean(a)) / (np.std(a) * len(a))
    b = (b - np.mean(b)) / (np.std(b))
    x_corr = np.correlate(a, b, 'full')
    lags = np.hstack((np.flipud(-t_data)[0:len(t_data)-1],t_data))

    return lags, x_corr

'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def xcorr_matrix(t_data, data):
    '''---------------------------------------------------------------------------------------------------------
    Compute cross-correlation matrix of data

    Input:
        t_data (int/float): time vector associated with data
        data (array): array of data where each row is cross-correlated with every other row and itself

    Returns:
        xcorr_coef (array): matrix of maximum normalized correlation coefficients
        xcorr_lag_times (array): lag times associated with each value in xcorr_coef
        ref_signal (int): index of reference signal to use if further analysis is warranted
    ---------------------------------------------------------------------------------------------------------'''
    xcorr_coef = []; xcorr_lag_times = []
    for i in range(0,len(data)):
        xcorr_coef_i = []; xcorr_lag_times_i = []
        for m in range(len(data)):
            lags, xcorr = norm_xcorr(t_data, data[i,:], data[m,:])
            max_idx = np.where((xcorr == xcorr.max()))[0]
            lag_times = lags[max_idx]
            xcorr_lag_times_i.append(lag_times)
            xcorr_coef_i.append(max(xcorr))
        xcorr_lag_times.append(xcorr_lag_times_i)
        xcorr_coef.append(xcorr_coef_i)
    xcorr_lag_times = np.array(xcorr_lag_times)
    xcorr_lag_times = xcorr_lag_times[:, :, 0]
    xcorr_coef = np.array(xcorr_coef)
    xcorr_coef_mean = xcorr_coef.mean(1)
    #-----------------------------------------------------------------------------------------------------------------#
    ref_signal = np.where(xcorr_coef_mean == np.max(xcorr_coef_mean)) # Choosing reference signal based on highest average correlation

    return xcorr_coef, xcorr_lag_times, ref_signal

'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def retrieve_subarray_data(st, subarrays_stnms, verbose=1):
    '''---------------------------------------------------------------------------------------------------------
    Get data for each subarray in subarrays_stnms

    Inputs:
        st: ObsyPy stream object
        subarrays_stnms (list): output from adaptive_array containing station names for each subarray
        verbose (boolean or int): if True or 1, will let you know if subarray data has been retrieved...if False or 0, it won't notify user

    Outputs:
        st_subarrays (list): list of ObsPy stream objects containing data for each subarray

    Note: Use this function if you are processing batches of data and don't want to re-run adaptive_array repeatedly
    ---------------------------------------------------------------------------------------------------------'''
    # Append sensor data for each subarray
    st_subarrays = []
    try:
        for subarray in np.array(subarrays_stnms, dtype=object):
            st_subarray = Stream()
            for stn in subarray:
                st_subarray.append(st.select(station=stn)[0])
            st_subarrays.append(st_subarray)
        if (verbose==1) or (verbose==True):
            print('Data for each subarray has been retrieved')
        elif (verbose==0) or (verbose==False):
            pass
    except Exception as inst:
        print(inst)
        print('Station in adaptive array subarrays may not be in original ObsPy stream object. Please check stream and re-run adaptive array with available stations.')

    return st_subarrays

'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def compute_masked_gaps(tr):
    '''---------------------------------------------------------------------------------------------------------
    Computes gaps in a masked trace by looking for time discontinuities
    
    Inputs:
        tr: masked ObsPy trace object 
    
    Outputs:
        total_masked_gap_time (float): total calculated data gap length (in s) for masked trace
        masked_gap_indices (list): masked data gap indices
    ---------------------------------------------------------------------------------------------------------'''
    if not isinstance(tr.data, np.ma.MaskedArray):
        return 0  # No masked data, so no masked gaps
    sample_rate = tr.stats.sampling_rate
    delta = 1 / sample_rate  # Time interval per sample
    #-----------------------------------------------------------------------------------------------------------------#
    # Indices where data is masked
    masked_gap_indices = np.where(tr.data.mask)[0]
    if len(masked_gap_indices) == 0:
        return 0  # No masked gaps
    #-----------------------------------------------------------------------------------------------------------------#
    # Identify time gaps based on non-contiguous masked regions
    total_masked_gap_time = 0
    gap_start_idx = masked_gap_indices[0]
    for i in range(1, len(masked_gap_indices)):
        if masked_gap_indices[i] != masked_gap_indices[i - 1] + 1:
            # Compute time duration of the masked gap
            total_masked_gap_time += (masked_gap_indices[i - 1] - gap_start_idx + 1) * delta
            gap_start_idx = masked_gap_indices[i]  # New gap start
    # Add the last masked gap
    total_masked_gap_time += (masked_gap_indices[-1] - gap_start_idx + 1) * delta

    return total_masked_gap_time, masked_gap_indices
'----------------------------------------------'
'---------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------'
'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------'
'---------------------------------------------------------------------------'
'----------------------------------------------'
############### ############### ###############
###############  The Segmentor  ################
############### ############### ###############

def make_custom_fbands(f_min=0.01, f_max=50, win_min=3, win_max=200, overlap=0.1, type='third_octave'):
   '''---------------------------------------------------------------------------------------------------------
    Makes a set of custom frequency bands and time windows for processing

    Input:
        f_min (float/int): minimum frequency (Hz)
        f_max (float/int): maximum frequency (Hz)
        win_min (float/int): minimum time window (to be used for maximum frequency - in seconds)
        win_max (float/int): maximum time window (to be used for minimum frequency - in seconds)
        type (str): type of filter band to use - options are "octave", "third_octave", or "decadal"

    Output:
        f_bands (Pandas DataFrame): frequency bands and time windows 
   ---------------------------------------------------------------------------------------------------------'''
   m, b = np.polyfit([1/f_min, 1/f_max], [win_max, win_min], 1)
   column_names = ['band', 'fmin', 'fcenter', 'fmax', 'win', 'step']
   f_bands = pd.DataFrame(columns = column_names)
   #-----------------------------------------------------------------------------------------------------------------------#
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
           f_bands.loc[-1] = [i, fmin, fcenter, fmax, win, step]  # adding a row
           f_bands.index = f_bands.index + 1  # shifting index
   #-----------------------------------------------------------------------------------------------------------------------#
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
           f_bands.loc[-1] = [i, fmin, fcenter, fmax, win, step]  # adding a row
           f_bands.index = f_bands.index + 1  # shifting index
   #-----------------------------------------------------------------------------------------------------------------------#
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
           f_bands.loc[-1] = [i, fmin, fcenter, fmax, win, step]  # adding a row
           f_bands.index = f_bands.index + 1  # shifting index
   #-----------------------------------------------------------------------------------------------------------------------#
   f_bands.index = f_bands.index[::-1]
   return f_bands

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def pmcc_fbands():
    '''---------------------------------------------------------------------------------------------------------
    PMCC array processing time-window and frequency-band configurations, as reported in Matoza et al. (2013)

    Note the variables are:
    band = Band number
    fmin = Minimum frequency of band (Hz)
    fmax = Maximum frequency of band (Hz)
    win = Window length (s)
    step = Time step (s) (10% of window length for PMCC)
    ---------------------------------------------------------------------------------------------------------'''

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
    f_bands = pd.read_csv(data, delim_whitespace=True)

    return f_bands

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def extend_pmcc_fbands(f_bands, fmax):
    '''---------------------------------------------------------------------------------------------------------
    Extends the PMCC frequency-band configuration up to fmax using the same logarithmic frequency band spacing, and window length
    spacing rules, where window length is linearly proportional to period
    
    Input:
        f_bands (Pandas DataFrame): original PMCC frequency bands
        fmax (float/int): maximum frequency to extend PMCC bands to (Hz)

    Output:
        f_bands (Pandas Dataframe): extended PMCC frequency bands
    ---------------------------------------------------------------------------------------------------------'''

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
        f_bands = pd.concat([f_bands, pd.DataFrame([{'band': band_ix, 'fmin': f_min, 'fcenter': f_cen, 'fmax': f_max, 'win': win, 'step': step}])], ignore_index=True)
        f_max_moving = f_max
    
    return f_bands
'----------------------------------------------'
'---------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------'
'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------'
'---------------------------------------------------------------------------'
'----------------------------------------------'
############### ############### ###############
############## The Adaptive Array #############
############### ############### ###############

def adaptive_array(st, f_bands, array_type, n_clusters=None, plot=False, figsize=(12,12), verbose=0, plot_units='km'):
    '''---------------------------------------------------------------------------------------------------------
    Algorithm to determine optimal subarray geometry for each frequency band in f_bands.
    
    Input:
        st: ObsPy stream object
        f_bands (Pandas DataFrame): frequency bands and time windows constructed using the Segmentor
        array_type (str): either "seismic" or "infrasound" - specifies velocities and scalers to use
        vel (float/int): designated wave speed (in km) used to determine whether clustered subarray apertures can resolve for specified wavelength
        n_clusters (int): number of clusters to use to group subarrays (clusters by array aperture - defaults to len(st)-2)
        plot (boolean): whether to plot each unique optimal subarray
        figsize (tuple): if plot, defines size of figure containing all subarray geometries
        verbose (int/boolean): if 1 or True, prints optimally determined subarray for each sequential frequency band
        plot_units (str): if plot, specifies the scale of each subarray plot - options are "m" or "km"

    Output:
        st_subarrays (list): a list of ObsPy stream objects containing data for each subarray
        optimal_subarray_stnms (list): list of station names for each subarray

    Note: Use optimal_subarray_stnms as input to retrieve_subarray_data() for batch processing or processing data in real-time,
    this way you don't need to re-run the adaptive_array for each new batch of array data (check notebook for example implementation)
    ----------------------------------------------------------------------------------------------------------'''
    # Load RC ConvFormer
    tf.keras.config.enable_unsafe_deserialization() # to allow unsafe deserialization
    RC_ConvFormer = tf.keras.models.load_model(RC_ConvFormer_dir, custom_objects={'TransformerBlock': TransformerBlock}, compile=False)
    #-----------------------------------------------------------------------------------------------------------------#
    # Load Scalers
    with open(X_infrasound_scaler_dir, 'rb') as f:
        X_infrasound_scaler = pickle.load(f)
    with open(X_seismic_scaler_dir, 'rb') as f:
        X_seismic_scaler = pickle.load(f)
    with open(F_scaler_dir, 'rb') as f:
        F_scaler = pickle.load(f)
    with open(y_infrasound_scaler_dir, 'rb') as f:
        y_infrasound_scaler = pickle.load(f)
    with open(y_seismic_scaler_dir, 'rb') as f:
        y_seismic_scaler = pickle.load(f)
    #-----------------------------------------------------------------------------------------------------------------#
    # To mitigate for high-frequency decorrelation - cluster subarray geometries by array aperture
    if len(st) <= 10:
        elements_in_subarray = np.arange(3,len(st)+1)
    else:
        elements_in_subarray = np.arange(3,11)
    subarrays = []; subarrays_X = []; max_dists = []
    #-----------------------------------------------------------------------------------------------------------------#
    # Compile all possible subarrays
    X, stnm = get_array_coords(st, st[0].stats.station, units='m') # RC ConvFormer was trained in meters for both seismic and infrasound
    for num_elements in elements_in_subarray:
        subarrays_i = []; subarrays_X_i = []; max_dists_i = []
        # Generate all possible groupings of num_elements in array
        combinations = list(itertools.combinations(stnm, num_elements))
        combinations = [*set(combinations)] # remove duplicates
        for subarray in combinations:
            subarray, subarray_X, max_dist = get_subarray_geometry(subarray, stnm, X)
            subarrays_i.append(subarray); subarrays_X_i.append(subarray_X); max_dists_i.append(max_dist)
        #-----------------------------------------------------------------------------------------------------------------#
        # Storing
        subarrays.append(subarrays_i); subarrays_X.append(subarrays_X_i); max_dists.append(np.array(max_dists_i))
    max_dists_concat = np.concatenate((max_dists))
    #-----------------------------------------------------------------------------------------------------------------#
    # Subarray clustering via aperture
    if n_clusters == None:
        km = KMeans(n_clusters=int(np.ceil(len(st)/2)), init='k-means++', # manually tuned
                    n_init=10, max_iter=1000,
                    tol=1e-04, random_state=42)
    else:
        km = KMeans(n_clusters=n_clusters, init='k-means++', # user specified
                    n_init=10, max_iter=1000,
                    tol=1e-04, random_state=42)
    #-----------------------------------------------------------------------------------------------------------------#
    # Using array apertures to cluster
    ss = StandardScaler()
    X_km = max_dists_concat.reshape(-1,1)
    X_scaled = ss.fit_transform(X_km)
    y_km = km.fit_predict(X_scaled)
    all_subarrays = [item for sublist in subarrays for item in sublist]
    all_subarrays_X = [item for sublist in subarrays_X for item in sublist]
    all_max_dists = [item for sublist in max_dists for item in sublist]
    #-----------------------------------------------------------------------------------------------------------------#
    # Sorting cluster id's from largest to smallest
    ids = np.unique(y_km)
    ids_max_dists = np.zeros((len(ids)))
    for id_tmp in ids:
        y_km_idx = np.where((id_tmp == y_km))[0]
        ids_max_dists[id_tmp] = max_dists_concat[y_km_idx].max()
    ids_sorted = ids[np.argsort(ids_max_dists)][::-1] # from largest to smallest
    #-----------------------------------------------------------------------------------------------------------------#
    # Combining subarrays into respective clusters
    clustered_subarrays = []; clustered_subarrays_X = []; clustered_max_dists = []
    for id_sorted in ids_sorted:
        cluster_idxs = np.where((y_km == id_sorted))[0]
        clustered_subarrays.append(np.array(all_subarrays, dtype=object)[cluster_idxs])
        clustered_subarrays_X.append(np.array(all_subarrays_X, dtype=object)[cluster_idxs])
        clustered_max_dists.append(np.array(all_max_dists)[cluster_idxs])
    clustered_max_dists = np.array(clustered_max_dists, dtype='object')
    #-----------------------------------------------------------------------------------------------------------------#
    # Calculate and sort by RC
    RC = []; subarray_stnms = []
    dbscan_dist_thresh = np.std(element_pair_distances(st, return_params=True)[0]) # here we set the outlier distance threshold (in km) used to identify outlier stations in subarrays
    dbscan_dist_thresh *= 1000 # convert to meters
    if array_type=='seismic':
        scaler_geometry = X_seismic_scaler
        scaler_target = y_seismic_scaler
        vel = 3 * 1000 # 3 km/s for seismic
        OHE_values = [1,0]
    elif array_type=='infrasound':
        scaler_geometry = X_infrasound_scaler
        scaler_target = y_infrasound_scaler
        vel = 0.34 * 1000 # 0.34 km/s for infrasound
        OHE_values = [0,1]
    # Adaptive wavelength resolution
    wavelength_scale_min = 0.5; wavelength_scale_max = 2
    wavelength_scales = np.geomspace(wavelength_scale_min, wavelength_scale_max, len(f_bands)) # log space
    cluster_idx = 0; allow_outlier_detection = 1
    for band_idx in range(len(f_bands)):
        redo_1 = True
        while redo_1:
            # Calculate largest wavelength to resolve for
            wavelength_tmp = vel / f_bands.iloc[band_idx]['fmin']
            wavelength_tmp /= wavelength_scales[band_idx]
            if clustered_max_dists[cluster_idx].max() > (wavelength_tmp): # go to next cluster if current array aperture can't resolve for wavelength
                if cluster_idx+1 == len(clustered_subarrays):
                    pass
                else:
                    cluster_idx += 1
            redo_2 = True
            while redo_2:
                RC_tmp = []; subarray_stnms_tmp = []
                groupings = clustered_subarrays[cluster_idx]; groupings_X = clustered_subarrays_X[cluster_idx]
                for subarray, subarray_X in zip(groupings, groupings_X):
                    if allow_outlier_detection == 1:
                        # Need to check how many station outliers there are (if 1 then move on - means we have only linearly configured arrays in cluster)
                        _, num_outliers, _ = dbscan_outliers(X=subarray_X, eps=dbscan_dist_thresh, min_samples=2)
                        if num_outliers == 1:
                            continue # move on from linear subarray
                    #-----------------------------------------------------------------------------------------------------------------#
                    # Need to convert 0 to small number for scaling
                    ref_idx = np.where((subarray_X == 0.))
                    try:
                        ref_idx = np.where((subarray_X == 0.)); subarray_X[ref_idx] = np.array([1e-10, 1e-10]) # need to mask out zeros for scaling
                    except:
                        for zero_idx in range(len(ref_idx[0])):
                            subarray_X[ref_idx[0][zero_idx], ref_idx[1][zero_idx]] = 1e-10
                    #-----------------------------------------------------------------------------------------------------------------#
                    # Pad subbary geometry
                    padded_geometry = pad_geometry(subarray_X)
                    # Retrieve freqrange
                    freqmin = f_bands.iloc[band_idx]['fmin']; freqmax = f_bands.iloc[band_idx]['fmax']
                    freqrange = np.zeros((1,2)); freqrange[:,0] = freqmin.copy(); freqrange[:,1] = freqmax.copy()
                    #-----------------------------------------------------------------------------------------------------------------#
                    # Scale inputs - geometry
                    nonzero_mask_geometry = padded_geometry != 0
                    nonzero_values_geometry = padded_geometry[nonzero_mask_geometry]
                    nonzero_values_scaled_geometry = scaler_geometry.transform(nonzero_values_geometry.reshape(-1,1)).flatten()
                    scaled_geometry = np.zeros_like(padded_geometry, dtype=float)
                    scaled_geometry[nonzero_mask_geometry] = nonzero_values_scaled_geometry.copy()
                    # Scale inputs - freqrange
                    scaled_freqrange_tmp = F_scaler.transform(freqrange.reshape(-1,1))
                    scaled_freqrange = np.zeros_like(freqrange, dtype=float)
                    scaled_freqrange[:,0] = scaled_freqrange_tmp[0]; scaled_freqrange[:,1] = scaled_freqrange_tmp[1]
                    # Merge OHE with scaled geometries 
                    scaled_geometry = scaled_geometry.reshape(1, scaled_geometry.shape[0], scaled_geometry.shape[1])
                    OHE = np.tile(OHE_values, (scaled_geometry.shape[0], scaled_geometry.shape[1], 1)) 
                    scaled_geometry = np.concatenate((scaled_geometry, OHE), axis=2)
                    #-----------------------------------------------------------------------------------------------------------------#
                    # RC ConvFormer
                    RC_metric = RC_ConvFormer.predict([scaled_geometry, scaled_freqrange], verbose=None)
                    #-----------------------------------------------------------------------------------------------------------------#
                    # Inverse transform target with appropriate scaler
                    RC_tmp.append(scaler_target.inverse_transform(RC_metric[0][0].reshape(1,-1))[0][0]) # calculate RC
                    subarray_stnms_tmp.append(subarray)
                RC_tmp = np.array(RC_tmp); subarray_stnms_tmp = np.array(subarray_stnms_tmp, dtype=object)
                if (len(RC_tmp) != 0) and (len(subarray_stnms_tmp) != 0):
                    redo_2 = False # exit freq band for loop
                elif (len(RC_tmp) == 0) and (len(subarray_stnms_tmp) == 0):
                    if cluster_idx == y_km.max(): # for last cluster no need to redo
                        allow_outlier_detection = 0
                    else:
                        cluster_idx += 1
            RC_tmp_sorted = np.argsort(RC_tmp)[::-1] # indices from max to min RC
            RC.append(RC_tmp[RC_tmp_sorted]); subarray_stnms.append(subarray_stnms_tmp[RC_tmp_sorted])
            if (len(RC) not in [0]) and (len(subarray_stnms) not in [0]): # if number of subarrays is not 0 or 1 then move on
                redo_1 = False
            elif (len(RC) in [0]) and (len(subarray_stnms) in [0]): # if number of subarrays is 0 or 1 then redo without outlier detection
                allow_outlier_detection = 0; cluster_idx = 0
    #-----------------------------------------------------------------------------------------------------------------#
    # Return subarray with minimum RC from each cluster for each frequency band
    st_subarrays = []; optimal_subarray_stnms = []
    for band_idx in range(len(f_bands)):
        redo = True; last_idx = 1
        while redo:
            # Retrieve relative geometry
            st_subarray = Stream()
            for subarray_idx in range(len(subarray_stnms[band_idx][-last_idx])):
                st_subarray.append(st.select(station=subarray_stnms[band_idx][-last_idx][subarray_idx])[0])
            _, subarray_stnm = get_array_coords(st_subarray, ref_station=st_subarray[0].stats.station) # no need to worry about distance scale here (only extracting subarray station names)
            if len(st_subarrays) == 0:
                st_subarrays.append(st_subarray); optimal_subarray_stnms.append(subarray_stnm)
                redo = False
            else:
                if st_subarray != st_subarrays[-1]:
                    if st_subarray not in st_subarrays[:-1]:
                        st_subarrays.append(st_subarray); optimal_subarray_stnms.append(subarray_stnm)
                        redo = False
                    elif st_subarray in st_subarrays[:-1]:
                        last_idx += 1
                        pass
                elif st_subarray == st_subarrays[-1]:
                    st_subarrays.append(st_subarray); optimal_subarray_stnms.append(subarray_stnm)
                    redo = False
    if (verbose == 1) or (verbose == True):
        for band_idx in range(len(f_bands)):
            print('Subarray for band '+str(band_idx+1)+': '+str(optimal_subarray_stnms[band_idx]))
    #-----------------------------------------------------------------------------------------------------------------#
    # Plot
    if plot:
        try: # assuming each subarray is of equal length - need to add axis=0 or else unique will choose subarray elements instead of subarrays
            optimal_subarray_stnms_as_array = np.array(optimal_subarray_stnms)
            optimal_subarray_indices, optimal_subarrays = np.unique(optimal_subarray_stnms_as_array, axis=0, return_index=True)[::-1]
        except: # if subarrays are not of same length must be set as dtype=object - it will choose subarrays as unique
            optimal_subarray_stnms_as_array = np.array(optimal_subarray_stnms, dtype=object)
            optimal_subarray_indices, optimal_subarrays = np.unique(optimal_subarray_stnms_as_array, return_index=True)[::-1]
        optimal_subarrays = optimal_subarrays[np.argsort(optimal_subarray_indices)]
        fig = plt.figure(figsize=figsize)
        for optimal_subarray_idx in range(len(optimal_subarrays)):
            band_range_tmp = []
            for idx, lst in enumerate(optimal_subarray_stnms_as_array):
                if tuple(lst) == tuple(optimal_subarrays[optimal_subarray_idx]):
                    band_range_tmp.append(idx)
            band_range_tmp = np.array(band_range_tmp)
            if len(optimal_subarrays) % 2 == 1: # if odd number of optimal subarrays
                if optimal_subarray_idx == 0:
                    ax = fig.add_subplot((len(optimal_subarrays)//2)+1,2,optimal_subarray_idx+1)
                else:
                    ax_tmp = fig.add_subplot((len(optimal_subarrays)//2)+1,2,optimal_subarray_idx+1, sharex=ax, sharey=ax)
            else:
                if optimal_subarray_idx == 0:
                    ax = fig.add_subplot(len(optimal_subarrays)//2,2,optimal_subarray_idx+1)
                else:
                    ax_tmp = fig.add_subplot(len(optimal_subarrays)//2,2,optimal_subarray_idx+1, sharex=ax, sharey=ax)
            # Get subarray coords
            st_tmp = Stream()
            for stn_label in optimal_subarrays[optimal_subarray_idx]:
                st_tmp.append(st.select(station=stn_label)[0])
            # Need to make sure we can use same reference station across both array and subarrays
            remove_first_station = False
            if st[0].stats.station != st_tmp[0].stats.station:
                st_tmp.insert(0, st.select(station=st[0].stats.station)[0])
                remove_first_station = True
            # Get relative array and relative subarray coordinates
            if plot_units == 'km':
                X_tmp, _ = get_array_coords(st_tmp, st_tmp[0].stats.station, units='km')
                X, stnm = get_array_coords(st, st[0].stats.station, units='km')
            elif plot_units == 'm':
                X_tmp, _ = get_array_coords(st_tmp, st_tmp[0].stats.station, units='m')
                X, stnm = get_array_coords(st, st[0].stats.station, units='m')
            else:
                raise Exception('Options for plot_units are "m" or "km"')
            # Plot array
            plt.plot(X[:,0], X[:,1], '.r')
            for i in range(0, len(stnm)):
                plt.text(X[i,0], X[i,1], stnm[i])
            # Plot subarray - but first remove first station in st_tmp if we had to add it earlier
            if remove_first_station == True:
                X_tmp = X_tmp[1:,:]
            for i in range(0, len(X_tmp)):
                for j in range(i+1, len(X_tmp)):
                    x_points = [X_tmp[i,0], X_tmp[j,0]]
                    y_points = [X_tmp[i,1], X_tmp[j,1]]
                    plt.plot(x_points, y_points, '.b-', lw=0.25)
            if plot_units == 'km':
                if (optimal_subarray_idx+1) % 2 == 0: # if even
                    if optimal_subarray_idx == 0:
                        ax.tick_params(labelleft=False)
                    else:
                        ax_tmp.tick_params(labelleft=False)
                else:
                    plt.ylabel('km')
                if (optimal_subarray_idx+1 == len(optimal_subarrays)) or (optimal_subarray_idx+1 == len(optimal_subarrays)-1):
                    plt.xlabel('km')
                else:
                    if optimal_subarray_idx == 0:
                        ax.tick_params(labelbottom=False)
                    else:
                        ax_tmp.tick_params(labelbottom=False)
            elif plot_units == 'm':
                if (optimal_subarray_idx+1) % 2 == 0: # if even
                    if optimal_subarray_idx == 0:
                        ax.tick_params(labelleft=False)
                    else:
                        ax_tmp.tick_params(labelleft=False)
                else:
                    plt.ylabel('m')
                if (optimal_subarray_idx+1 == len(optimal_subarrays)) or (optimal_subarray_idx+1 == len(optimal_subarrays)-1):
                    plt.xlabel('m')
                else:
                    if optimal_subarray_idx == 0:
                        ax.tick_params(labelbottom=False)
                    else:
                        ax_tmp.tick_params(labelbottom=False)
            plt.grid(lw=0.25)
            if len(band_range_tmp) > 1:
                plt.title('Bands ' + str(band_range_tmp.min()+1) + ' to ' + str(band_range_tmp.max()+1))
            else:
                plt.title('Band ' + str(band_range_tmp[0]+1))
        plt.suptitle('Optimal Subarrays')
    
    # Return optimal subarray for each band as ObsPy stream object (with data) and list of station names (without data)
    return st_subarrays, optimal_subarray_stnms
'----------------------------------------------'
'---------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------'
'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------'
'---------------------------------------------------------------------------'
'----------------------------------------------'
############### ############### ###############
############# The Array Processor #############
############### ############### ###############

def array_lsq(st, X):
    '''---------------------------------------------------------------------------------------------------------
    Performs pairwise cross-correlation on each trace in st, and least-squares inversion for the slowness vector corresponding to the best-fitting plane wave
    
    Input:
        st: ObsPy stream object (all traces must begin at the same time - within the sampling interval)
        X (array): array coordinates relative to a reference station

    Output:
        baz (float): back azimuth (in degrees from North)
        v (float): trace velocity
    ---------------------------------------------------------------------------------------------------------'''
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

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def sliding_time_array_lsq(st, X, tstart, tend, twin, overlap):
    '''---------------------------------------------------------------------------------------------------------
    Performs sliding time-window array processing using the least-squares array processing method in array_lsq
    
    Input:
        st: ObsPy stream object containing array data
        X (array): array coordinates relative to a reference station
        tstart (int): start time for processing (in seconds after st start time)
        tend (int): end time for processing (in seconds after st start time)
        twin (float/int): time window for array processing (s)
        overlap (float/int): overlap for array processing (s)
    
    Output:
        T (array): times of array processing estimates (center of time windows) (s)
        V (array): trace velocities
        B (array): back azimuths
    ---------------------------------------------------------------------------------------------------------'''
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

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def sliding_time_array_fk(st, element, tstart=None, tend=None, win_len=20, win_frac=0.5, frqlow=0.5, frqhigh=4, sll_x=-3.6, slm_x=3.6, sll_y=-3.6, slm_y=3.6, sl_s=0.18, sl_corr=[0.,0.],
                          normalize_waveforms=True, use_geographic_coords=True):
    '''---------------------------------------------------------------------------------------------------------
    Processes st with sliding window FK analysis. Default parameters are suitable for most regional infrasound arrays.

    Input:
        st: ObsPy stream object containing array data
        element (str): station name to be used as reference
        tstart (int): start time for processing (in seconds after st start time)
        tend (int): end time for processing (in seconds after st start time)
        win_len (float/int): time window length for array processing (s)
        win_frac (float): percent overlap of successive windows (from 0 to 1)
        frqlow (float/int): minimum frequency (Hz)
        frqhigh (float/int): maximum frequency (Hz)
        sll_x, slm_x (float/int): extent of x-axis in slowness grid
        sll_y, slm_y (float/int): extent of y-axis in slowness grid
        sl_s (float/int): slowness grid resolution
        sl_corr (array): specified correction for slowness
        normalize_waveforms (boolean): whether to normalize waveforms by max absolute value
        use_geographic_coordinates (boolean): whether to use lat/lon of array elements for processing

    Output:
        T (array): times of array processing estimates (center of time windows) (s)
        V (array): trace velocities
        B (array): back azimuths
        S (array): semblance (coherence metric)
    ---------------------------------------------------------------------------------------------------------'''
    # Trace of reference element
    tr = st.select(station=element)[0]
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
    #-----------------------------------------------------------------------------------------------------------------#
    # Process data
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
    #-----------------------------------------------------------------------------------------------------------------#
    # Convert times to seconds after start time of reference element and adjusting to center of time window:
    T = ((slid_fk[:,0] - date2num(tr.stats.starttime.datetime))*86400) + win_len/2
    # Convert backazimuths to degrees from North:
    B = slid_fk[:,3] % 360.
    # Convert slowness to phase velocity:
    V = 1/slid_fk[:,4]
    # Semblance:
    S = slid_fk[:,1]

    return T, B, V, S

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def sliding_time_array_fk_multifreq(st, f_bands, client=None, t_start=None, t_end=None, signal_type='infrasound', sl_corr=[0.,0.],
                                    use_geographic_coords=True, adaptive_array=False, memory_usage_threshold=90):
    '''---------------------------------------------------------------------------------------------------------
    Processes st with sliding window FK analysis in multiple frequency bands

    Input:
        st: ObsPy stream object containing array data
        f_bands (Pandas DataFrame): frequency bands and time windows constructed by using the Segmentor
        client (object): central object that facilitates connection between Python environment and the distributed Dask cluster
        tstart (int): start time for processing (in seconds after st start time)
        tend (int): end time for processing (in seconds after st start time)
        signal_type (str): type of signal intended to be detected - options are 'infrasound' or 'seismic' (defines slowness grid for regional distances)
        sl_corr (array): specified correction for slowness
        use_geographic_coordinates (boolean): whether to use lat/lon of array elements for processing
        adaptive_array (boolean): whether the input st is a list of optimally determined subarrays for each frequency band - output of adaptive_array
        memory_usage_threshold (int): specifies client to restart once memory allocation limit has been reached (defaults to 90%)

    Output:
        T (array): times of array processing estimates (center of time windows) (s)
        V (array): trace velocities
        B (array): back azimuths
        S (array): semblance (coherence metric)

    Note: Client restarts automatically once 80% memory has been allocated
    IMPORTANT: Client needs to be specified outside of function to avoid unnecessary overhead
    ---------------------------------------------------------------------------------------------------------'''
    # Slowness grid
    if signal_type == 'infrasound':
        sll_x=-3.6; slm_x=3.6; sll_y=-3.6; slm_y=3.6; sl_s=0.18
    elif signal_type == 'seismic':
        sll_x=-0.5; slm_x=0.5; sll_y=-0.5; slm_y=0.5; sl_s=0.01
    #-----------------------------------------------------------------------------------------------------------------#
    # Set warning message for parallel processing
    if client is not None:
        n_workers = len(client.scheduler_info()["workers"])
    else:
        n_workers = 1
    if (client == None) and (n_workers > 1):
        raise Exception('Client must be specified if n_workers > 1')
    elif (client is not None) and (n_workers == 1):
        raise Exception('n_workers must be > 1 for parallel computing')
    # Defining t_start, t_end:
    if (t_start == None) and (t_end == None):
        if adaptive_array == True:
            tr = st[0].select(station=st[0][0].stats.station)[0]
        else:
            tr = st.select(station=st[0].stats.station)[0]
        t_start = 1
        t_end = (tr.stats.npts * tr.stats.delta)-1
    if adaptive_array == False:
        element = st[0].stats.station
        st = [st]*len(f_bands)
    #-----------------------------------------------------------------------------------------------------------------#
    # Processing each frequency band with sliding window FK processing:
    T_all = []; B_all = []; V_all = []; S_all = []; dask_all = []
    for f_band, st_tmp in zip(f_bands['band'].values, st):
        # Initialize array processing params
        win_len = f_bands[f_bands['band'] == f_band]['win'].values[0]
        frqlow = f_bands[f_bands['band'] == f_band]['fmin'].values[0]
        frqhigh = f_bands[f_bands['band'] == f_band]['fmax'].values[0]
        win_frac = f_bands[f_bands['band'] == f_band]['step'].values[0]/f_bands[f_bands['band'] == f_band]['win'].values[0]
        if adaptive_array == True:
            element = st_tmp[0].stats.station # reference station may vary with adaptive array
        # Process array data
        if n_workers == 1: # don't run parallel computing
            T, B, V, S = sliding_time_array_fk(st_tmp, element, tstart=t_start, tend=t_end,
                                                win_len=win_len, win_frac=win_frac, frqlow=frqlow, frqhigh=frqhigh,
                                                sll_x=sll_x, slm_x=slm_x, sll_y=sll_y, slm_y=slm_y, sl_s=sl_s, sl_corr=sl_corr, 
                                                use_geographic_coords=use_geographic_coords)
            T_all.append(T); B_all.append(B); V_all.append(V); S_all.append(S)
        elif n_workers > 1: # run parallel computing 
            dask_out = dask.delayed(sliding_time_array_fk)(st_tmp, element, tstart=t_start, tend=t_end, 
                                                           win_len=win_len, win_frac=win_frac, frqlow=frqlow, frqhigh=frqhigh, 
                                                           sll_x=sll_x, slm_x=slm_x, sll_y=sll_y, slm_y=slm_y, sl_s=sl_s, sl_corr=sl_corr,
                                                           use_geographic_coords=use_geographic_coords)
            dask_all.append(dask_out)
    if n_workers > 1:
        # Organizing output from distributed process and checking memory usage
        out = dask.compute(*dask_all)
        for out_i in out:
            T_all.append(out_i[0]); B_all.append(out_i[1]); V_all.append(out_i[2]); S_all.append(out_i[3])
        memory_usage = psutil.virtual_memory().percent
        if memory_usage > memory_usage_threshold:
            print(f"Memory usage at {memory_usage}%, restarting client...")
            gc.collect()
            client.restart()
    #-----------------------------------------------------------------------------------------------------------------#
    # Extracting the time vector corresponding to the maximum number of values:
    N = []
    for T in T_all:
        N.append(len(T))
    T = T_all[np.argmax(N)] # Times for f-band with highest number of DOA estimates
    #-----------------------------------------------------------------------------------------------------------------#
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

    return T, B, V, S

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def sliding_time_array_fk_multifreq_tblock(st, f_bands, client, t_start=None, t_end=None, signal_type='infrasound', sl_corr=[0.,0.],
                                           use_geographic_coords=True, adaptive_array=False, memory_usage_threshold=90):
    '''---------------------------------------------------------------------------------------------------------
    Runs array processing by parsing different time blocks to different threads and applying parallel computing via Dask

    Input:
        st: ObsPy stream object containing array data
        f_bands (Pandas DataFrame): frequency bands and time windows constructed by using the Segmentor
        client (object): central object that facilitates connection between Python environment and the distributed Dask cluster
        tstart (int): start time for processing (in seconds after st start time)
        tend (int): end time for processing (in seconds after st start time)
        signal_type (str): type of signal intended to be detected - options are 'infrasound' or 'seismic' (defines slowness grid for regional distances)
        sl_corr (array): specified correction for slowness
        use_geographic_coordinates (boolean): whether to use lat/lon of array elements for processing
        adaptive_array (boolean): whether the input st is a list of optimally determined subarrays for each frequency band - output of adaptive_array
        memory_usage_threshold (int): specifies client to restart once memory allocation limit has been reached (defaults to 90%)

    Output:
        T (array): times of array processing estimates (center of time windows) (s)
        V (array): trace velocities
        B (array): back azimuths
        S (array): semblance (coherence metric)

    Note: This should only be used on large amounts of data at least some hours long
    ---------------------------------------------------------------------------------------------------------'''
    # Slowness grid
    if signal_type == 'infrasound':
        sll_x=-3.6; slm_x=3.6; sll_y=-3.6; slm_y=3.6; sl_s=0.18
    elif signal_type == 'seismic':
        sll_x=-0.5; slm_x=0.5; sll_y=-0.5; slm_y=0.5; sl_s=0.01
    #-----------------------------------------------------------------------------------------------------------------#
    # Specify n_workers > 1 to use parallel computing
    n_workers = len(client.scheduler_info()["workers"])
    if n_workers == 1:
        raise Exception('Please specify n_workers > 1 to apply parallel computing')
    if t_end is None:
        if adaptive_array == True:
            t_end = st[0][0].stats.npts * st[0][0].stats.delta
        else:
            t_end = st[0].stats.npts * st[0].stats.delta
    if t_start is None:
        t_start = 0
    t_dur = t_end - t_start
    t_block = int(t_dur/n_workers)
    #-----------------------------------------------------------------------------------------------------------------#
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
    #-----------------------------------------------------------------------------------------------------------------#
    # Running sliding_time_array_fk_multifreq for each start/end block:
    dask_all = []
    for i in range(0, len(t_start_times)): # running parallel computing in tblock only!
        dask_out = dask.delayed(sliding_time_array_fk_multifreq)(st, f_bands, t_start=t_start_times[i], t_end = t_end_times[i], 
                                                                 signal_type=signal_type, sl_corr=sl_corr, use_geographic_coords=use_geographic_coords, 
                                                                 adaptive_array=adaptive_array)
        dask_all.append(dask_out)
    out = dask.compute(*dask_all)
    # Restarting client if memory usage is above 80%
    memory_usage = psutil.virtual_memory().percent
    if memory_usage > memory_usage_threshold:
        print(f"Memory usage at {memory_usage}%, restarting client...")
        gc.collect()
        client.restart()
    #-----------------------------------------------------------------------------------------------------------------#
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

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def beamform(t_shifts, st, ref_station, normalize_data=False, normalize_beam=False, divide_by_num_stns=False, plot=False, legend_loc='upper right', legend_size=10, figsize=(9,6)):

    '''---------------------------------------------------------------------------------------------------------
     Beamforms time-series array data based on slowness vector time shifts

     Input:
        t_shifts (array): time shifts for each array element computed using get_slowness_vector_time_shifts
        st: ObsPy stream object
        ref_station (str): station name to be used for reference station
        normalize_data (boolean): whether to normalize each trace
        normalize_beam (boolean): whether to normalize beamformed data
        divide_by_num_stns (boolean): whether to divide the beam by the number of array elements, returning the average amplitude
        plot (boolean): whether to plot stream and beamformed data
        legend_loc (str): if plot, specifiy location of legend
        legend_size (float/int): if plot, specifiy legend size
        figsize (tuple): if plot, specify figure size
    ---------------------------------------------------------------------------------------------------------'''
    tr_ref = st.select(station=ref_station)[0]
    ix = 0
    if plot == True:
        fig = plt.figure(figsize=figsize)
    for i,tr in enumerate(st):
    # Need to truncate and pad data
        t_ix = np.arange(0, abs(t_shifts[ix]), tr.stats.delta)
        if t_shifts[ix] < 0:
            data = tr.data[len(t_ix)::]
        elif t_shifts[ix] > 0:
            data = np.concatenate((np.zeros((len(t_ix))), tr.data))
        else:
            data = tr.data  
    #-----------------------------------------------------------------------------------------------------------------#
    # Computing stack
        if ix == 0:
            stack = data
        else:
            diff = len(stack)-len(data)
            if diff > 0:
                data = np.concatenate((data, np.zeros(diff)))
            elif diff < 0:
                stack = np.concatenate((stack, np.zeros(np.abs(diff))))
            stack = data + stack
        if normalize_data == True:
            data = data/(np.max(np.abs(data)))
        t_beam = np.arange(0, len(data)*tr_ref.stats.delta, tr_ref.stats.delta)
        if plot == True:
            if i > 0:
                ax_tmp = fig.add_subplot(len(st)+1,1,i+1, sharex=ax_tmp, sharey=ax_tmp)
            else:
                ax_tmp = fig.add_subplot(len(st)+1,1,i+1)
            try:
                ax_tmp.plot(t_beam, data, 'k', label=tr.stats.station)
            except:
                t_beam = np.arange(0.001, len(data)*tr_ref.stats.delta, tr_ref.stats.delta)
                ax_tmp.plot(t_beam, data, 'k', label=tr.stats.station)
            ax_tmp.tick_params(labelbottom=False)
            plt.legend(loc=legend_loc, prop={ "size": legend_size})
        ix += 1
    if divide_by_num_stns == True:
        beam = stack/len(st)
    else:
        beam = stack.copy()
    if normalize_beam == True:
        beam = beam/np.max(np.abs(beam))
    #-----------------------------------------------------------------------------------------------------------------#
    # Plotting beam
    if plot == True:
        ax_tmp = fig.add_subplot(len(st)+1, 1, len(st)+1, sharex=ax_tmp, sharey=ax_tmp)
        try:
            ax_tmp.plot(np.arange(0, len(data)*tr_ref.stats.delta, tr_ref.stats.delta), beam, 'blue', label='Beam')
        except:
            ax_tmp.plot(np.arange(0.001, len(data)*tr_ref.stats.delta, tr_ref.stats.delta), beam, 'blue', label='Beam')
        plt.xlabel('Time (s) after ' + str(tr_ref.stats.starttime).split('.')[0].replace('T', ' '))
        plt.legend(loc=legend_loc, prop={ "size": legend_size})

    # Return params
    return t_beam, beam

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def save_sliding_window_multifreq(fname, st, f_bands, T, B, V, S):
    '''---------------------------------------------------------------------------------------------------------
    Saves the inputs and results of Cardinal processing to a file

    Input:
        fname (str): filepath to save results to
        st: ObsPy stream object to be saved
        f_bands (Pandas DataFrame): frequency bands and time windows to be saved
        T (array): timestamps to be saved
        B (array): back azimuths to be saved
        V (array): trace velocities to be saved
        S (array): semblance values to be saved
    ---------------------------------------------------------------------------------------------------------'''
    pickle.dump([st,f_bands,T,B,V,S], open(fname, 'wb'))

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def load_sliding_window_multifreq(fname):
    '''---------------------------------------------------------------------------------------------------------
    Loads the inputs and results of Cardinal processing from a file

    Input:
        fname (str): filepath to saved Cardinal processing results

    Output:
        st: saved ObsPy stream objecy
        f_bands (Pandas DataFrame): saved frequency bands and time windows used for processing
        T (array): saved timestamps
        B (array): saved back azimuths
        V (array): saved trace velocities
        S (array): saved semblance values
    ---------------------------------------------------------------------------------------------------------'''
    results = pickle.load(open(fname, 'rb'))
    st = results[0]; f_bands = results[1]; T = results[2]; B = results[3]; V = results[4]; S = results[5]

    return st, f_bands, T, B, V, S
'----------------------------------------------'
'---------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------'
'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------'
'---------------------------------------------------------------------------'
'----------------------------------------------'
############### ############### ###############
##############  The Aggregator  ###############
############### ############### ###############

def _compute_percentile(S, p_threshold):
    '''---------------------------------------------------------------------------------------------------------
    Compute adaptive semblance percentile per frequency band
    ---------------------------------------------------------------------------------------------------------'''
    semblance_values = []
    for i in range(S.shape[0]):
        row = np.ravel(S[i])
        if row.size == 0:  # **Check if empty before fitting KernelDensity**
            semblance_values.append(0)  # Assign a default low semblance threshold
            continue
        kde = KernelDensity(bandwidth=0.01, kernel='gaussian')
        kde.fit(row[:, None])
        x = np.linspace(0, 1, 1000)
        log_density = kde.score_samples(x[:, None])
        density = np.exp(log_density)
        cdf = cumulative_trapezoid(density, x, initial=0)
        interp_cdf = interp1d(cdf, x, bounds_error=False, fill_value=(x[0], x[-1]))
        semblance_values.append(interp_cdf(1 - p_threshold))

    return np.array(semblance_values)

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def make_families(T, B, V, S, f_bands, ref_time, segment_duration=3600,
                  semblance_threshold=0.6, dist_threshold=2, min_pixels=50,
                  sigma_t=2, sigma_f=2, sigma_b=5, p_threshold=None, GT_baz=None, n_forward=200,
                  baz_dev=45, expected_vel=None, vel_dev=0.15, family_grouping='adaptive_kdtree_window'):
    '''---------------------------------------------------------------------------------------------------------
    Makes families by applying a semblance threshold and clustering resultant pixels based on weighted distance
    within user-defined blocked segments of time.

    Input:
        T (array): timestamps corresponding to array processing results
        B (array): estimated backazimuths for all times and frequencies
        V (array): estimated phase velocities
        S (array): estimated semblances
        f_bands (Pandas DataFrame): frequency band information
        ref_time: A Matplotlib datenumber containing the reference time for T
        semblance_threshold (float): semblance threshold for detection
        dist_threshold (float): Threshold weighted distance for clustering pixels
        min_pixels (int): Threshold minimum number of pixels for a family
        sigma_t (int): Standard deviation in time indices for clustering
        sigma_f (int): Standard deviation in frequency indices for clustering
        sigma_b (int): Standard deviation in backazimuth indices for clustering
        n_forward (int): Number of pixels after a pixel (in time) to compute weighted distances
        p_threshold (float): Percentile threshold for adaptive semblance threshold
        GT_baz (float/int): ground-truth back azimuth to be used with baz_dev to filter out pixels outside range
        baz_dev (float/int): deviation from ground-truth to filter pixels
        exepcted_vel (float/int): like GT_baz but now using expected trace velocity
        vel_dev (float/int): deviaton from expected vel - anything outside range will filter out
        family_grouping (str): 'original', 'kdtree', or 'kdtree_window' to determine how to group families
            - 'original': Original method using simple pairwise-distance calculations
            - 'kdtree': KDTree method for faster distance calculations
            - 'adaptive_kdtree_window': KDTree method with a windowed approach for more efficient distance calculations
        segment_duration (int): Duration of each segment in seconds for the windowed approach (i.e. when family_grouping='kdtree_window')


    Output:
        ix (array): Indices of frequencies, times
        pixels_in_families (array): NumPy array of all unique pixel ID's that are in families
        families (list): each entry contains unique pixel ID's in that family
    
    Note: ix and pixels_in_families are provided for plotting purposes, such that:
    x = np.zeros(S.shape)
    x[ix[0][pixels_in_families],ix[1][pixels_in_families]] = 1   # Makes a mask where 1 means plot value
    ---------------------------------------------------------------------------------------------------------'''
    if family_grouping == 'adaptive_kdtree_window':
        T_zeroed = T - np.min(T)
        total_duration = np.max(T_zeroed)
        num_segments = int(np.ceil(total_duration / segment_duration))
        segment_edges = np.arange(0, total_duration + segment_duration, segment_duration)
        filtered_indices = []

        for i in range(num_segments):
            start, end = segment_edges[i], segment_edges[i + 1]
            time_mask = (T_zeroed >= start) & (T_zeroed < end)

            if np.sum(time_mask) == 0:
                continue

            if p_threshold is not None:
                semblance_values = _compute_percentile(S[:, time_mask], p_threshold)[:, None]
                cond = (S[:, time_mask] >= semblance_values)
            else:
                cond = (S[:, time_mask] >= semblance_threshold)

            if GT_baz is not None:
                cond &= (B[:, time_mask] > GT_baz - baz_dev) & (B[:, time_mask] < GT_baz + baz_dev)
            if expected_vel is not None:
                cond &= (V[:, time_mask] > expected_vel - vel_dev) & (V[:, time_mask] < expected_vel + vel_dev)

            ix = np.where(cond)
            if ix[0].size == 0:
                continue

            global_ix = (ix[0], np.where(time_mask)[0][ix[1]])
            filtered_indices.append(global_ix)

        filtered_indices = [idx for idx in filtered_indices if isinstance(idx, tuple) and len(idx[0]) > 0 and len(idx[1]) > 0]
        if not filtered_indices:
            return np.array([]), np.array([]), []

        filtered_indices = tuple(np.concatenate(idx) for idx in zip(*filtered_indices))
        S_ix, B_ix, V_ix = S[filtered_indices], B[filtered_indices], V[filtered_indices]
        F_ix, T_ix = f_bands['fcenter'].values[filtered_indices[0]], T[filtered_indices[1]]

        sort_idx = np.argsort(T_ix)
        ix_f, ix_t = filtered_indices[0][sort_idx], filtered_indices[1][sort_idx]
        B_ix, T_ix, F_ix, S_ix, V_ix = B_ix[sort_idx], T_ix[sort_idx], F_ix[sort_idx], S_ix[sort_idx], V_ix[sort_idx]

        points = np.column_stack((ix_f / sigma_f, ix_t / sigma_t, B_ix / sigma_b))
        tree = KDTree(points)
        assoc_pairs = np.array(list(tree.query_pairs(dist_threshold)))

    elif family_grouping == 'kdtree':
        if p_threshold is not None:
            semblance_values = _compute_percentile(S, p_threshold)[:, None]
            cond = (S >= semblance_values)
        else:
            cond = (S >= semblance_threshold)

        if GT_baz is not None:
            cond &= (B > GT_baz - baz_dev) & (B < GT_baz + baz_dev)
        if expected_vel is not None:
            cond &= (V > expected_vel - vel_dev) & (V < expected_vel + vel_dev)

        ix = np.where(cond)
        if ix[0].size == 0:
            return np.array([]), np.array([]), []

        S_ix, B_ix, V_ix = S[ix], B[ix], V[ix]
        F_ix, T_ix = f_bands['fcenter'].values[ix[0]], T[ix[1]]

        sort_idx = np.argsort(T_ix)
        ix_f, ix_t = ix[0][sort_idx], ix[1][sort_idx]
        B_ix, T_ix, F_ix, S_ix, V_ix = B_ix[sort_idx], T_ix[sort_idx], F_ix[sort_idx], S_ix[sort_idx], V_ix[sort_idx]

        points = np.column_stack((ix_f / sigma_f, ix_t / sigma_t, B_ix / sigma_b))
        tree = KDTree(points)
        assoc_pairs = np.array(list(tree.query_pairs(dist_threshold)))

    elif family_grouping == 'original':
        if p_threshold is not None:
            semblance_values = _compute_percentile(S, p_threshold)[:, None]
            cond = (S >= semblance_values)
        else:
            cond = (S >= semblance_threshold)

        if GT_baz is not None:
            cond &= (B > GT_baz - baz_dev) & (B < GT_baz + baz_dev)
        if expected_vel is not None:
            cond &= (V > expected_vel - vel_dev) & (V < expected_vel + vel_dev)

        ix = np.where(cond)
        S_ix, B_ix, V_ix = S[ix], B[ix], V[ix]
        F_ix, T_ix = f_bands['fcenter'].values[ix[0]], T[ix[1]]
        ix_f, ix_t = ix[0], ix[1]

        ix_sort_time = np.argsort(ix_t)
        ix_f, ix_t = ix_f[ix_sort_time], ix_t[ix_sort_time]
        B_ix, T_ix, F_ix, S_ix, V_ix = B_ix[ix_sort_time], T_ix[ix_sort_time], F_ix[ix_sort_time], S_ix[ix_sort_time], V_ix[ix_sort_time]

        assoc_pairs = []
        for i in range(len(B_ix)):
            if i + 1 >= len(B_ix):
                break
            d1 = (ix_f[i] - ix_f[i+1:i+n_forward])**2 / sigma_f**2
            d2 = (ix_t[i] - ix_t[i+1:i+n_forward])**2 / sigma_t**2
            d3 = (B_ix[i] - B_ix[i+1:i+n_forward])**2 / sigma_b**2
            d = np.sqrt(d1 + d2 + d3)
            ixd = i + 1 + np.where(d <= dist_threshold)[0]
            assoc_pairs.extend([(i, j) for j in ixd])
        assoc_pairs = np.array(assoc_pairs)

    else:
        raise ValueError(f"Unsupported family_grouping: {family_grouping}. Choose 'original', 'kdtree', or 'adaptive_kdtree_window'.")

    G = nx.Graph()
    G.add_edges_from(assoc_pairs)
    components = list(nx.connected_components(G))
    families = [list(c) for c in components if len(c) > min_pixels]
    pixels_in_families = np.concatenate(families) if families else np.array([], dtype=int)

    detections = np.array([
        [ref_time + np.min(T_ix[f]) / 86400. for f in families],
        [ref_time + np.max(T_ix[f]) / 86400. for f in families],
        [np.min(F_ix[f]) for f in families],
        [np.max(F_ix[f]) for f in families],
        [np.mean(B_ix[f]) for f in families],
        [np.std(B_ix[f]) for f in families],
        [np.mean(V_ix[f]) for f in families],
        [np.std(V_ix[f]) for f in families],
        [np.max(S_ix[f]) for f in families],
        [len(f) for f in families]
    ]) if families else np.empty((10, 0))

    return (ix_f, ix_t), pixels_in_families, detections

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def df_families(ref_time, families):
    '''---------------------------------------------------------------------------------------------------------
    Returns a Pandas dataframe of families, sorted by start time, and with time in seconds
    for direct comparison with results plotted with plot_sliding_window_multifreq

    Input:
        ref_time: A Matplotlib datenumber containing the reference time for T
        families (list): each entry contains unique pixel ID's in that family - output from make_families

    Output:
        df (Pandas DataFrame): dataframe of families sorted by time
    ---------------------------------------------------------------------------------------------------------'''
    families_df = families.copy()
    families_df[0,:] = (families_df[0,:] - ref_time)*86400
    families_df[1,:] = (families_df[1,:] - ref_time)*86400

    df = pd.DataFrame(data=families_df.transpose(),
                    columns=['start_time','end_time','min_freq','max_freq',
                            'mean_baz','std_baz','mean_vel','std_vel','max_semb','n_pixels'])
    df = df.sort_values('start_time')

    return df

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def write_families_to_db(dbname, families):
    '''---------------------------------------------------------------------------------------------------------
    Writes a set of families (or detections) to a SQLite3 database

    Input:
        dbname (str): filename to save families to
        families (list): each entry contains unique pixel ID's in that family - output from make_families
    ---------------------------------------------------------------------------------------------------------'''
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

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def read_families_from_db(dbname):
    '''---------------------------------------------------------------------------------------------------------
    Reads families from a SQLite3 database

    Input:
        dbname (str): filepath to SQLite3 database containing families detections

    Output:
        detections (array): families detection characteristics
    ---------------------------------------------------------------------------------------------------------'''
    conn = sqlite3.connect(dbname)
    c = conn.cursor()

    detections = c.execute('select start_time, end_time, min_freq, max_freq, mean_baz, std_baz, mean_vel, std_vel, max_sem, n_pixels from detect').fetchall()
    conn.close()

    detections = np.array(detections).transpose()

    return detections
'----------------------------------------------'
'---------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------'
'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------'
'---------------------------------------------------------------------------'
'----------------------------------------------'
############### ############### ###############
############## Plotting Functions #############
############### ############### ###############

def plot_array_data(data_filepath, array_coords_filepath, event_time=None, source_lat=None, source_lon=None, parameter_table=None, array=None, trim_stream=None, amp_lim=None, amp_units='Amplitude', 
                    channel=None, bandpass=[0.5,5], taper='cosine', max_percentage=0.05, max_length=60, remove_stations=None, convert_units=None, plot=True, equal_scale=True, interpolate_gaps=False, 
                    fname_plot=None, figsize=(1200,800)):
    '''----------------------------------------------------------------------------------------------------------------------------------
    Plots array data in UTC time

    Input:
        data_filepath (str): filepath where data are stored (optimized for wfdisc/mseed)
        array_coords_filepath (str): filepath where geographic coordinates of array are stored - columns should be: [station name], [lat], [lon], [elevation]
        event_time (str): origin time of event ("YYYY-MM-DDTHH:MM:SS")
        source_lat (float/int): latitude of event
        source_lon (float/int): longitude of event
        parameter_table (str): filepath to precomputed spreadsheet with back azimuth, distance, and delay times for stations/arrays - columns necessary in spreadsheet:
                                    ['IMS_Station' or 'Station' or 'Array'], ['Back_Azimuth [deg.]'], ['Distance [km]'], 
                                    ['ETA_Trop' in UTC], ['ETA_Strat' in UTC], ['ETA_Therm' in UTC], 
                                    ['Delay_Trop [s]'], ['Delay_Strat [s]'], ['Delay_Therm [s]']
        array (str): name of array
        trim_stream (list): both inputs trim stream from start time
        amp_lim (list): range in amplitude for visualization in time-series plot
        amp_units (str): amplitude units of measurement used for labeling y-axis
        channel (str): channel of sensor to retrieve data from
        bandpass (list): frequency range to use to bandpass time-series data (defaults to [0.5,5])
        taper (str): type of taper to use - defaults to "cosine"
        max_percentage (float/int): percentage to use for taper
        max_length (float/int): length to use for taper
        remove_stations (list): each entry is station name to remove from array
        convert_to_units (list): the value necessary to convert counts to units (Pa, m/s, etc.), assumes units of Pa/counts (if it's counts/Pa input value as 1/convert_to_units)
        plot (boolean): whether to plot array data
        equal_scale (boolean): whether to make each trace the same scale in plot
        interpolate_gaps (boolean): whether to interpolate gaps in array data
        fname_plot (str): filename to save data quality plot
        figsize (tuple): if plot, specifies figure size

    Output:
        st: unfiltered ObsPy stream
        st_filt: filtered ObsPy stream
        delay_times (list): predicted tropospheric/stratospheric/thermospheric arrivals in seconds relative to event time and location (must input parameter_table or event time)
        baz (float): ground truth back azimuth relative to event location (must input at least source lat and source lon)
        distance (float): distance from array to event in kilometers (must input at least source lat and source lon)
    ----------------------------------------------------------------------------------------------------------------------------------'''
    # Reading in data
    try:
        try:
            if channel is not None:
                st = read(data_filepath).select(channel=channel)
            else:
                st = read(data_filepath)
        except:
            # if a directory is chosen as input
            st = Stream()
            for filename in os.listdir(data_filepath):
                f = os.path.join(data_filepath, filename)
                try:
                    if channel is not None:
                        tr = read(f).select(channel=channel)
                    else:
                        tr = read(f)
                    st.append(tr[0])
                except:
                    continue
    except: # if wfdisc
        if channel is not None:
            st = read_wfdisc(data_filepath).select(channel=channel)
        else:
            st = read_wfdisc(data_filepath)
    try:
        st = st.merge()
    except:
        calibs = np.zeros((len(st)))
        for i, tr in enumerate(st):
            calibs[i] = tr.stats.calib
        if len(np.unique(calibs)) > 1:
            for tr in st:
                tr.stats.calib == 1
        st = st.merge()
    #-----------------------------------------------------------------------------------------------------------------#
    # Converting counts to unit measurement (assumes input is Pa/counts)
    if convert_units is not None:
        if len(convert_units) == 1: # same value for all
            for tr in st:
                tr.data = tr.data*convert_units
        elif len(convert_units) > 1: # or specify value for each trace
            for i, tr in enumerate(st):
                tr.data = tr.data*convert_units[i]
    #-----------------------------------------------------------------------------------------------------------------#
    # Removing stations
    if remove_stations is not None:
        for station in remove_stations:
            for tr in st.select(station=station):
                st.remove(tr)
    #-----------------------------------------------------------------------------------------------------------------#
    # Trimming stream
    if trim_stream is not None:
        dt = st[0].stats.starttime
        st.trim(dt+trim_stream[0], dt+trim_stream[1])
    #-----------------------------------------------------------------------------------------------------------------#
    # Interpolating to fix data gaps
    if interpolate_gaps:
        st = st.merge(); data = []
        for tr in st:
            data.append(tr.data)
        data = np.array(data)
        t_data = np.arange(0, st[0].stats.delta*st[0].stats.npts, st[0].stats.delta)
        for idx, tr in enumerate(st):
            ix = np.where((data[idx,:] < -1e9))[0]
            for i,j in enumerate(ix):
                data[idx,:][j] = np.interp(ix[i], t_data, data[idx,:])
            tr.data = data[idx,:].copy()
    #-----------------------------------------------------------------------------------------------------------------#
    # Appending geographic coordinates
    try:
        st = append_location_info(st, array_coords_filepath)
    except Exception as inst:
        print('Error: '+str(inst) + ' - could not append location info')
    #-----------------------------------------------------------------------------------------------------------------#
    # Filter data and plot stream
    if bandpass is not None:
        st_taper = st.copy()
        st_taper.taper(type=taper, max_percentage=max_percentage, max_length=max_length)
        st_filt = st_taper.copy()
        try:
            st_filt.filter(type='bandpass', freqmin=min(bandpass), freqmax=max(bandpass))
        except:
            st_filt = st_taper.copy().split()
            st_filt.filter(type='bandpass', freqmin=min(bandpass), freqmax=max(bandpass))
            st_filt = st_filt.merge() 
    if plot:
        fig = st_filt.plot(handle=True, method='Full', type='normal', equal_scale=equal_scale, size=figsize)
    if amp_lim is not None:
        for ax in fig.axes:
            ax.set_ylim(amp_lim)
    plt.xlabel('Time [UTC]'); plt.ylabel(amp_units)
    #-----------------------------------------------------------------------------------------------------------------#
    # Plotting precomputed predicted arrivals using spreadsheet - FOR INFRASOUND ONLY
    if (parameter_table is not None) and (event_time is not None):
        # Extracting computed params from analysis spreadsheet
        df_analysis = pd.read_csv(parameter_table)
        try:
            try:
                array_idx = np.where((df_analysis['IMS_Station'] == array))[0]
            except:
                try:
                    array_idx = np.where((df_analysis['Station'] == array))[0]
                except:
                    array_idx = np.where((df_analysis['Array'] == array))[0]
        except Exception as inst:
            print(inst)
            print("Array must be specified if analysis spreadsheet is provided. Column header for station label must be: 'IMS_Station', 'Station', or 'Array' ")
        arrival_times = [df_analysis['ETA_Trop'][array_idx], df_analysis['ETA_Strat'][array_idx], df_analysis['ETA_Therm'][array_idx]]
        baz = df_analysis['Back_Azimuth [deg.]'][array_idx].values; distance = np.round(df_analysis['Distance [km]'][array_idx].values[0],2)
        if baz < 0: baz += 360
        if plot:
            plt.axvline(x=arrival_times[0], lw=1, ls='--', color='red') # trop
            plt.axvline(x=arrival_times[1], lw=1, ls='--', color='green') # strat
            plt.axvline(x=arrival_times[2], lw=1, ls='--',color='blue') # therm
            plt.suptitle('Distance to Event: ' + str(df_analysis['Distance [km]'][array_idx].values[0]) + ' [km]')
        #-----------------------------------------------------------------------------------------------------------------#
        # Returning delay times corrected for stream start time (this way they can be used for array processing results)
        try:
            event_starttime = UTCDateTime(event_time)
        except:
            raise Exception('Event time must be specified if analysis spreadsheet is provided')
        st = st.merge().copy(); correction = st[0].stats.starttime - event_starttime
        delay_times = [df_analysis['Delay_Trop [s]'][array_idx].values - correction, 
                        df_analysis['Delay_Strat [s]'][array_idx].values - correction, 
                        df_analysis['Delay_Therm [s]'][array_idx].values - correction]
    # Computing delay times, GT back azimuth and distance - FOR INFRASOUND ONLY
    elif (parameter_table == None) and (event_time is not None):
        # Extracting year, month, day, hour, minute, second from source time input
        source_time = event_time.replace('T', ' ')
        year = int(source_time.split()[0].split('-')[0]); month = int(source_time.split()[0].split('-')[1]); day = int(source_time.split()[0].split('-')[2])
        hour = int(source_time.split()[1].split(':')[0]); minute = int(source_time.split()[1].split(':')[1]); second = int(source_time.split()[1].split(':')[2])
        source_time = datetime(year,month,day,hour,minute,second)
        #-----------------------------------------------------------------------------------------------------------------------#
        # Computing distance, back azimuth, delay times (trop, strat, therm), and eta's in UTC
        _, baz, distance = g.inv(source_lon, source_lat, st[0].stats.sac.stlo, st[0].stats.sac.stla); distance /= 1000
        if baz < 0: baz += 360
        distance = np.round(distance,4).astype(float)
        trop_delay_time = np.round(distance / 0.340,4) # Tropospheric delay (in s)
        eta_trop = (source_time + datetime_module.timedelta(seconds=trop_delay_time)).strftime("%Y-%m-%d %H:%M:%S") # Tropospheric ETA
        strat_delay_time = np.round(distance / 0.285,4) # Stratospheric delay (in s)
        eta_strat = (source_time + datetime_module.timedelta(seconds=strat_delay_time)).strftime("%Y-%m-%d %H:%M:%S") # Stratospheric ETA
        therm_delay_time = np.round(distance / 0.220,4) # Thermospheric delay (in s)
        eta_therm = (source_time + datetime_module.timedelta(seconds=therm_delay_time)).strftime("%Y-%m-%d %H:%M:%S") # Thermospheric ETA
        plt.axvline(x=UTCDateTime(eta_trop), lw=1, ls='--', color='red') # trop
        plt.axvline(x=UTCDateTime(eta_strat), lw=1, ls='--', color='green') # strat
        plt.axvline(x=UTCDateTime(eta_therm), lw=1, ls='--', color='blue') # therm
        plt.suptitle('Distance to Event: ' + str(distance) + ' [km]')
        #-----------------------------------------------------------------------------------------------------------------------#
        # Returning delay times corrected for stream start time
        event_starttime = UTCDateTime(event_time)
        st = st.merge(); correction = st[0].stats.starttime - event_starttime
        delay_times = [trop_delay_time - correction, 
                        strat_delay_time - correction, 
                        therm_delay_time - correction]
    elif (parameter_table == None) and (event_time == None) and (source_lat is not None) and (source_lon is not None):
        # Computing distance, back azimuth, delay times (trop, strat, therm), and eta's in UTC
        _, baz, distance = g.inv(source_lon, source_lat, st[0].stats.sac.stlo, st[0].stats.sac.stla); distance /= 1000
        if baz < 0: baz += 360
        distance = np.round(distance,4).astype(float)
        plt.suptitle('Distance to Event: ' + str(distance) + ' [km]')
    #-----------------------------------------------------------------------------------------------------------------#
    # Save figure
    if fname_plot is not None:
        fig.savefig(fname_plot)
    #-----------------------------------------------------------------------------------------------------------------------#  
    # Return params  
    if (parameter_table is not None) or (event_time is not None):
        return st.merge(), st_filt.merge(), delay_times, baz, distance
    elif (parameter_table == None) and (event_time == None) and (source_lat is not None) and (source_lon is not None):
        return st.merge(), st_filt.merge(), baz, distance
    else:
        return st.merge(), st_filt.merge()

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def plot_data_quality(st, 
                      ############### Sensor SOH #####################################################
                      amp_units='Amplitude', window='hann', nperseg=2**8, noverlap_percent=50, PSD_xlim=None, PSD_ylim=None, 
                      db_units=True, k=1, min_samples=3, slope_threshold=0.1, 
                      
                      ############### Polarity Check #####################################################
                      bandpass=[0.5,5], taper='cosine', max_percentage=0.05, max_length=60, inverted_polarity_threshold=-0.95, 
                      reverse_polarities=False, t_lim=None, plot_UTC=True, UTC_time_interval=None, legend_loc='upper right', 
                      
                      ############### Data gap #####################################################
                      percent_gap_threshold=20, 

                      return_stream=False, fname_plot=None, figsize=(16,4)):
    '''---------------------------------------------------------------------------------------------------------
    Assess sensor quality using power spectral density and cross-correlation (for inverted polarity - uses first sensor in stream as reference)

    Inputs:
        st: ObsyPy stream object
        amp_units (str): units used to measure amplitude (e.g., Pressure [Pa] - make sure units are in brackets)
        window (str): desired window to use (defaults to hann)
        nperseg (int): length of each segment (defaults to 256 - same as scipy.welch())
        noverlap_percent (int): percent overlap between segments (defaults to 50% same as scipy.signal.welch())
        PSD_xlim (list): frequency limits on PSD plot (defaults to [0.01, Nyquist frequency])
        PSD_ylim (list): power spectral limits on PSD plot
        db_units (boolean): whether to plot the PSD in decibels (common practice for infrasound data - defaults to true)
        k (int): number of standard deviations from the mean to set outlier distance threshold (defaults to 1)
        min_samples (int): minimum number of sensors required to form a cluster in DBSCAN (defaults to 3)
        slope_threshold (int/float): slope to characterize PSD flatness (automatically removes if PSD slope falls below threshold - defaults to 0.25)
        bandpass (list): frequency range to use to bandpass time-series data prior to period measurements (defaults to [0.5,5])
        taper (str): type of taper to use - defaults to "cosine"
        max_percentage (float/int): percentage to use for taper
        max_length (float/int): length to use for taper
        inverted_polarity_threshold (float): negative normalized correlation threshold to characterize polarity inversion (defaults to -0.9)
        reverse_polarities (boolean): whether to reverse polarity on inverted sensor in correlation plot (if True will return stream object with polarity reversal corrected)
        t_lim (list): range in time window start and end times to plot (in seconds relative to start time)
        plot_UTC (boolean): whether to plot time in UTC
        UTC_time_interval (int): time interval (in minutes) to be used for x-axis UTC formatting
        legend_loc (str): relative location of legend in polarity check plot
        percent_gap_threshold (int/float): gap percentage threshold, if exceeded will remove station
        return_stream (boolean): whether to remove faulty sensor identified in PSD plot and return corrected stream
        fname_plot (str): filename to save data quality plot
        figsize (tuple): specifies size of figure
        
    Note: If 3 or more faulty sensors are nearest neighbors, make sure to increase min_samples to 1 greater than the number of faulty sensors that are nearest neighbors
    Anti-aliasing points are removed from PSD (20% aliased) and tapered samples are removed from cross-correlation (5% taper)
    Flat PSDs are automatically removed, if there are not enough sensors for array processing (3) after identifying PSD outliers, it will keep the 3 best sensors (removing extreme outliers)
    ---------------------------------------------------------------------------------------------------------'''
    # Compute PSD
    Pxx = []; f = []
    for tr in st:
        f_tmp, Pxx_tmp = power_spectrum(tr, window=window, nperseg=nperseg, noverlap_percent=noverlap_percent) # compute signal PSD
        Pxx.append(Pxx_tmp); f.append(f_tmp)
    Pxx = np.array(Pxx); f = np.array(f)
    #-----------------------------------------------------------------------------------------------------------------#
    # Checking for flat slope
    flat_idxs = []; flat_stns = []; sampling_rate = st[0].stats.sampling_rate
    for spec_idx in range(Pxx.shape[0]):
        nyquist = sampling_rate / 2
        anti_alias_range = f[spec_idx,:] < (0.8*nyquist)
        f_filt = f[spec_idx,:][anti_alias_range]
        Pxx_filt = Pxx[spec_idx,:][anti_alias_range]
        log_f_tmp = np.log10(f_filt[1:])
        log_Pxx_tmp = np.log10(Pxx_filt[1:])
        slope_tmp, _, _, _, _ = linregress(log_f_tmp, log_Pxx_tmp)
        is_flat = np.abs(slope_tmp) < slope_threshold # returns boolean whether PSD is flat
        if is_flat == True:
            flat_idxs.append(spec_idx)
            flat_stns.append(st[spec_idx].stats.station)
        if sum(Pxx[spec_idx,:]) == 0: # remove if all values are 0
            flat_idxs.append(spec_idx)
            flat_stns.append(st[spec_idx].stats.station)
    # Create new array with flat PSDs removed
    Pxx_new = np.delete(Pxx, flat_idxs, axis=0)
    f_new = np.delete(f, flat_idxs, axis=0)
    st_new = st.copy()        
    for stn in flat_stns:
        st_new.remove(st_new.select(station=stn)[0])
        print('Removing sensor '+str(stn) + ' due to flat PSD')
    if len(st_new) < 3:
        raise Exception('Fewer than 3 sensors left after removing flat PSDs. Array may not be suitable for processing.')
    #-----------------------------------------------------------------------------------------------------------------#
    # Convert to decibels
    if db_units == True:
        Pxx = 10*np.log10(Pxx)
        Pxx_new = 10*np.log10(Pxx_new)
    #-----------------------------------------------------------------------------------------------------------------#
    # Remove anti-aliasing points (only using first 80% of PSD)
    n_cols_Pxx_new = Pxx_new.shape[1]; remove_n_Pxx_new = int(n_cols_Pxx_new * 0.20)
    Pxx_new_DBSCAN = Pxx_new[:, remove_n_Pxx_new:-remove_n_Pxx_new]
    # Compute mean and standard deviation of absolute value of power spectrum
    mean_dist = np.mean(np.abs(Pxx_new_DBSCAN))
    std_dist = np.std(np.abs(Pxx_new_DBSCAN))
    # Set eps (outlier distance threshold) to be k standard deviations from the mean
    outlier_dist_thresh = mean_dist + k * std_dist
    outliers_idx, num_outliers, labels = dbscan_outliers(X=np.abs(Pxx_new_DBSCAN), eps=outlier_dist_thresh, min_samples=min_samples) # identifying outliers using DBSCAN
    #-----------------------------------------------------------------------------------------------------------------#
    # Plotting PSD's
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(1,3,1)
    for psd_idx in range(Pxx_new.shape[0]):
        ax1.semilogx(f_new[psd_idx,:], Pxx_new[psd_idx,:],lw=0.75, ls='-', color='k')
    if len(flat_idxs) > 1:
        for idx in flat_idxs:
            ax1.semilogx(f[idx,:], Pxx[idx,:],lw=0.75, ls='-', color='red', label=st[idx].stats.station)
        plt.legend(loc='upper right')
    if num_outliers > 0:
        for outlier_idx in outliers_idx:
            ax1.semilogx(f_new[outlier_idx,:], Pxx_new[outlier_idx,:],lw=0.75, ls='-', color='red', label=st_new[outlier_idx].stats.station)
        plt.legend(loc='upper right')
    if PSD_xlim is not None:
        ax1.set_xlim([PSD_xlim[0], PSD_xlim[1]])
    else:
        ax1.set_xlim([0.01, st_new[0].stats.sampling_rate/2])
    if PSD_ylim is not None:
        ax1.set_ylim([PSD_ylim[0], PSD_ylim[1]])
    if amp_units == 'Amplitude': # make sure amplitude units are input properly for plotting
        units_str = amp_units
    else:
        try:
            units_str = re.findall(r'\[(.*?)\]', amp_units)[0]
        except:
            raise Exception('Make sure to include amplitude units in brackets (e.g., Pressure [Pa])')
    if db_units == True:
        plt.ylabel('dB Rel. 1 ['+units_str+'\u00b2/Hz]')
    else:
        plt.ylabel('Power Spectrum ['+units_str+'\u00b2/Hz]')
    plt.xlabel('Frequency [Hz]')
    plt.grid(linewidth=0.25, which='both')
    ax1_title_obj = ax1.set_title('Power Spectral Density: Sensor SOH')
    # Store the station names for removal with DBSCAN
    stns_to_remove = []
    for outlier_idx in outliers_idx:
        stns_to_remove.append(st_new[outlier_idx].stats.station)
    #-----------------------------------------------------------------------------------------------------------------#
    # Computing cross-correlation for inverted polarity
    if bandpass is not None:
        st_taper = st_new.copy()
        st_taper.taper(type=taper, max_percentage=max_percentage, max_length=max_length)
        st_filt = st_taper.copy()
        try:
            st_filt.filter(type='bandpass', freqmin=min(bandpass), freqmax=max(bandpass))
        except:
            st_filt = st_taper.copy().split()
            st_filt.filter(type='bandpass', freqmin=min(bandpass), freqmax=max(bandpass))
            st_filt = st_filt.merge() 
    max_trace = max(st_filt, key=lambda tr: tr.stats.npts) # need to make sure to use trace with max data samples
    t = np.arange(0, max_trace.stats.npts*max_trace.stats.delta, max_trace.stats.delta)
    data_array_tmp = []
    for tr in st_filt:
        tr_data_tmp = tr.data / np.max(np.abs(tr.data)) # normalize by trace
        data_array_tmp.append(tr_data_tmp)
    # Check if sensors have same number of data samples
    max_len = max(len(x) for x in data_array_tmp)
    data_array_tmp = [x for x in data_array_tmp if len(x) == max_len] # keep anything equal to the max length
    data_array_tmp = np.array(data_array_tmp); data_array = []
    for tr_idx in range(data_array_tmp.shape[0]):
        t, tr_idx_data = fix_lengths(t, data_array_tmp[tr_idx,:]) # will be using this time vector
        data_array.append(tr_idx_data)
    data_array = np.array(data_array)    
    # Specify window length and window step
    global_start = min(tr.stats.starttime for tr in st_new)
    global_end = max(tr.stats.endtime for tr in st_new)
    win_len = (global_end-global_start) // 100; step = win_len * 0.1 # use time window equal to 1/100th of total data length in seconds
    # Step through time-series recordings
    max_xcorrs = []; max_lags = []; times = []
    for win_start in np.arange(t.min(), t.max()+win_len, step):
        if win_start + win_len > t.max()+win_len:
            break
        # Extract data
        max_xcorrs_tmp = []; lags_tmp = []
        t_win, data_array_win = data_time_window(t, data_array, t_start=win_start, t_end=win_start+win_len)
        for tr_idx in range(data_array.shape[0]):
            if tr_idx == 0:
                continue
            else:
                lags, xcorrs = norm_xcorr(t_win, data_array_win[0,:], data_array_win[tr_idx,:]) # use first sensor as reference
                max_xcorr = xcorrs[np.argmax(np.abs(xcorrs))]
                lag = lags[np.argmax(np.abs(xcorrs))]
                max_xcorrs_tmp.append(max_xcorr); lags_tmp.append(lag)
        max_xcorrs.append(np.array(max_xcorrs_tmp)); max_lags.append(np.array(lags_tmp)); times.append(win_start)
    max_xcorrs = np.array(max_xcorrs).T; max_lags = np.array(max_lags).T; times = np.array(times).reshape(1,len(times))
    # Removing first and last 5% of samples
    n_cols = max_xcorrs.shape[1]; remove_n = int(n_cols * 0.05); n_cols_t = len(t); remove_t = int(n_cols_t * 0.05)
    max_xcorrs = max_xcorrs[:, remove_n:-remove_n]; max_lags = max_lags[:, remove_n:-remove_n]; times = times[:, remove_n:-remove_n]; t = t[remove_t:-remove_t]
    #-----------------------------------------------------------------------------------------------------------------#
    # Plot
    ax2 = fig.add_subplot(1,3,2)
    for xcorr_idx in range(max_xcorrs.shape[0]):
        plt.plot(times[0,:], max_xcorrs[xcorr_idx,:], color='k')
    rows_with_condition = np.any(max_xcorrs < inverted_polarity_threshold, axis=1)
    row_idxs = np.where(rows_with_condition)[0]
    for idx in row_idxs:
        plt.plot(times[0,:], max_xcorrs[idx,:], color='red', label=st_new[idx+1].stats.station) # need to add plus 1 here to the stream since we are removing first sensor from calculations
        if reverse_polarities == True:
            st_new[idx+1].data *= -1
            print('Sensor ' + st_new[idx+1].stats.station + ' polarity corrected')
    if len(row_idxs) > 0:
        plt.legend(loc=legend_loc)
    if (plot_UTC == True) and (t_lim == None):
        # Plot UTC - round down to the nearest minute_interval for start
        if UTC_time_interval is not None:
            minute_interval = UTC_time_interval
        else:
            if global_end-global_start <= 3600*2:
                minute_interval = 15 # less than 2 hours make it 15 minute intervals
            else:
                # round to the nearest 15 minutes and partition data stream into 5 segements if stream is longer than 2 hours
                n = global_end-global_start
                sec_round = round(n / 900) * 900
                min_round = sec_round // 60
                minute_interval = min_round // 6
        rounded_start_tick = global_start.replace(minute=(global_start.minute // minute_interval) * minute_interval, second=0, microsecond=0)
        if rounded_start_tick < global_start:
            rounded_start_tick += (minute_interval*60)
        x_tick_positions = np.arange(rounded_start_tick - global_start, global_end - global_start, minute_interval*60)
        x_tick_labels = [(global_start + t).strftime('%H:%M:%S') for t in x_tick_positions]
        ax2.set_xticks(x_tick_positions); ax2.set_xticklabels(x_tick_labels)
        ax2.set_xlim([0,(global_end-global_start)]); ax2.set_xlabel(global_start.strftime('%Y-%m-%d')+' Time [UTC]')
    elif (plot_UTC == False) and (t_lim is not None):
        ax2.set_xlim(t_lim)
        ax2.set_xlabel('Time [s] after ' + str(global_start).split('.')[0])
    elif (plot_UTC == False) and (t_lim == None):
        ax2.set_xlim([t[0],t[-1]])
        ax2.set_xlabel('Time [s] after ' + str(global_start).split('.')[0])
    elif (plot_UTC == True) and (t_lim is not None):
        raise Exception("Can't have t_lim and plot_UTC set to True, since t_lim is relative.")
    ax2.set_ylim([-1.05,1.05])
    plt.ylabel('Normalized Correlation Coefficient')
    ax2.set_title('Polarity Check\nReference Station: '+st_new[0].stats.station)
    #-----------------------------------------------------------------------------------------------------------------#
    # Data gaps
    total_duration = global_end - global_start
    # Threshold calculation
    threshold = (percent_gap_threshold/100) * total_duration
    stns_masked = []; stns_to_remove_data_gap = []
    filtered_st = st_new.copy()
    for tr in filtered_st:
        if isinstance(tr.data, np.ma.MaskedArray):
            filtered_st.remove(tr) # remove if masked
            stns_masked.append(tr.stats.station)
    gap_data = {} # this is for data gap plot
    if len(filtered_st) > 0:
        gaps = filtered_st.get_gaps()
        gap_durations = {tr.id: 0 for tr in filtered_st}
        #-----------------------------------------------------------------------------------------------------------------#
        # Add internal gaps detected by `get_gaps()`
        for gap in gaps:
            trace_id = f"{gap[0]}.{gap[1]}.{gap[2]}.{gap[3]}"  # Network.Station.Location.Channel
            gap_durations[trace_id] += float(gap[4]) # gap[4] is the gap length in seconds
        #-----------------------------------------------------------------------------------------------------------------#
        # Check for start-time and end-time gaps
        for tr in filtered_st:
            trace_id = tr.id
            # Gap at the start (trace starts late)
            if tr.stats.starttime > global_start:
                gap_durations[trace_id] += (tr.stats.starttime - global_start)
            # Gap at the end (trace ends early)
            if tr.stats.endtime < global_end:
                gap_durations[trace_id] += (global_end - tr.stats.endtime)
            # Get gap indices for plotting
            gap_indices = []
            for gap_start, gap_end in gaps:
                # Check if gap intersects with current trace
                if (gap_start < tr.stats.endtime and gap_end > tr.stats.starttime):
                    overlap_start = max(gap_start, tr.stats.starttime)
                    num_gap_samples = int(overlap_start * tr.stats.sampling_rate)
                    gap_indices_tmp = np.arange(num_gap_samples)
                    gap_indices.extend(gap_indices_tmp)
            if len(gap_indices) > 0:
                gap_data[tr.stats.station] = np.array(gap_indices) # append to empty dictionary if there are gaps with keys as station names
        # Filter traces that have gaps ≥ percent_gap_threshold of the stream duration
        filtered_st = Stream(tr for tr in filtered_st if gap_durations[tr.id] < threshold)
        gap_st = Stream(tr for tr in filtered_st if gap_durations[tr.id] > threshold)
        if len(gap_st) > 0:
            for tr in gap_st:
                stns_to_remove_data_gap.append(tr.stats.station)
        for trace_id, gap_time in gap_durations.items():
            if gap_time >= threshold:
                print(f"Removing {trace_id} due to excessive gaps ({gap_time:.2f}s ≥ {threshold:.2f}s)")
    # Return masked traces to stream
    if len(stns_masked) > 0:
        for stn in stns_masked:
            filtered_st.append(st.select(station=stn)[0])
    # Compute masked gaps
    masked_gap_data = {} # this is for data gap plot
    for tr in filtered_st:
        if isinstance(tr.data, np.ma.MaskedArray):
            trace_id = tr.id
            # Compute gaps within masked trace
            masked_gap_time, masked_gap_indices = compute_masked_gaps(tr)
            # Need to also compute gaps outside of masked trace if start and end times don't match global start and end times
            if tr.stats.starttime != global_start:
                start_masked_gap_time = (tr.stats.starttime - global_start)
                masked_gap_time += start_masked_gap_time
                num_start_gap_samples = int(start_masked_gap_time * tr.stats.sampling_rate)
                start_gap_indices = np.arange(num_start_gap_samples)
                masked_gap_indices = np.concatenate((start_gap_indices, masked_gap_indices))
            if tr.stats.endtime != global_end:
                end_masked_gap_time = (global_end - tr.stats.endtime)
                masked_gap_time += end_masked_gap_time
                num_end_gap_samples = int(end_masked_gap_time * tr.stats.sampling_rate)
                start_index = len(tr.data)
                end_gap_indices = np.arange(start_index, start_index + num_end_gap_samples)
                masked_gap_indices = np.concatenate((masked_gap_indices, end_gap_indices))
            masked_gap_data[tr.stats.station] = masked_gap_indices # append to empty dictionary where key is station name
            if masked_gap_time > threshold:
                stns_to_remove_data_gap.append(tr.stats.station)
                filtered_st.remove(tr)
                print(f"Removing {trace_id} due to excessive gaps ({masked_gap_time:.2f}s ≥ {threshold:.2f}s)")   
        else:
            continue # move on if not masked trace
            # Key is trace id and values are masked_gap_indices
    st_new = filtered_st.copy() # new stream with stations removed that exceed gap threshold
    #-----------------------------------------------------------------------------------------------------------------#
    # Find the latest start time and earliest end time
    latest_start = max(tr.stats.starttime for tr in st_new)
    earliest_end = min(tr.stats.endtime for tr in st_new)
    st_new.trim(starttime=latest_start, endtime=earliest_end)
    #-----------------------------------------------------------------------------------------------------------------#
    # Plot data gaps (use all traces!)
    ax3 = fig.add_subplot(1,3,3, sharex=ax2)
    if (gap_data != {}) and (masked_gap_data == {}):
        pass
    elif (gap_data != {}) and (masked_gap_data != {}):
        gap_data.update(masked_gap_data)
    elif (gap_data == {}) and (masked_gap_data == {}):
        pass
    elif (gap_data == {}) and (masked_gap_data != {}):
        gap_data.update(masked_gap_data) 
    t = np.arange(int((total_duration) * st[0].stats.sampling_rate)) / st[0].stats.sampling_rate
    k = 0; ticks = []; labels = []; labels_right = []; total_samples = int((total_duration)*st[0].stats.sampling_rate)
    for idx, tr in enumerate(st.sort()):
        ax3.plot(t, np.full(len(t), k), color='k', lw=5, zorder=1)
        if any(tr.stats.station in key for key in gap_data.keys()): # if station has gaps then plot them in red
            ax3.scatter(t[gap_data[tr.stats.station]], np.full(len(t), k)[gap_data[tr.stats.station]], color='red', marker='o', s=15, zorder=2)
            gap_percentage = (len(gap_data[tr.stats.station]) / total_samples) * 100
            data_percentage = str(round(100-gap_percentage,3))
        else:
            data_percentage = str(float(round(100,3)))
        ticks.append(k); labels.append(tr.stats.station); labels_right.append(data_percentage); k += 0.2
    ax3.set_yticks(ticks)
    ax3.set_yticklabels(labels)
    ax3.set_ylabel('Station')
    ax3_right = ax3.secondary_yaxis('right')
    ax3_right.set_yticks(ticks)
    ax3_right.set_yticklabels(labels_right)
    ax3_right.set_ylabel('Data Availability')
    if (plot_UTC == True) and (t_lim == None):
        ax3.set_xlabel(global_start.strftime('%Y-%m-%d')+' Time [UTC]')
    elif (plot_UTC == False) and (t_lim is not None):
        ax3.set_xlabel('Time [s] after ' + str(global_start).split('.')[0])
    elif (plot_UTC == False) and (t_lim == None):
        ax3.set_xlabel('Time [s] after ' + str(global_start).split('.')[0])
    elif (plot_UTC == True) and (t_lim is not None):
        raise Exception("Can't have t_lim and plot_UTC set to True, since t_lim is relative.")
    ax3.set_title('Data Gaps')
    #-----------------------------------------------------------------------------------------------------------------#
    # Return corrected stream
    if return_stream == True:
        if len(st_new) >= 3: # this means after flat PSDs and data gap we stil have enough sensors for an array
            if num_outliers > 0: # no DBSCAN outliers - leave remaining sensors
                ax1_current_title = ax1_title_obj.get_text()
                # Need to merge DBSCAN and data gap stations to remove lists to get an updated number of outliers
                stns_to_remove = stns_to_remove + stns_to_remove_data_gap
                counts = Counter(stns_to_remove)
                stns_to_remove = [x for x in stns_to_remove if counts[x] == 1] # remove duplicate stations (i.e., found in both DBSCAN outliers and data gap)
                if len(stns_to_remove) == 0: # this means all DBSCAN outliers were removed due to data gaps
                    # Save figure
                    if fname_plot is not None:
                        fig.savefig(fname_plot)
                    return st_new
                # Need to update num_outliers to add new stations removed by data gap or remove previous stations from DBSCAN that were removed by data gap
                num_outliers = len(stns_to_remove)
                if (len(st_new) - num_outliers) >= 3: # if we still have 3 or more stations for array processing
                    for stn in stns_to_remove:
                        try:
                            st_new.remove(st_new.select(station=stn)[0]) # no need to remove data gap stations - they were already removed
                            print('Removing sensor '+str(stn) + ' due to PSD')
                        except:
                            pass
                    # Save figure
                    if fname_plot is not None:
                        fig.savefig(fname_plot)
                    return st_new
                elif (len(st_new) - num_outliers) < 3: # if we don't have enough stations for array processing, only remove most extreme outliers until we have 3 sensors
                    print('Not enough sensors left in array after PSD analysis, only removing extreme outliers or keeping all sensors if only 3 remain.')
                    num_to_remove = len(st_new) - 3
                    if num_to_remove == 0:
                        ax1_new_title = ax1_current_title + '\nReturning all sensors for array processing'
                        ax1_title_obj.set_text(ax1_new_title)
                        # Save figure
                        if fname_plot is not None:
                            fig.savefig(fname_plot)
                        return st_new
                    else:
                        # Re-compute PSD using stream with flat PSDs and data gaps removed
                        Pxx = []; f = []
                        for tr in st_new:
                            f_tmp, Pxx_tmp = power_spectrum(tr, window=window, nperseg=nperseg, noverlap_percent=noverlap_percent)
                            Pxx.append(Pxx_tmp); f.append(f_tmp)
                        Pxx = np.array(Pxx); f = np.array(f)
                        #-----------------------------------------------------------------------------------------------------------------#
                        # Convert to decibels
                        if db_units == True:
                            Pxx = 10*np.log10(Pxx)
                        #-----------------------------------------------------------------------------------------------------------------#
                        # Remove anti-aliasing points (only using first 80% of PSD)
                        n_cols_Pxx = Pxx.shape[1]; remove_n_Pxx = int(n_cols_Pxx * 0.20)
                        Pxx_DBSCAN = Pxx[:, remove_n_Pxx:-remove_n_Pxx]
                        # Compute mean and standard deviation of absolute value of power spectrum
                        mean_dist = np.mean(np.abs(Pxx_DBSCAN))
                        std_dist = np.std(np.abs(Pxx_DBSCAN))
                        # We've already defined our outlier distance threshold
                        _, _, labels = dbscan_outliers(X=np.abs(Pxx_DBSCAN), eps=outlier_dist_thresh, min_samples=min_samples) # only need labels here
                        inlier_mask = labels != -1
                        if np.any(inlier_mask): # if we have inliers
                            centroid = np.median(Pxx_DBSCAN[inlier_mask], axis=0)
                        else:
                            centroid = np.median(Pxx_DBSCAN, axis=0)
                        # Compute distances from sensors to centroid
                        distances = cdist(Pxx_DBSCAN, centroid.reshape(1,-1)).flatten()
                        remove_indices = np.argsort(distances)[-num_to_remove:]
                        # Keep closest 3 sensors
                        keep_mask = np.ones(len(st_new), dtype=bool)
                        keep_mask[remove_indices] = False
                        remove_mask_idxs = np.where((keep_mask == False))[0]
                        keep_mask_idxs = np.where((keep_mask == True))[0]
                        stns_to_remove = []; stns_to_keep = []
                        for remove_idx in remove_mask_idxs:
                            stns_to_remove.append(st_new[remove_idx].stats.station)
                        for keep_idx in keep_mask_idxs:
                            stns_to_keep.append(st_new[keep_idx].stats.station)
                        ax1_new_title = ax1_current_title + '\nReturning '+str(stns_to_keep)+' sensors for array processing'
                        ax1_title_obj.set_text(ax1_new_title)
                        for stn in stns_to_remove:
                            print('Removing sensor ' +str(stn))
                            st_new.remove(st_new.select(station=stn)[0])
                        # Save figure
                        if fname_plot is not None:
                            fig.savefig(fname_plot)
                        return st_new
            else:
                if fname_plot is not None:
                    fig.savefig(fname_plot)
                return st_new
        elif len(st_new) < 3:
            if fname_plot is not None:
                fig.savefig(fname_plot)
            raise Exception('Not enough stations left after removing flat PSDs and/or excessive data gaps')
    else:
        if fname_plot is not None:
            fig.savefig(fname_plot)

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def plot_source_receiver_map(ev_lat, ev_lon, arr_lat, arr_lon, extent=[122.5, 131, 42, 32.5], projection=ccrs.PlateCarree(), transform=ccrs.Geodetic(), gl_lw=0.5, gl_ls='--', 
                             add_axes_labels=True, draw_labels=True, legend_loc='lower left', title=None, title_size=10.5, markersize=5, legend_fontsize=7.5, markerscale=2,
                             array_label=None, event_label=None, figsize=(9,6)):
    '''----------------------------------------------------------------------------------------------------------------------------------
    Plots geographic coordinates of source and receiver

    Input:
        ev_lat (float/int): event latitude
        ev_lon (float/int): event longitude
        arr_lat (float/int): array latitude
        arr_lon (float/int): array longitude
        extent (list): longitude exent (first two numbers) and latitude extent (last two numbers) of plot
        projection: type of map projection to use from cartopy
        transform: type of coordinate system to use to transform between different projections in cartopy
        gl_lw (float/int): grid linewidth to use for plot
        gl_ls (str): linestyle
        add_axes_labels (boolean): whether to add labels to each axis of the plot
        draw_labels (boolean): whether to draw labels within the gridlines
        legend_loc (str): location of legend within plot
        title (str): title of plot
        title_size (float/int): specifies size of title
        markersize (float/int): specifies size of markers in plot
        legend_fontsize (float/int): specifies size of font in legend
        markerscale (float/int): specifies size of marker in legend
        array_label (str): name of array - will plot next to array marker
        figsize (tuple): specifies size of figure
    ----------------------------------------------------------------------------------------------------------------------------------'''
    # Define the figure
    fig, axs = plt.subplots(nrows=1,ncols=1,
                            subplot_kw={'projection': projection},
                            figsize=figsize)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Plotting event
    if event_label is not None: pass
    else: event_label='Event'
    axs.plot(ev_lat, ev_lon, ls='none', marker='*', mec='k', mfc='red', markersize=markersize, markeredgewidth=1, label=event_label)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Plotting array
    if array_label is not None: pass
    else: array_label='Array'
    axs.plot(arr_lat, arr_lon, ls='none', marker='^', mec='k', mfc='blue', markersize=markersize, markeredgewidth=1, label=array_label)
    #-----------------------------------------------------------------------------------------------------------------------#
    axs.set_extent(extent, crs=transform)
    if add_axes_labels == True:
        gl = axs.gridlines(crs=projection, draw_labels=draw_labels, linewidth=0)
        gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
        gl.top_labels = False; gl.right_labels = False
    axs.add_feature(cartopy.feature.LAKES, alpha=0.5)
    axs.add_feature(cartopy.feature.RIVERS, alpha=0.5)
    axs.add_feature(cartopy.feature.OCEAN, alpha=0.5, color='grey')
    axs.add_feature(cartopy.feature.BORDERS, alpha=0.2)
    axs.add_feature(cartopy.feature.LAND, alpha=0.2, color='white')
    axs.add_feature(cartopy.feature.COASTLINE)
    axs.gridlines(lw=gl_lw, ls=gl_ls)
    plt.legend(loc=legend_loc, fontsize=legend_fontsize, markerscale=markerscale)
    if title == None:
        axs.set_title('Source-Receiver Map', size=title_size)
    else:
        axs.set_title(title, size=title_size)

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def plot_array_coords(X, stnm, x_lim=None, y_lim=None, figsize=(9,6), units='m'):
    '''----------------------------------------------------------------------------------------------------------------------------------
    Plots the array coordinates

    Input:
        X (array): coordinates of array relative to a reference station
        stnm (list): names of array elements
        figsize (tuple): specifies size of figure
        units (str): specifies scale of plot - options are "m" and "km"
    ----------------------------------------------------------------------------------------------------------------------------------'''
    fig = plt.figure(figsize=figsize)
    plt.plot(X[:,0], X[:,1], '.')
    for i in range(0, len(stnm)):
        plt.text(X[i,0], X[i,1], stnm[i])
    plt.grid(lw=0.25)
    if units == 'km':
        plt.xlabel('X [km]')
        plt.ylabel('Y [km]')
    elif units == 'm':
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
    else:
        print('Unrecognized units (Options are "km" and "m")')
    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def plotFK(st, startTime, endTime, frqlow, frqhigh,
           sll_x=-3.6, slm_x=3.6, sll_y=-3.6, slm_y=3.6, sl_s=0.18,
           plot=True, normalize=True, sl_corr=[0.,0.], show_peak=False,
           cmap='viridis'):
    '''----------------------------------------------------------------------------------------------------------------------------------
    Computes and displays an FK plot for an ObsPy Stream object

    Input:
        st: ObsPy stream object
        startTime (UTCDateTime object): start time of stream
        endTime (UTCDateTime object): end time of stream
        frqlow, frqhigh (float/int): frequency range used for processing
        sll_x, slm_x (float/int): extent of x-axis in slowness grid
        sll_y, slm_y (float/int): extent of y-axis in slowness grid
        sl_s (float/int): slowness grid resolution
        sl_corr (array): specified correction for slowness
        plot (boolean): whether to plot FK
        normalize (boolean): whether to normalize the data in the time window before running FK
        show_peak (boolean): whether to show the peak of the FK
        cmap (str): colormap to be used for plotting

    Output:
        relpow_map (array): relative power map of FK
        baz (float/int): back azimuth corresponding to peak
        vel (float/int): trace velocity corresponding to peak
    ----------------------------------------------------------------------------------------------------------------------------------'''
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

    geometry = get_geometry(stream, coordsys=coordsys)

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
        plt.xlabel('Slowness x [s/km]')
        plt.ylabel('Slowness y [s/km]')
        plt.title('Peak semblance at ' + str(round(baz % 360., 2)) + ' [deg.] ' + str(round(1/slow, 2)) + ' [km/s]')

    # only flipping left-right, when using imshow to plot the matrix is takes points top to bottom
    # points are now starting at top-left in row major
    return np.fliplr(relpow_map.transpose()), baz % 360, 1. / slow

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def plot_sliding_window(st, element, T, B, V, C=None, v_min=0, v_max=5., 
                        semblance_threshold=None, twin_plot=None, clim=[0,1], figsize=(9,5)):
    '''----------------------------------------------------------------------------------------------------------------------------------
    Plots the results of sliding-window array processing within a single frequency band
    
    Input:
        st: ObsPy stream object containing array data
        element (str): name of the element to plot the time series data for
        T (array): timestamps of array processing estimates (center of time windows) (s)
        B (array): back azimuths
        V (array): trace velocities (km/s)
        C (array): optional color of points (e.g., Semblance, F-statistic, Correlation)
        v_min, v_max (float/int): range in trace velocity for y-axis to plot
        semblance_threshold (float/int): filter out any points below set coherence threshold
        twin_plot (list): start and end times (in seconds) to plot
        clim (list): range in C parameter to plot
        figsize (tuple): specifies figure size
    ----------------------------------------------------------------------------------------------------------------------------------'''
    tr = st.select(station=element)[0]

    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(3,1,1)
    t_tr = np.arange(0, tr.stats.npts*tr.stats.delta, tr.stats.delta)
    plt.plot(t_tr, tr.data/np.max(np.abs(tr.data)), 'k-')
    ax1.tick_params(labelbottom=False)

    ax2 = fig.add_subplot(3,1,2, sharex=ax1)
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

    ax3 = fig.add_subplot(3,1,3, sharex=ax1)
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
    ax3.set_xlabel('Time [s] after ' + str(tr.stats.starttime).split('.')[0].replace('T', ' '))
    
    plt.xlim([t_tr[0], t_tr[len(t_tr)-1]])

    ax1.get_yaxis().set_ticks([])

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def plot_sliding_window_multifreq(st, f_bands, T, B, V, S, 
                                ############### Time-series #####################################################
                                normalize=False, beamform_data=False, bandpass=[0.5,5], taper='cosine', 
                                max_percentage=0.05, max_length=60, amp_units='Amplitude', legend_loc='upper right', 
                                t_lim=None, amp_lim=None, delay_times=None, element=None, return_beam=False,

                                ############### Aggregator ######################################################
                                ix=None, pixels_in_families=None, families=None, families_table=None, family_idx=None,

                                ############### Processing results ##############################################
                                semblance_threshold=0.7, clim_baz=None, clim_vtr=[0,1], log_freq=True, cmap_cyclic='twilight', 
                                cmap_sequential='pink_r', f_lim=None, GT_baz=None,

                                ############### Power Spectral Density ##############################################
                                compute_metrics=False, trim_family_window=None, window='hann', nperseg=2**8, noverlap_percent=50, 
                                noise_start=10, db_units=True, PSD_xlim=None, PSD_ylim=None, PSD_title=None,

                                ############### Plotting params #################################################
                                title=None, figsize=(9,6), fname_plot=None, fname_plot_PSD=None):
    '''----------------------------------------------------------------------------------------------------------------------------------
    Plots the results of sliding-window array processing that span multiple frequency bands

    Input:
        st: unfiltered ObsPy stream object
        f_bands (Pandas DataFrame): contains frequency bands and time windows constructed using the Segmentor
        T (array): times of length NT (the number of time windows with array processing results)
        B (array): backazimuths of length [NB, NT] where NB is the number of frequency bands
        V (array): phase/trace velocities of length [NB, NT]
        S (array):semblances of length [NB, NT]
        normalize (boolean): whether to normalize time-series recording
        beamform_data (booelan): whether to beamform the data and plot
        bandpass (list): frequency range to use to bandpass time-series data (defaults to [0.5,5])
        taper (str): type of taper to use - defaults to "cosine"
        max_percentage (float/int): percentage to use for taper
        max_length (float/int): length to use for taper
        amp_units (str): units used to measure amplitude (e.g., Pressure [Pa] - make sure units are in brackets)
        legend_loc (str): relative location of legend in time-series plot
        t_lim (list): range in time window start and end times to plot (in seconds relative to start time)
        amp_lim (list): range in amplitude for visualization in time-series plot (defaults to [data.min(), data.max()])
        delay_times (array or list): tropospheric, stratospheric, and thermospheric arrival times (in seconds relative to stream start time)
        element (str): name of the element to plot the time-series data for (defaults to first element if not provided)
        return_beam (boolean): if True, returns modified ObsPy stream where the last trace contains the beamformed data
        ix (array): Indices of frequencies, times (output from make_families)
        pixels_in_families (array): NumPy array of all unique pixel ID's that are in families (output from make_families)
        families (list): each entry contains unique pixel ID's in that family (output from make_families - used for beamforming and specifying windows for PSD)
        families_table (Pandas DataFrame): dataframe of families sorted by time (output from df_families - used for beamforming and specifying windows for PSD)
        family_idx (int): index of family detection to be used for beamforming and/or PSD construction (if None, family with max semblance will be used)
        semblance_threshold (float): threshold value where anything below will be filtered out - not used if Aggregator parameters are specified
        clim_baz (list): range in back azimuth values to plot
        clim_vtr (list): range in trace velocity values to plot
        log_freq (boolean): whether to plot frequency in log-scale (defaults to True)
        cmap_cyclic (str): colormap to be used for cyclical value ranges
        cmap_sequential (str): colormap to be used for seqeuntial value ranges
        f_lim (list): range in frequencies to plot
        GT_baz (float/int): ground truth back azimuth - specified only if you want to plot deviation (clim_baz turns into plus/minus deviation)
        compute_metrics (boolean): whether to produce PSD plot and calculate average SNR, beam SNR, absolute peak beam amplitude, and peak-to-peak amplitude (results will be displayed in title)
        trim_family_window (list): both inputs decrease family window size relative to the start (similar to t_lim)
        window (str): desired window to use for PSD (defaults to hann)
        nperseg (int): length of each segment used for PSD (defaults to 256 - same as signal.welch())
        noverlap_percent (int): percent overlap between segments (defaults to 50% - same as signal.welch())
        noise_start (int/float): time in minutes prior to start time of signal window, used for noise PSD
        db_units (boolean): whether to plot the PSD in decibels (common practice for infrasound data - defaults to true)
        PSD_xlim (list): frequency limits on PSD plot (defaults to [minimum frequency in f_bands, Nyquist frequency])
        PSD_ylim (list): power spectral limits on PSD plot
        PSD_title (str): title of PSD plot (defaults to "Power Spectral Density")
        title (str): title of array processing plot
        figsize (tuple): specifies size of figure
        fname_plot (str): filename to save array processing plot
        fname_plot_PSD (str): filename to save PSD plot

    Output:
        st_beam: ObsPy stream object where first element is beamformed trace (only returns if return_beam = True)
    ----------------------------------------------------------------------------------------------------------------------------------'''
    # For plotting families
    S_filt = S.copy()
    if (pixels_in_families is not None) and (ix is not None):
        # Set all semblances to zero where the pixel is not in a family:
        x = np.zeros(S.shape)
        x[ix[0][pixels_in_families],ix[1][pixels_in_families]] = 1   # Makes a mask where 1 means plot value
        S_filt[x == 0] = 0
        ix = np.where(S_filt < 1e-6)
    elif (pixels_in_families == None) and (ix == None):
        ix = np.where(S_filt < semblance_threshold)
    #-----------------------------------------------------------------------------------------------------------------#
    # Plotting time-series
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(4,1,1)
    # Filter
    if bandpass is not None:
        st_taper = st.copy()
        st_taper.taper(type=taper, max_percentage=max_percentage, max_length=max_length)
        st_filt = st_taper.copy()
        try:
            st_filt.filter(type='bandpass', freqmin=min(bandpass), freqmax=max(bandpass))
        except:
            st_filt = st_taper.copy().split()
            st_filt.filter(type='bandpass', freqmin=min(bandpass), freqmax=max(bandpass))
            st_filt = st_filt.merge() 
    # Plot single array element as time-series
    if beamform_data == False:
        if element is not  None:
            tr = st_filt.select(station=element)[0].copy()
        else:
            tr = st_filt.select(station=st_filt[0].stats.station)[0].copy()
        t_tr = np.arange(0, tr.stats.npts*tr.stats.delta, tr.stats.delta)
        t, data = fix_lengths(t_tr, tr.data)
        if normalize == True:
            data = data / np.max(np.abs(data))
        ax1.plot(t, data, 'k-', label=tr.stats.station)
    # Plot beam as time-series
    elif beamform_data == True:
        # Use Aggregator output to beamform (families table)
        if (families is not None) and (families_table == None):
            families_table = df_families(st[0].stats.starttime.matplotlib_date, families)
        elif (families == None) and (families_table is not None):
            pass
        elif (families == None) and (families_table == None):
            raise Exception('If beamforming, families or families_table must be specified.')
        # Choose family (either user defined or defaults to family with max semb)
        if family_idx is not None:
            baz = families_table['mean_baz'][family_idx]
            vel = families_table['mean_vel'][family_idx]
        else:
            family_idx = np.where((families_table['max_semb'] == families_table['max_semb'].max()))[0][0] # choose family with max semblance if not specified by user
            baz = families_table['mean_baz'][family_idx]
            vel = families_table['mean_vel'][family_idx]
        t_shifts = get_slowness_vector_time_shifts(st_filt, st_filt[0].stats.station, baz=baz, vel=vel, units='km')
        t_beam, beam_data = beamform(t_shifts, st_filt, st_filt[0].stats.station, normalize_beam=normalize)
        t, data = fix_lengths(t_beam, beam_data)
        ax1.plot(t, data, color='grey', label='Beam')
    if (normalize == True) and (amp_lim == None):
        if len(amp_units.split(' ')) > 1:
            plt.ylabel('Normalized\n '+ amp_units.split(' ')[0])
        else:
            plt.ylabel('Normalized\n '+ amp_units)
        plt.ylim([-1.05,1.05])
    elif (normalize == True) and (amp_lim is not None):
        raise Exception('Y-axis limits are fixed if normalize = True, no need to define amp_lim')
    elif (normalize == False) and (amp_lim is not None):
        plt.ylabel(amp_units)
        plt.ylim(amp_lim)
    elif (normalize == False) and (amp_lim == None):
        plt.ylabel(amp_units)
        plt.ylim([data.min(), data.max()])
    if t_lim is not None:
        plt.xlim(t_lim)
    else:
        plt.xlim([0, t[-1]])
    ax1.tick_params(labelbottom=False)
    if delay_times is not None:
        plt.axvline(x=delay_times[0], lw=1, ls='--', color='red')
        plt.axvline(x=delay_times[1], lw=1, ls='--', color='green')
        plt.axvline(x=delay_times[2], lw=1, ls='--', color='blue')
    if (title is not None) and (bandpass is not None):
        ax1_title_obj = ax1.set_title(title+ " - Bandpass: " + str(bandpass))
    elif (title is not None) and (bandpass == None):
        ax1_title_obj = ax1.set_title(title)
    elif (title == None) and (bandpass is not None):
        ax1_title_obj = ax1.set_title("Bandpass: " + str(bandpass))
    plt.legend(loc=legend_loc)
    #-----------------------------------------------------------------------------------------------------------------#
    # Plotting back azimuth
    ax2 = fig.add_subplot(4,1,2, sharex=ax1)
    if GT_baz is not None:
        B_dev = B.copy()
        B_dev -= GT_baz
        for B_row in range(len(B_dev[:,0])):
            for B_col in range(len(B_dev[0,:])):
                if B_dev[B_row, B_col] < -270: 
                    B_dev[B_row, B_col] += 360
                elif B_dev[B_row, B_col] > 270:
                    B_dev[B_row, B_col] -= 360
                else:
                    pass
        B_plt = B_dev.copy()
        if clim_baz is not None:
            ix_low = np.where(B_plt < clim_baz[0])
            ix_high = np.where(B_plt > clim_baz[1])
            B_plt[ix_low] = None; B_plt[ix_high] = None
        else:
            pass
    else:
        B_plt = B.copy()
    B_plt[ix] = None
    t_plot = np.hstack((T,T[len(T)-1]+np.diff(T)[0]))
    f_plot = np.hstack((f_bands['fmin'].values, f_bands['fmax'].values[len(f_bands['fmax'])-1]))
    pcm1 = plt.pcolor(t_plot, f_plot, B_plt, cmap=plt.get_cmap(cmap_cyclic), shading='flat')
    if (clim_baz is not None) and (clim_baz[0]<clim_baz[1]):
        plt.clim([clim_baz[0], clim_baz[1]])
    elif (clim_baz is not None) and (clim_baz[0]>clim_baz[1]):
        plt.clim([clim_baz[0], clim_baz[1]+360])
    plt.ylabel('Freq. [Hz]')
    if log_freq == True:
        plt.yscale('log')
    if f_lim is not None:
        plt.ylim(f_lim)
    ax2.tick_params(labelbottom=False)
    #-----------------------------------------------------------------------------------------------------------------#
    # Plotting trace velocity
    ax3 = fig.add_subplot(4,1,3, sharex=ax1, sharey=ax2)
    V_plt = V.copy()
    V_plt[ix] = None
    if GT_baz is not None:
        V_plt[ix_low] = None; V_plt[ix_high] = None
    pcm2 = plt.pcolor(t_plot, f_plot, V_plt, cmap=plt.get_cmap(cmap_sequential), shading='flat')
    if clim_vtr is not None:
        plt.clim([clim_vtr[0], clim_vtr[1]])
    plt.ylabel('Freq. [Hz]')
    ax3.tick_params(labelbottom=False)
    #-----------------------------------------------------------------------------------------------------------------#
    # Plotting semblance
    ax4 = fig.add_subplot(4,1,4, sharex=ax1, sharey=ax2)
    start_time_string = str(st_filt[0].stats.starttime).split('.')[0].replace('T',' ')
    pcm3 = plt.pcolor(t_plot, f_plot, S, cmap=plt.get_cmap(cmap_sequential), shading='flat')
    plt.clim([0,1])
    plt.ylabel('Freq. [Hz]')
    plt.xlabel('Time [s] after ' + start_time_string)
    #-----------------------------------------------------------------------------------------------------------------#
    # Manually adding colorbars - colorbar for back azimuth
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.8575, 0.5175, 0.0200, 0.1550])
    fig.colorbar(pcm1, cax=cbar_ax)
    if GT_baz is not None:
        cbar_ax.set_ylabel('Azimuth\nDeviation [\N{DEGREE SIGN}]')
    else:
        cbar_ax.set_ylabel('Azimuth [\N{DEGREE SIGN}]')
    cbar_ax.locator_params(nbins=6)
    if clim_baz is not None:
        if clim_baz[0]>clim_baz[1]:
            tick_labels= np.array(cbar_ax.get_yticks())
            tick_labels= tick_labels.astype(int)
            new_labels=(np.where((tick_labels > 360), tick_labels-360, tick_labels))
            cbar_ax.set_yticklabels(new_labels)
    # colorbar for trace velocity
    cbar_ax = fig.add_axes([0.8575, 0.3175, 0.0200, 0.1550])
    fig.colorbar(pcm2, cax=cbar_ax)
    cbar_ax.locator_params(nbins=4)
    cbar_ax.set_ylabel('Velocity [km/s]')
    # colorbar for trace velocity
    cbar_ax = fig.add_axes([0.8575, 0.1175, 0.0200, 0.1550])
    fig.colorbar(pcm3, cax=cbar_ax)
    cbar_ax.locator_params(nbins=4)
    cbar_ax.set_ylabel('Semblance')
    #-----------------------------------------------------------------------------------------------------------------#
    # Optionally plot PSD and compute avg SNR, beam SNR, absolute peak amplitude, and peak to peak amplitude
    if amp_units == 'Amplitude': # make sure amplitude units are input properly for plotting
        units_str = amp_units
    else:
        try:
            units_str = re.findall(r'\[(.*?)\]', amp_units)[0]
        except Exception as inst:
            print(inst)
            print('Make sure to include amplitude units in brackets (e.g., Pressure [Pa])')
    if compute_metrics == True:
        # Define signal window using Aggregator output (families table)
        if beamform_data == True:
            plot_beam_metrics = True
            pass
        else: # if data was not beamformed, we don't have families table just yet
            if (families is not None) and (families_table == None):
                families_table = df_families(st[0].stats.starttime.matplotlib_date, families)
            elif (families == None) and (families_table is not None):
                pass
            elif (families == None) and (families_table == None):
                raise Exception('If constructing PSD, families or families_table must be specified.')
            plot_beam_metrics = False
        # Choose family (either user defined or defaults to family with max semb)
        if family_idx is not None:
            fam_start = families_table['start_time'][family_idx]
            fam_end = families_table['end_time'][family_idx]
        else:
            family_idx = np.where((families_table['max_semb'] == families_table['max_semb'].max()))[0][0] # choose family with max semblance if not specified by user
            fam_start = families_table['start_time'][family_idx]
            fam_end = families_table['end_time'][family_idx]
        if trim_family_window is not None:
            sig_start = fam_start + trim_family_window[0]
            sig_end = fam_start + trim_family_window[1]
        else:
            sig_start = fam_start.copy()
            sig_end = fam_end.copy()
        #-----------------------------------------------------------------------------------------------------------------#
        # Shade in signal and noise windows on time-series subplot
        noise_start = 60*(noise_start)
        y_fill = np.linspace(data.min(), data.max(), 1000)
        ax1.fill_betweenx(y_fill, x1=sig_start, x2=sig_end, alpha=0.25,  color='red') # signal
        ax1.fill_betweenx(y_fill, x1=sig_start - noise_start, x2=sig_end - noise_start, alpha=0.25,  color='grey') # noise
        #-----------------------------------------------------------------------------------------------------------------#
        # Compute PSD
        Pxx_N = []; Pxx_S = []; f_N = []; f_S = []
        for tr in st:
            try:
                f_N_tmp, Pxx_N_tmp = power_spectrum(tr, t_start=sig_start - noise_start, t_end=sig_end - noise_start, window=window, nperseg=nperseg, noverlap_percent=noverlap_percent) # compute noise PSD
            except Exception as inst:
                print(inst)
                print('Could not compute noise power spectrum, make sure window start time is within the data stream and window size is not too small, otherwise reduce noverlap_percent.')
            f_S_tmp, Pxx_S_tmp = power_spectrum(tr, t_start=sig_start, t_end=sig_end, window=window, nperseg=nperseg, noverlap_percent=noverlap_percent) # compute signal PSD
            Pxx_N.append(Pxx_N_tmp[0,:]); Pxx_S.append(Pxx_S_tmp[0,:]); f_N.append(f_N_tmp); f_S.append(f_S_tmp)
        Pxx_N = np.array(Pxx_N); Pxx_S = np.array(Pxx_S); f_N = np.array(f_N); f_S = np.array(f_S)
        #-----------------------------------------------------------------------------------------------------------------#
        # Convert to decibels
        if db_units == True:
            Pxx_S = 10*np.log10(Pxx_S)
            Pxx_N = 10*np.log10(Pxx_N)
        #-----------------------------------------------------------------------------------------------------------------#
        # Plot average PSD and standard deviation
        fig_PSD = plt.figure(figsize=(figsize))
        sig_avg = np.mean(Pxx_S, axis=0, keepdims=True)[0,:]; sig_std = np.std(Pxx_S, axis=0, keepdims=True)[0,:]
        noise_avg = np.mean(Pxx_N, axis=0, keepdims=True)[0,:]; noise_std = np.std(Pxx_N, axis=0, keepdims=True)[0,:]
        plt.semilogx(f_S[0], sig_avg, 'red', linewidth=0.75, linestyle='-', label='Signal') # average signal
        plt.semilogx(f_N[0], noise_avg, 'black', linewidth=0.75, linestyle='-', label='Noise') # average noise
        plt.fill_between(f_S[0], sig_avg - sig_std, sig_avg + sig_std, alpha=0.2, color='red')
        plt.fill_between(f_N[0], noise_avg - noise_std, noise_avg + noise_std, alpha=0.2, color='black')
        # Find peak frequency (where signal minus noise is at a maximum)
        psd_difference = sig_avg - noise_avg; max_idx = np.argmax(psd_difference)
        if float(f_S[0][max_idx]) == float(0): # just in case there's not much separation between signal and noise energy (can end up choosing peak freq at 0 Hz)
            second_highest = np.max(psd_difference[psd_difference != np.max(psd_difference)])
            max_idx = np.where((second_highest == psd_difference))[0][0]
        plt.axvline(f_S[0][max_idx], color='red', ls='--', lw=1, label='Peak Freq')
        if PSD_xlim is not None:
            plt.xlim([PSD_xlim[0], PSD_xlim[1]])
        else:
            plt.xlim([f_bands['fmin'][0], st[0].stats.sampling_rate/2])
        if PSD_ylim is not None:
            plt.ylim([PSD_ylim[0], PSD_ylim[1]])
        plt.xlabel('Frequency [Hz]')
        if db_units == True:
            plt.ylabel('dB Rel. 1 ['+units_str+'\u00b2/Hz]')
        else:
            plt.ylabel('Power Spectrum ['+units_str+'\u00b2/Hz]')
        plt.grid(linewidth=0.25, which='both')
        plt.legend(loc='upper right')
        if PSD_title is not None:
            plt.title(PSD_title + '\nPeak Freq: %.3f'%f_S[0][max_idx] + ' [Hz]')
        else:
            plt.title('Power Spectral Density' + '\nPeak Freq: %.3f'%f_S[0][max_idx] + ' [Hz]')
        #-----------------------------------------------------------------------------------------------------------------#
        # Compute avg SNR - absolute peak amplitude - peak-to-peak amplitude
        t_snr = np.arange(0, st_filt[0].stats.npts*st_filt[0].stats.delta, st_filt[0].stats.delta)
        _, sig_data = data_time_window(t_snr, st_filt, t_start=sig_start, t_end=sig_end)
        _, noise_data = data_time_window(t_snr, st_filt, t_start=sig_start - noise_start, t_end=sig_end - noise_start)
        avg_snr = 0; abs_peak_amp = 0; peak_to_peak = 0
        for tr_idx in range(len(st_filt)):
            snr_tmp = get_SNR(sig_data[tr_idx,:], noise_data[tr_idx,:])
            abs_peak_amp_tmp = np.max(np.abs(sig_data[tr_idx,:]))
            peak_to_peak_tmp = sig_data[tr_idx,:].max() - sig_data[tr_idx,:].min()
            avg_snr += snr_tmp; abs_peak_amp += abs_peak_amp_tmp; peak_to_peak += peak_to_peak_tmp
        avg_snr /= len(st_filt); abs_peak_amp /= len(st_filt); peak_to_peak /= len(st_filt)
        #-----------------------------------------------------------------------------------------------------------------#
        # Append values to title of array processing figure
        ax1_current_title = ax1_title_obj.get_text()
        if plot_beam_metrics == True: # if beamformed data, compute beam snr and use beam for peak amplitude measurements
            # Compute beam SNR
            if family_idx is not None:
                baz = families_table['mean_baz'][family_idx]
                vel = families_table['mean_vel'][family_idx]
            else:
                family_idx = np.where((families_table['max_semb'] == families_table['max_semb'].max()))[0][0] # choose family with max semblance if not specified by user
                baz = families_table['mean_baz'][family_idx]
                vel = families_table['mean_vel'][family_idx]
            t_shifts = get_slowness_vector_time_shifts(st, st[0].stats.station, baz=baz, vel=vel, units='km')
            t_beam, beam = beamform(t_shifts, st_filt, st_filt[0].stats.station); st_beam = add_beam_to_stream(st_filt, beam)
            _, beam_sig_data = data_time_window(t_beam, st_beam[-1], t_start=sig_start, t_end=sig_end)
            _, beam_noise_data = data_time_window(t_beam, st_beam[-1], t_start=sig_start - noise_start, t_end=sig_end - noise_start)
            beam_snr = get_SNR(beam_sig_data, beam_noise_data)
            # Compute absolute peak amplitude and peak-to-peak amplitude
            abs_peak_amp = np.max(np.abs(beam_sig_data)) / len(st_filt)
            peak_to_peak = (beam_sig_data.max() - beam_sig_data.min()) / len(st_filt)
            if amp_units == 'Amplitude':
                ax1_new_title = ax1_current_title + '\nAvg SNR: %.3f'%avg_snr + ' - Beam SNR: %.3f'%beam_snr + ' - Peak Amp: %.3f'%abs_peak_amp + ' - P2P Amp: %.3f'%peak_to_peak
            else:
                ax1_new_title = ax1_current_title + '\nAvg SNR: %.3f'%avg_snr + ' - Beam SNR: %.3f'%beam_snr + ' - Peak Amp: %.3f'%abs_peak_amp + ' [' + units_str + '] - P2P Amp: %.3f'%peak_to_peak + ' [' + units_str + ']'
        elif plot_beam_metrics == False:
            if amp_units == 'Amplitude':
                ax1_new_title = ax1_current_title + '\nAvg SNR: %.3f'%avg_snr + ' - Peak Amp: %.3f'%abs_peak_amp + ' - P2P Amp: %.3f'%peak_to_peak
            else:
                ax1_new_title = ax1_current_title + '\nAvg SNR: %.3f'%avg_snr + ' - Peak Amp: %.3f'%abs_peak_amp + ' [' + units_str + '] - P2P Amp: %.3f'%peak_to_peak + ' [' + units_str + ']'
        ax1_title_obj.set_text(ax1_new_title)
    #-----------------------------------------------------------------------------------------------------------------#
    if fname_plot is not None:
        fig.savefig(fname_plot)
    if fname_plot_PSD is not None:
        fig_PSD.savefig(fname_plot_PSD)
    if (return_beam == True) and (beamform_data == True):
        return add_beam_to_stream(st, beamform(t_shifts, st, st[0].stats.station)[1]) # return unfiltered beam for further analysis
    elif (return_beam == True) and (beamform_data == False):
        raise Exception('beamform_data must be set to True if return_beam is True')

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def polar_plot_families(families, r_axis='velocity'):
    '''----------------------------------------------------------------------------------------------------------------------------------
    Produces a summary polar plot of all families

    Input:
        families (list): each entry contains unique pixel ID's in that family (output from make_families)
        r_axis (str): the parameter to plot as a function of radius
    ----------------------------------------------------------------------------------------------------------------------------------'''
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

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def plot_spectrogram(st, element=None,
                    ############### Time-series #################################################
                    bandpass=[0.5,5], taper='cosine', max_percentage=0.05, max_length=60,
                    delay_times=None, normalize=False, amp_units='Amplitude', legend_loc='upper right', 
                    t_lim=None, amp_lim=None,

                    ############### Spectrogram ################################################
                    nperseg=2**8, noverlap_percent=50, f_lim=None, v_lim=None, log_scale=False, colormap='viridis_r', 
                    shading='gouraud', log_normalize=True, colorbar_location='bottom', colorbar_pad=0.25, plot_UTC=False, 
                    UTC_time_interval=None,

                    ############### Plotting params #################################################
                    title=None, figsize=(12,6), fname_plot=None):
    '''----------------------------------------------------------------------------------------------------------------------------------
    Computes spectrogram with consecutive Fourier transforms using Scipy's signal.spectrogram() function

    Input:
        st: ObsPy stream object
        element (str): array element name used to compute spectrogram for (defaults to first element in array)
        bandpass (list): frequency range to use to bandpass time-series data (defaults to [0.5,5])
        taper (str): type of taper to use - defaults to "cosine"
        max_percentage (float/int): percentage to use for taper
        max_length (float/int): length to use for taper
        delay_times (list): theoretical tropospheric, stratospheric, and thermospheric arrival times (in seconds relative to stream start time)
        normalize (boolean): whether to normalize time-series recording
        amp_units (str): units used to measure amplitude (e.g., Pressure (Pa) - make sure units are in parentheses)
        legend_loc (str): relative location of legend in time-series plot
        t_lim (list): range in time window start and end times to plot (in seconds relative to start time)
        amp_lim (list): range in amplitude for visualization in time-series plot (defaults to [data.min(), data.max()])
        nperseg (int): number of sample points in each segment of time window used for consecutive Fourier transforms (defaults to 256 - same as signal.spectrogram())
        noverlap_percent (int): percent overlap between segments (defaults to 50%)
        f_lim (list): range in frequencies to plot (defaults to [f.min(), Nyquist])
        v_lim (list): range in spectrogram values to plot (defaults to minimum and maximum power spectrum values)
        log_scale (boolean): whether to plot the frequency in log-scale for spectrogram
        colormap (str): colormap used to visualize spectrogram
        shading (str): shading to be used for spectrogram plot
        log_normalize (boolean): whether to log normalize the spectrogram for plotting (defaults to True)
        colorbar_location (str): location of colorbar relative to spectrogram plot
        colorbar_pad (float/int): how far away colorbar should be from plot
        plot_UTC (boolean): whether to plot time in UTC
        UTC_time_interval (int): time interval (in minutes) to be used for x-axis UTC formatting
        title (str): title of plot
        figsize (tuple): specifies size of figure
        fname_plot (str): filename to save spectrogram plot
    
    Note: Default configuration for higher frequencies at linear scale.
    If plotting in log scale, we suggest increasing nperseg (or use the plot_scalogram() function for the best resolution at broadband scales)
    ----------------------------------------------------------------------------------------------------------------------------------'''
    # Plot time-series
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(2,1,1)
    # Make sure sampling rate is consistent across all array elements
    try:
        st = st.merge()
    except:
        st = st.resample(int(st[0].stats.sampling_rate))
        st = st.merge()
    # Filter
    if bandpass is not None:
        st_taper = st.copy()
        st_taper.taper(type=taper, max_percentage=max_percentage, max_length=max_length)
        st_filt = st_taper.copy()
        try:
            st_filt.filter(type='bandpass', freqmin=min(bandpass), freqmax=max(bandpass))
        except:
            st_filt = st_taper.copy().split()
            st_filt.filter(type='bandpass', freqmin=min(bandpass), freqmax=max(bandpass))
            st_filt = st_filt.merge() 
    if element is not None:
        tr = st_filt.select(station=element)[0]
    else:
        tr = st_filt[0].copy()
    # Construct time vector
    t = np.arange(0, tr.stats.npts*tr.stats.delta, tr.stats.delta)
    t, data = fix_lengths(t, tr.data)
    if normalize == True:
        data = data/np.max(np.abs(data))
    if element == 'Beam':
        ax1.plot(t, data, 'grey', label=tr.stats.station)
    else:
        ax1.plot(t, data, 'k-', label=tr.stats.station)
    if (normalize == True) and (amp_lim == None):
        if len(amp_units.split(' ')) > 1:
            plt.ylabel('Normalized\n '+ amp_units.split(' ')[0])
        else:
            plt.ylabel('Normalized\n '+ amp_units)
        plt.ylim([-1.05,1.05])
    elif (normalize == True) and (amp_lim is not None):
        raise Exception('Y-axis limits are fixed if normalize = True, no need to define amp_lim')
    elif (normalize == False) and (amp_lim is not None):
        plt.ylabel(amp_units)
        plt.ylim(amp_lim)
    elif (normalize == False) and (amp_lim == None):
        plt.ylabel(amp_units)
        plt.ylim([data.min(), data.max()])
    global_start = min(tr.stats.starttime for tr in st)
    global_end = max(tr.stats.endtime for tr in st)
    if (plot_UTC == True) and (t_lim == None):
        # Plot UTC - round down to the nearest minute_interval for start
        if UTC_time_interval is not None:
            minute_interval = UTC_time_interval
        else:
            if global_end-global_start <= 3600*2:
                minute_interval = 15 # less than 2 hours make it 15 minute intervals
            else:
                # round to the nearest 15 minutes and partition data stream into 6 segements if stream is longer than 2 hours
                n = global_end-global_start
                sec_round = round(n / 900) * 900
                min_round = sec_round // 60
                minute_interval = min_round // 6
        rounded_start_tick = global_start.replace(minute=(global_start.minute // minute_interval) * minute_interval, second=0, microsecond=0)
        if rounded_start_tick < global_start:
            rounded_start_tick += (minute_interval*60)
        x_tick_positions = np.arange(rounded_start_tick - global_start, global_end - global_start, minute_interval*60)
        x_tick_labels = [(global_start + t).strftime('%H:%M:%S') for t in x_tick_positions]
        ax1.set_xticks(x_tick_positions); ax1.set_xticklabels(x_tick_labels)
        ax1.set_xlim([0,(global_end-global_start)]); ax1.set_xlabel(global_start.strftime('%Y-%m-%d')+' Time [UTC]')
    elif (plot_UTC == False) and (t_lim is not None):
        ax1.set_xlim(t_lim)
    elif (plot_UTC == False) and (t_lim == None):
        ax1.set_xlim([t[0],t[-1]])
    elif (plot_UTC == True) and (t_lim is not None):
        raise Exception("Can't have t_lim and plot_UTC set to True, since t_lim is relative.")
    if delay_times is not None:
        plt.axvline(x=delay_times[0], lw=1, ls='--', color='red')
        plt.axvline(x=delay_times[1], lw=1, ls='--', color='green')
        plt.axvline(x=delay_times[2], lw=1, ls='--', color='blue')
    if (title is not None) and (bandpass is not None):
        ax1.set_title(title+ ' Spectrogram - Bandpass: ' + str(bandpass))
    elif (title is not None) and (bandpass == None):
        ax1.set_title(title + ' Spectrogram')
    elif (title == None) and (bandpass is not None):
        ax1.set_title('Spectrogram Bandpass: ' + str(bandpass))
    ax1.tick_params(labelbottom=False)
    plt.legend(loc=legend_loc)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Compute spectrogram - use unfiltered trace
    if element is not None:
        tr_spec = st.select(station=element)[0]
    else:
        tr_spec = st[0].copy()
    noverlap = (nperseg) * (noverlap_percent/100)
    f, t_f, Sxx = signal.spectrogram(tr_spec.data, tr_spec.stats.sampling_rate, nperseg=nperseg, noverlap=noverlap)
    if log_normalize == True:
        Sxx = np.log10(Sxx + 1e-10) # log normalize (small constant added to avoid log10(0))
    # Plot spectrogram
    ax2 = plt.subplot(2,1,2, sharex=ax1)
    cmap = cm.get_cmap(colormap, 256)
    cmap = cmap(np.linspace(2,0.001,100))
    cmap = ListedColormap(cmap)
    if f_lim is not None:
        ix = np.where((f >= f_lim[0]) & (f <= f_lim[1]))
    else:
        ix = np.where((f >= float(f.min()+1e-10)) & (f <= float(f.max())))
    if v_lim is not None:
        vmin = v_lim[0]; vmax = v_lim[1]
    else:
        vmin = Sxx[ix].min(); vmax = Sxx[ix].max()
    plt.pcolormesh(t_f, f[ix], Sxx[ix], cmap=cmap, shading=shading, vmin=vmin, vmax=vmax)
    colorbar = plt.colorbar(location=colorbar_location, pad=colorbar_pad)
    if amp_units == 'Amplitude': # make sure amplitude units are input properly for plotting
        units_str = amp_units
    else:
        try:
            units_str = re.findall(r'\[(.*?)\]', amp_units)[0]
        except Exception as inst:
            print(inst)
            print('Make sure to include amplitude units in brackets (e.g., Pressure [Pa])')
    if log_normalize == True:
        colorbar.set_label('Log10['+units_str+'\u00b2/Hz]', fontsize=10)
    else:
        colorbar.set_label('['+units_str+'\u00b2/Hz]', fontsize=10)
    if log_scale == True:
        ax2.set_yscale('log')
    if (plot_UTC == True) and (t_lim == None):
        ax2.set_xlabel(global_start.strftime('%Y-%m-%d')+' Time [UTC]')
    elif (plot_UTC == False) and (t_lim is not None):
        ax2.set_xlabel('Time [s] after ' + str(global_start).split('.')[0])
    elif (plot_UTC == False) and (t_lim == None):
        ax2.set_xlabel('Time [s] after ' + str(global_start).split('.')[0])
    elif (plot_UTC == True) and (t_lim is not None):
        raise Exception("Can't have t_lim and plot_UTC set to True, since t_lim is relative.")
    ax2.set_ylabel('Frequency [Hz]')
    # Save figure
    if fname_plot is not None:
        fig.savefig(fname_plot)

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def plot_scalogram(st, element=None, 
                    ############### Time-series #################################################
                    bandpass=[0.5,5], trim_stream=None, taper='cosine', max_percentage=0.05, max_length=60,
                    delay_times=None, normalize=False, amp_units='Amplitude', legend_loc='upper right', 
                    t_lim=None, amp_lim=None,

                    ############### Scalogram ################################################
                    scale_range=[1,5500], scale_res=200, wavelet='cmor2.25-2.75', log_normalize=True,
                    f_lim=None, v_lim=None, colormap='seismic_r', shading='gouraud', log_scale=True, 
                    colorbar_location='bottom', colorbar_pad=0.25, plot_UTC=False, UTC_time_interval=None,
                    use_filtered_data=False,

                    ############### Plotting params #################################################
                    title=None, figsize=(12,6), fname_plot=None):
    '''----------------------------------------------------------------------------------------------------------------------------------
    Computes scalogram with continuous wavelet transforms using PyWavelet's cwt() and plots associated power spectrum

    Input:
        st: ObsPy stream object
        element (str): array element name used to compute spectrogram for (defaults to first element in array)
        bandpass (list): frequency range to use to bandpass time-series data (defaults to [0.5,5])
        trim_stream (list): both inputs trim stream from start time (we suggest trimming to make cwt faster)
        taper (str): type of taper to use - defaults to "cosine"
        max_percentage (float/int): percentage to use for taper
        max_length (float/int): length to use for taper
        delay_times (list): theoretical tropospheric, stratospheric, and thermospheric arrival times (in seconds relative to stream start time)
        normalize (boolean): whether to normalize time-series recording
        amp_units (str): units used to measure amplitude (e.g., Pressure (Pa) - make sure units are in parentheses)
        legend_loc (str): relative location of legend in time-series plot
        t_lim (list): range in time window start and end times to plot (in seconds relative to start time)
        amp_lim (list): range in amplitude for visualization in time-series plot (defaults to [data.min(), data.max()])
        scale_range (list): range in wavelet scales (default configuration constructs floor at 0.01 Hz for 20 Hz sampling rate)
        scale_res (int): scale resolution for wavelet
        wavelet (str): wavelet used for cwt (defaults to complex morlet with bandwidth of 2.25 and center frequency of 2.75)
        log_normalize (boolean): whether to log normalize the scalogram power spectrum for better visualization (defaults to True)
        f_lim (list): frequency range for scalogram (defaults to [f.min(), Nyquist])
        v_lim (list): range in scalogram values to plot (defaults to minimum and maximum power spectrum values)
        colormap (str): colormap used to visualize scalogram
        shading (str): shading to be used for scalogram plot
        log_scale (boolean): whether to plot the frequency in log-scale for spectrogram (defaults to True)
        colorbar_location (str): location of colorbar relative to spectrogram plot
        colorbar_pad (float/int): how far away colorbar should be from plot
        plot_UTC (boolean): whether to plot time in UTC
        UTC_time_interval (int): time interval (in minutes) to be used for x-axis UTC formatting
        use_filtered_data (boolean): whether to use filtered data for scalogram (if normalize=True, will use normalized filtered data)
        title (str): title of plot
        figsize (tuple): specifies size of figure
        fname_plot (str): filename to save spectrogram plot
    
    Note: For a more detailed explanation on wavelets, please visit PyWavelets Documentation (https://pywavelets.readthedocs.io/en/latest/ref/cwt.html)
    Increasing scale range drops frequency floor in plot, while having a higher sampling rate raises frequency floor
    ----------------------------------------------------------------------------------------------------------------------------------'''
    # Plot time-series
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(2,1,1)
    # Make sure sampling rate is consistent across all array elements
    try:
        st = st.merge()
    except:
        st = st.resample(int(st[0].stats.sampling_rate))
        st = st.merge()
    # Trimming stream to make cwt faster
    st_trim = st.copy()
    if trim_stream is not None:
        dt = st_trim[0].stats.starttime
        st_trim.trim(dt+trim_stream[0], dt+trim_stream[1])
    # Filter
    if bandpass is not None:
        st_taper = st_trim.copy()
        st_taper.taper(type=taper, max_percentage=max_percentage, max_length=max_length)
        st_filt = st_taper.copy()
        try:
            st_filt.filter(type='bandpass', freqmin=min(bandpass), freqmax=max(bandpass))
        except:
            st_filt = st_taper.copy().split()
            st_filt.filter(type='bandpass', freqmin=min(bandpass), freqmax=max(bandpass))
            st_filt = st_filt.merge() 
    if element is not None:
        tr = st_filt.select(station=element)[0]
    else:
        tr = st_filt[0].copy()
    # Construct time vector
    t = np.arange(0, tr.stats.npts*tr.stats.delta, tr.stats.delta)
    t, data = fix_lengths(t, tr.data)
    if normalize == True:
        data = data/np.max(np.abs(data))
    if element == 'Beam':
        ax1.plot(t, data, 'grey', label=tr.stats.station)
    else:
        ax1.plot(t, data, 'k-', label=tr.stats.station)
    if (normalize == True) and (amp_lim == None):
        if len(amp_units.split(' ')) > 1:
            plt.ylabel('Normalized\n '+ amp_units.split(' ')[0])
        else:
            plt.ylabel('Normalized\n '+ amp_units)
        plt.ylim([-1.05,1.05])
    elif (normalize == True) and (amp_lim is not None):
        raise Exception('Y-axis limits are fixed if normalize = True, no need to define amp_lim')
    elif (normalize == False) and (amp_lim is not None):
        plt.ylabel(amp_units)
        plt.ylim(amp_lim)
    elif (normalize == False) and (amp_lim == None):
        plt.ylabel(amp_units)
        plt.ylim([data.min(), data.max()])
    global_start = min(tr.stats.starttime for tr in st)
    global_end = max(tr.stats.endtime for tr in st)
    if (plot_UTC == True) and (t_lim == None):
        # Plot UTC - round down to the nearest minute_interval for start
        if UTC_time_interval is not None:
            minute_interval = UTC_time_interval
        else:
            if global_end-global_start <= 3600*2:
                minute_interval = 15 # less than 2 hours make it 15 minute intervals
            else:
                # round to the nearest 15 minutes and partition data stream into 6 segements if stream is longer than 2 hours
                n = global_end-global_start
                sec_round = round(n / 900) * 900
                min_round = sec_round // 60
                minute_interval = min_round // 6
        rounded_start_tick = global_start.replace(minute=(global_start.minute // minute_interval) * minute_interval, second=0, microsecond=0)
        if rounded_start_tick < global_start:
            rounded_start_tick += (minute_interval*60)
        x_tick_positions = np.arange(rounded_start_tick - global_start, global_end - global_start, minute_interval*60)
        x_tick_labels = [(global_start + t).strftime('%H:%M:%S') for t in x_tick_positions]
        ax1.set_xticks(x_tick_positions); ax1.set_xticklabels(x_tick_labels)
        ax1.set_xlim([0,(global_end-global_start)]); ax1.set_xlabel(global_start.strftime('%Y-%m-%d')+' Time [UTC]')
    elif (plot_UTC == False) and (t_lim is not None):
        ax1.set_xlim(t_lim)
    elif (plot_UTC == False) and (t_lim == None):
        ax1.set_xlim([t[0],t[-1]])
    elif (plot_UTC == True) and (t_lim is not None):
        raise Exception("Can't have t_lim and plot_UTC set to True, since t_lim is relative.")
    if (delay_times is not None) & (trim_stream == None):
        plt.axvline(x=delay_times[0], lw=1, ls='--', color='red')
        plt.axvline(x=delay_times[1], lw=1, ls='--', color='green')
        plt.axvline(x=delay_times[2], lw=1, ls='--', color='blue')
    elif (delay_times is not None) & (trim_stream is not None):
        plt.axvline(x=delay_times[0] - trim_stream[0], lw=1, ls='--', color='red')
        plt.axvline(x=delay_times[1] - trim_stream[0], lw=1, ls='--', color='green')
        plt.axvline(x=delay_times[2] - trim_stream[0], lw=1, ls='--', color='blue')
    if (title is not None) and (bandpass is not None):
        ax1.set_title(title+ ' Scalogram - Bandpass: ' + str(bandpass))
    elif (title is not None) and (bandpass == None):
        ax1.set_title(title + ' Scalogram')
    elif (title == None) and (bandpass is not None):
        ax1.set_title('Scalogram Bandpass: ' + str(bandpass))
    ax1.tick_params(labelbottom=False)
    plt.legend(loc=legend_loc)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Using tapered unfiltered trace
    if element is not None:
        tr = st_taper.select(station=element)[0]
    else:
        tr = st_taper[0].copy()
    # Compute scalogram
    scale = np.geomspace(scale_range[0], scale_range[1], scale_res) # using a log scale for cwt (good for broader frequencies associated with seismic and infrasound signals)
    if (bandpass is not None) & (wavelet == 'cmor'): # using bandpass for bandwidth and center frequency of complex morlet (this is only if user specifies wavelet = "cmor")
        center_frequency = (max(bandpass) + min(bandpass)) / 2 # avg of bandpass range
        bandwidth = (max(bandpass) - min(bandpass)) / 2 # symmetrical bandwidth
        wavelet = wavelet + str(float(bandwidth)) + '-' + str(float(center_frequency))
    if use_filtered_data == True:
        coefs, freqs = pywt.cwt(data, scale, wavelet, sampling_period=tr.stats.delta)
    else:
        coefs, freqs = pywt.cwt(tr.data, scale, wavelet, sampling_period=tr.stats.delta)
    scalogram = (np.abs(coefs))**2 # power spectrum
    if log_normalize == True:
        scalogram = np.log10(scalogram + 1e-10) # log normalize (small constant added to avoid log10(0))
    if f_lim is not None:
        ix = np.where((freqs >= (f_lim[0]) - (f_lim[0]/10) ) & (freqs <= (f_lim[1]) + (f_lim[1]/10))) # add small constant to make sure range encompasses both f_lims
    else:
        ix = np.where((freqs >= min(freqs)) & (freqs <= (tr.stats.sampling_rate/2) + ((tr.stats.sampling_rate/2)/10))) # need to add small constant to make sure Nyquist is encompassed in range
    if v_lim is not None:
        vmin = v_lim[0]; vmax = v_lim[1]
    else:
        vmin = scalogram[ix].min(); vmax = scalogram[ix].max()
    freqs = freqs[ix]
    scalogram = scalogram[ix,:][0]
    #-----------------------------------------------------------------------------------------------------------------------#
    # Plot scalogram
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)
    cmap = cm.get_cmap(colormap, 256)
    cmap = cmap(np.linspace(2,0.001,100))
    cmap = ListedColormap(cmap)
    plt.pcolormesh(t, freqs, scalogram, cmap=cmap, shading=shading, vmin=vmin, vmax=vmax)
    colorbar = plt.colorbar(location=colorbar_location, pad=colorbar_pad)
    if amp_units == 'Amplitude': # make sure amplitude units are input properly for plotting
        units_str = amp_units
    else:
        try:
            units_str = re.findall(r'\[(.*?)\]', amp_units)[0]
        except Exception as inst:
            print(inst)
            print('Make sure to include amplitude units in brackets (e.g., Pressure [Pa])')
    if log_normalize == True:
        colorbar.set_label('Log10['+units_str+'\u00b2]', fontsize=10)
    else:
        colorbar.set_label('['+units_str+'\u00b2]', fontsize=10)
    if log_scale == True:
        ax2.set_yscale('log')
    if (plot_UTC == True) and (t_lim == None):
        ax2.set_xlabel(global_start.strftime('%Y-%m-%d')+' Time [UTC]')
    elif (plot_UTC == False) and (t_lim is not None):
        ax2.set_xlabel('Time [s] after ' + str(global_start).split('.')[0])
    elif (plot_UTC == False) and (t_lim == None):
        ax2.set_xlabel('Time [s] after ' + str(global_start).split('.')[0])
    elif (plot_UTC == True) and (t_lim is not None):
        raise Exception("Can't have t_lim and plot_UTC set to True, since t_lim is relative.")
    ax2.set_ylabel('Frequency [Hz]')
    # Save figure
    if fname_plot is not None:
        fig.savefig(fname_plot)

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def plot_zero_crossings(st, element=None, bandpass=[0.5,5], resample=100, taper='cosine', max_percentage=0.05, max_length=60, n_crossings=[2], smooth=False, sigma=10, amp_lim=None, amp_units='Amplitude', 
                        legend_loc='upper right', title=None, figsize=(12,6), fname_plot=None):
    '''----------------------------------------------------------------------------------------------------------------------------------
    Computes period (avg and std) and peak frequency using zero-crossings method

    Input:
        st: ObsPy stream object
        element (str): array element name used to compute period at zero-crossings (defaults to first element in array)
        bandpass (list): frequency range to use to bandpass time-series data prior to period measurements (defaults to [0.5,5])
        resample (int): integer to resample stream to (defaults to 100 - increase if zero-crossings aren't aligned well)
        taper (str): type of taper to use - defaults to "cosine"
        max_percentage (float/int): percentage to use for taper
        max_length (float/int): length to use for taper
        n_crossings (list): number of zero-crossings to compute to the left and right of the peak amplitude - if only one value it will use that for both sides of peak
        smooth (boolean): whether to smooth waveform data (only necessary if waveform isn't sufficiently coherent)
        sigma (int): standard deviation of the Gaussian kernel used for smoothing
        amp_lim (list): range in amplitude for visualization in time-series plot
        amp units (str): units of measurement on y-axis
        legend_loc (str): relative location of legend within plot
        title (str): title of plot
        figsize (tuple): sets the figure size
        fname_plot (str): filename to save zero-crossings plot
    ----------------------------------------------------------------------------------------------------------------------------------'''
    fig = plt.figure(figsize=figsize)
    # Resample to improve zero-crossings
    st_resample = st.copy()
    if resample is not None:
        st_resample = st_resample.resample(resample)
    # Filter
    if bandpass is not None:
        st_taper = st_resample.copy()
        st_taper.taper(type=taper, max_percentage=max_percentage, max_length=max_length)
        st_filt = st_taper.copy()
        try:
            st_filt.filter(type='bandpass', freqmin=min(bandpass), freqmax=max(bandpass))
        except:
            st_filt = st_taper.copy().split()
            st_filt.filter(type='bandpass', freqmin=min(bandpass), freqmax=max(bandpass))
            st_filt = st_filt.merge() 
    if element is not None:
        tr = st_filt.select(station=element)[0]
    else:
        tr = st_filt[0].copy()
    t = np.arange(0, tr.stats.npts*tr.stats.delta, tr.stats.delta)
    data = tr.data.copy()
    if smooth == True:
        data = gaussian_filter(data, sigma=sigma)
    if element == 'Beam':
        plt.plot(t, data, color='grey', label=tr.stats.station)
    else:
        plt.plot(t, data, color='k', label=tr.stats.station)
    if amp_lim is not None:
        plt.ylim(amp_lim)
    plt.axhline(y=0, color='red', lw=0.5)
    plt.legend(loc=legend_loc)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Compute zero-crossings centered around peak amplitude
    if len(n_crossings) == 1:
        n_crossings_before = n_crossings[0]; n_crossings_after = n_crossings[0]
    elif len(n_crossings) > 1:
        n_crossings_before = n_crossings[0]; n_crossings_after = n_crossings[1]
    peak_idx = np.argmax(np.abs(data))
    before_peak_idxs = np.where(np.diff(np.sign(data[:peak_idx])))[0]
    times_before = t[before_peak_idxs[-n_crossings_before:]] if len(before_peak_idxs) >= n_crossings_before else t[before_peak_idxs]
    after_peak_idxs = np.where(np.diff(np.sign(data[peak_idx:])))[0]
    times_after = t[peak_idx + after_peak_idxs[:n_crossings_after]] if len(after_peak_idxs) >= n_crossings_after else t[peak_idx + after_peak_idxs]
    times = np.concatenate((times_before, times_after))
    x1 = times[0]; x2 = times[1]; x3 = times[2]; x4 = times[3]
    #-----------------------------------------------------------------------------------------------------------------------#
    # Plot time points
    plt.axvline(x=x1, lw=1, color='blue', ls='--')
    plt.axvline(x=x2, lw=1, color='blue', ls='--')
    plt.axvline(x=x3, lw=1, color='blue', ls='--')
    plt.axvline(x=x4, lw=1, color='blue', ls='--')
    #-----------------------------------------------------------------------------------------------------------------------#
    # Compute period
    p = []
    p += [abs( (x1) - (x2) )*2]
    p += [abs( (x2) - (x3) )*2]
    p += [abs( (x3) - (x4) )*2]
    p += [abs( (x1) - (x3) )]
    p += [abs( (x2) - (x4) )]
    p = np.array(p); f = 1/p
    #-----------------------------------------------------------------------------------------------------------------------#
    plt.xlim([x1 - (x4-x1), x4 + (x4-x1)])
    plt.xlabel('Time [s] Relative to '+str(st[0].stats.starttime).split('.')[0].replace('T', ' ') + ' [UTC]')
    plt.ylabel(amp_units)
    if title is None:
        plt.title('Zero-Crossings \n Avg Period: %.3f'%np.mean(p)+ ' [s] - Standard Deviation: %.3f'%np.std(p) + ' [s] - Avg Peak Freq: %.3f'%np.mean(f) + ' [Hz] - Standard Deviation: %.3f'%np.std(f) + ' [Hz]')
    else:
        plt.title(title + ' Zero-Crossings \n Avg Period: %.3f'%np.mean(p)+ ' [s] - Standard Deviation: %.3f'%np.std(p) + ' [s] - Avg Peak Freq: %.3f'%np.mean(f) + ' [Hz] - Standard Deviation: %.3f'%np.std(f) + ' [Hz]')
    # Save figure
    if fname_plot is not None:
        fig.savefig(fname_plot)

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def plot_cross_correlation(st, f_bands, bandpass=[0.5,5], taper='cosine', max_percentage=0.05, max_length=60, delay_times=None, data_envelope=False, trace_spacing=2.25, t_lim=None, 
                           plot_UTC=False, UTC_time_interval=None, families=None, families_table=None, family_idx=None, trim_family_window=None, v_lim=None, return_xcorr_params=False, 
                           title1=None, title2=None, fname_plot=None, figsize=(18,6)):
    '''----------------------------------------------------------------------------------------------------------------------------------
    Computes spatial cross-correlation matrix of signal and spatiotemporal cross-correlation (automatically normalizes trace data)

    Input:
        st: ObsPy stream object
        f_bands (Pandas DataFrame): contains frequency bands and time windows constructed using the Segmentor
        bandpass (list): frequency range to use to bandpass time-series data prior to period measurements (defaults to [0.5,5])
        taper (str): type of taper to use - defaults to "cosine"
        max_percentage (float/int): percentage to use for taper
        max_length (float/int): length to use for taper
        delay_times (list): theoretical tropospheric, stratospheric, and thermospheric arrival times (in seconds relative to stream start time)
        data_envelope (boolean): whether to plot and cross-correlate array data envelopes
        trace_spacing (int/float): vertical spacing between each normalized trace (defaults to 2.25)
        t_lim (list): range in time window start and end times to plot (in seconds relative to start time)
        plot_UTC (boolean): whether to plot time in UTC
        UTC_time_interval (int): time interval (in minutes) to be used for x-axis UTC formatting
        families (list): each entry contains unique pixel ID's in that family (output from make_families - used for specifying windows for cross-correlation)
        families_table (Pandas DataFrame): dataframe of families sorted by time (output from df_families - used for specifying windors for cross-correlation)
        family_idx (int): index of family detection to be used for beamforming and/or PSD construction (if None, family with max semblance will be used)
        trim_family_window (list): both inputs decrease family window size relative to the start (similar to t_lim)
        v_lim (list): range in normalized cross-correlation values to plot for matrix (defaults to [0,1])
        return_xcorr_params (boolear): whether to return maximum normalized cross-correlation coefficients, lag times, and reference signal associated with matrix
        title1 (str): title of waveforms plot
        title2 (str): title of cross-correlation matrix
        fname_plot (str): filename to save spectrogram plot
        figsize (tuple): specifies size of figure
    ----------------------------------------------------------------------------------------------------------------------------------'''
    # Plot normalized waveforms
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(1,2,1)
    # Filter
    if bandpass is not None:
        st_taper = st.copy()
        st_taper.taper(type=taper, max_percentage=max_percentage, max_length=max_length)
        st_filt = st_taper.copy()
        try:
            st_filt.filter(type='bandpass', freqmin=min(bandpass), freqmax=max(bandpass))
        except:
            st_filt = st_taper.copy().split()
            st_filt.filter(type='bandpass', freqmin=min(bandpass), freqmax=max(bandpass))
            st_filt = st_filt.merge() 
    # Time and data vectors
    t = np.arange(0, st_filt[0].stats.npts*st_filt[0].stats.delta, st_filt[0].stats.delta)
    data_array_tmp = []
    for tr in st_filt:
        tr_data_tmp = tr.data / np.max(np.abs(tr.data)) # normalize by trace
        data_array_tmp.append(tr_data_tmp)
    data_array_tmp = np.array(data_array_tmp); data_array = []
    for tr_idx in range(data_array_tmp.shape[0]):
        t, tr_idx_data = fix_lengths(t, data_array_tmp[tr_idx,:]) # will be using this time vector
        data_array.append(tr_idx_data)
    data_array = np.array(data_array)
    # Plot array data
    k = 0; ticks = []; labels = []
    for data_idx, tr in zip(range(data_array.shape[0]), st_filt):
        ax1.plot(t, data_array[data_idx,:]+k, color='k')
        if data_envelope == True:
            ax1.plot(t, np.abs(signal.hilbert(data_array[data_idx,:]))+k, color='blue', lw=3) # plot envelope if specified by user
        ticks.append(k); labels.append(tr.stats.station)
        k += trace_spacing
    plt.yticks(ticks,labels)
    global_start = min(tr.stats.starttime for tr in st)
    global_end = max(tr.stats.endtime for tr in st)
    if (plot_UTC == True) and (t_lim == None):
        # Plot UTC - round down to the nearest minute_interval for start
        if UTC_time_interval is not None:
            minute_interval = UTC_time_interval
        else:
            if global_end-global_start <= 3600*2:
                minute_interval = 15 # less than 2 hours make it 15 minute intervals
            else:
                # round to the nearest 15 minutes and partition data stream into 6 segements if stream is longer than 2 hours
                n = global_end-global_start
                sec_round = round(n / 900) * 900
                min_round = sec_round // 60
                minute_interval = min_round // 6
        rounded_start_tick = global_start.replace(minute=(global_start.minute // minute_interval) * minute_interval, second=0, microsecond=0)
        if rounded_start_tick < global_start:
            rounded_start_tick += (minute_interval*60)
        x_tick_positions = np.arange(rounded_start_tick - global_start, global_end - global_start, minute_interval*60)
        x_tick_labels = [(global_start + t).strftime('%H:%M:%S') for t in x_tick_positions]
        ax1.set_xticks(x_tick_positions); ax1.set_xticklabels(x_tick_labels)
        ax1.set_xlim([0,(global_end-global_start)]); ax1.set_xlabel(global_start.strftime('%Y-%m-%d')+' Time [UTC]')
    elif (plot_UTC == False) and (t_lim is not None):
        ax1.set_xlim(t_lim)
        ax1.set_xlabel('Time [s] after ' + str(global_start).split('.')[0])
    elif (plot_UTC == False) and (t_lim == None):
        ax1.set_xlim([t[0],t[-1]])
        ax1.set_xlabel('Time [s] after ' + str(global_start).split('.')[0])
    elif (plot_UTC == True) and (t_lim is not None):
        raise Exception("Can't have t_lim and plot_UTC set to True, since t_lim is relative.")
    plt.ylabel('Trace')
    if (title1 is not None) and (bandpass is not None):
        ax1.set_title(title1+ " - Normalized Waveforms\nBandpass: " + str(bandpass))
    elif (title1 is not None) and (bandpass == None):
        ax1.set_title(title1)
    elif (title1 == None) and (bandpass is not None):
        ax1.set_title("Normalized Waveforms\nBandpass: " + str(bandpass))
    #-----------------------------------------------------------------------------------------------------------------------#
    # Compute spatiotemporal cross-correlation and map to waveforms
    val = np.mean([max(bandpass),min(bandpass)])
    f_val = find_closest(val, f_bands['fcenter'].values)
    f_idx = np.where((f_bands['fcenter'].values == f_val))[0][0]
    win_len = f_bands['win'].values[f_idx]; step = f_bands['step'].values[f_idx] # need to find win and step values from f_bands using bandpass
    # Step through time-series recordings
    starttime = 0; endtime = st[0].stats.endtime - st[0].stats.starttime
    xcorr_coefs = []; times = []
    for win_start in np.arange(0, endtime-starttime, step):
        if win_start + win_len > endtime:
            break
        # Extract data
        xcorr_coef = []
        t_win, data_array_win = data_time_window(t, data_array, t_start=win_start, t_end=win_start+win_len)
        for tr_idx in range(data_array.shape[0]):
            tr_data_idx = data_array_win[tr_idx,:]
            tr_xcorr_coef = []
            for i in range(data_array.shape[0]):
                if tr_idx == i:
                    continue
                _, xcorr_vals = norm_xcorr(t_win, tr_data_idx, data_array_win[i,:])
                tr_xcorr_coef.append(xcorr_vals.max())
            xcorr_coef.append(np.mean(tr_xcorr_coef)) # take the average maximum correlation coefficient for each trace
        xcorr_coef = np.array(xcorr_coef) # values are average max corr coef, rows are stations, column is time sample
        xcorr_coefs.append(xcorr_coef); times.append(win_start + (win_len/2))
    xcorr_coefs = np.array(xcorr_coefs).T; times = np.array(times).reshape(1,len(times))
    # Map times to stream times
    new_times = np.zeros((1,times.shape[1]))
    for times_idx in range(times.shape[1]):
        t_idx = map_time_to_sample(times[0,times_idx], t)
        new_times[0,times_idx] = t[t_idx]
    if v_lim is not None:
        vmin = v_lim[0]; vmax = v_lim[1]
    else:
        vmin = 0; vmax=1
    k = 0
    for tr_idx in range(xcorr_coefs.shape[0]):
        # Interpolate xcorr values to match waveform's time samples
        xcorr_coefs_interp = np.interp(t, new_times[0,:], xcorr_coefs[tr_idx,:])
        # Create segments of the waveform for coloring
        points = np.array([t, data_array[tr_idx,:]+k]).T.reshape(-1,1,2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # Use colormap to map cross-correlation values to colors
        colors = plt.cm.hot_r(np.linspace(0,1,256))
        cmap = LinearSegmentedColormap.from_list('custom_hot_r', colors)
        norm = Normalize(vmin=vmin, vmax=vmax)
        # Create LineCollection for waveform
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(xcorr_coefs_interp)
        lc.set_linewidth(2)
        # Plot waveform
        ax1.add_collection(lc)
        ax1.autoscale()
        k += trace_spacing
    if delay_times is not None:
        plt.axvline(x=delay_times[0], lw=1, ls='--', color='red')
        plt.axvline(x=delay_times[1], lw=1, ls='--', color='green')
        plt.axvline(x=delay_times[2], lw=1, ls='--', color='blue')
    #-----------------------------------------------------------------------------------------------------------------------#
    # Cross-correlation matrix using families detection
    if (families is not None) and (families_table == None):
        families_table = df_families(st[0].stats.starttime.matplotlib_date, families)
    elif (families == None) and (families_table is not None):
        pass
    elif (families == None) and (families_table == None):
        raise Exception('If generating cross-correlatin matrix, families or families_table must be specified.')
    # Choose family (either user defined or defaults to family with max semb)
    if family_idx is not None:
        fam_start = families_table['start_time'][family_idx]
        fam_end = families_table['end_time'][family_idx]
    else:
        family_idx = np.where((families_table['max_semb'] == families_table['max_semb'].max()))[0][0] # choose family with max semblance if not specified by user
        fam_start = families_table['start_time'][family_idx]
        fam_end = families_table['end_time'][family_idx]
    if trim_family_window is not None:
        sig_start = fam_start + trim_family_window[0]
        sig_end = fam_start + trim_family_window[1]
    else:
        sig_start = fam_start.copy()
        sig_end = fam_end.copy()
    # Shade in region of array data that is being used for cross-correlation matrix
    y_fill = np.linspace(data_array.min()-0.25, data_array.max()+k-trace_spacing+0.25, 1000)
    ax1.fill_betweenx(y_fill, x1=sig_start, x2=sig_end, alpha=0.25, color='grey')
    ax1.set_ylim(data_array.min()-0.25, data_array.max()+k-trace_spacing+0.25)
    # Extract data and compute cross-correlation
    if data_envelope == True:
        for data_idx in range(data_array.shape[0]):
            data_array[data_idx,:] = np.abs(signal.hilbert(data_array[data_idx,:])) # cross-correlate envelope if specified by user
    t_matrix, data_matrix = data_time_window(t, data_array, t_start=sig_start, t_end=sig_end)
    xcorr_coef_matrix, xcorr_lag_times, ref_signal = xcorr_matrix(t_matrix, data_matrix)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Plot correlation matrix
    ax2 = fig.add_subplot(1,2,2)
    windows = []
    for tr in st:
        windows.append(tr.stats.station)
    ticks = np.arange(0,len(st),1)
    plt.pcolormesh(xcorr_coef_matrix, vmin=vmin, vmax=vmax, cmap=cmap)
    colorbar = plt.colorbar(ax=ax2)
    colorbar.set_label('Normalized Correlation Coefficient')
    plt.xticks(ticks, windows, ha='left'); plt.yticks(ticks, windows, va='baseline')
    plt.xlabel('Trace')
    if title2 == None:
        if data_envelope == True:
            plt.title('Envelope Correlation Matrix')
        else:
            plt.title('Correlation Matrix')
    else:
        if data_envelope == True:
            plt.title(title2 + ' Envelope Correlation Matrix')
        else:
            plt.title(title2 + ' Correlation Matrix')
    # Save figure
    if fname_plot is not None:
        fig.savefig(fname_plot)
    if return_xcorr_params == True:
        return xcorr_coef_matrix, xcorr_lag_times, ref_signal
'----------------------------------------------'
'---------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------'
'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------'
'---------------------------------------------------------------------------'
'----------------------------------------------'
############### ############### ###############
###############  Neural Network ###############
############### ############### ###############

# The transformer architecture (no dropout)
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim)
        ])
        
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, dropout=False, mask=None, return_attention=False):
        attn_output, attn_weights = self.att(
            inputs, inputs, attention_mask=mask, return_attention_scores=True
        )  # <- Extract attention scores
        
        out1 = self.layernorm1(inputs + attn_output)
        out1 = self.dropout1(out1, training=dropout)
        
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + ffn_output)
        out2 = self.dropout2(out2, training=dropout)

        if return_attention == True:
            return out2, attn_weights  # Return both outputs and attention weights
        else:
            return out2

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def GMU(x_s, x_i, HIDDEN_STATE_DIM=16):
    '''----------------------------------------------------------------------------------------------------------------------------------
    Gated multimodal unit (Arevalo et al., 2020)
    ----------------------------------------------------------------------------------------------------------------------------------'''
    h_s = Dense(units=HIDDEN_STATE_DIM,
                kernel_initializer='glorot_uniform')(x_s)
    h_s = activations.tanh(h_s)

    h_i = Dense(units=HIDDEN_STATE_DIM,
                kernel_initializer='glorot_uniform')(x_i)
    h_i = activations.tanh(h_i)

    x = concatenate([x_s, x_i], axis=1)
    z = Dense(HIDDEN_STATE_DIM,
              activation='sigmoid',
              name='z_layer',
              kernel_initializer='glorot_uniform')(x)

    h = z * h_s + (1 - z) * h_i

    return h

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def build_RC_ConvFormer(X_train_geometries, X_train_freqs, lr=1e-3):
    '''----------------------------------------------------------------------------------------------------------------------------------
    Compile the neural network used to predict RC metric
    ----------------------------------------------------------------------------------------------------------------------------------'''
    # Define optimizer
    adam = Adam(learning_rate=lr, epsilon=1e-6)
    #-----------------------------------------------------------------------------------------------------------------------#
    # We need to define the shape of our input tensor first
    input_tensor_geometries = Input(shape=(X_train_geometries.shape[1], X_train_geometries.shape[2], 1), name='Input_Geometries')
    input_tensor_freqs = Input(shape=(X_train_freqs.shape[1],), name='Input_Freqs')
    #-----------------------------------------------------------------------------------------------------------------------#
    # Need to separate continuous features and OHE before convolution
    input_tensor_geometries_continuous = input_tensor_geometries[:,:,:2]
    ohe = input_tensor_geometries[:,:,2:] # only taking one row of OHE's
    #-----------------------------------------------------------------------------------------------------------------------#
    # Conv 1 
    x = Conv2D(16,
               (3,3),
               strides=(1,1), # [10,4]
               padding='same',
               kernel_initializer='he_uniform', 
               kernel_regularizer=regularizers.l2(1e-5), # weak regularization
               name='Conv_1')(input_tensor_geometries_continuous)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Conv 1 - Resblock
    x_res = BatchNormalization()(x)
    x_res = activations.relu(x_res)
    x_res = Dropout(0.2)(x_res)
    x_res = Conv2D(16,
                   (3,3),
                   strides=(1,1), 
                   padding='same',
                   kernel_initializer='he_uniform', 
                   name='Conv_1a')(x_res)
    x_res = BatchNormalization()(x_res)
    x_res = activations.relu(x_res)
    x_res = Conv2D(16,
                   (3,3),
                   strides=(1,1), 
                   padding='same',
                   kernel_initializer='he_uniform', 
                   name='Conv_1b')(x_res)
    out = Add()([x, x_res])
    x = BatchNormalization()(out)
    x = activations.relu(x)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Conv 2
    x = Conv2D(32,
               (3,3),
               strides=(1,1),
               padding='same',
               kernel_regularizer=regularizers.l2(1e-4), # moderate regularization
               kernel_initializer='he_uniform', 
               name='Conv_2')(x)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Conv 2 - Resblock
    x_res = BatchNormalization()(x)
    x_res = activations.relu(x_res)
    x_res = Dropout(0.2)(x_res)
    x_res = Conv2D(32,
                   (3,3),
                   strides=(1,1), 
                   padding='same',
                   kernel_initializer='he_uniform', 
                   name='Conv_2a')(x_res)
    x_res = BatchNormalization()(x_res)
    x_res = activations.relu(x_res)
    x_res = Conv2D(32,
                   (3,3),
                   strides=(1,1), 
                   padding='same',
                   kernel_initializer='he_uniform', 
                   name='Conv_2b')(x_res)
    out = Add()([x, x_res])
    x = BatchNormalization()(out)
    x = activations.relu(x)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Concatenate OHE to frequency bands
    x = concatenate([x, ohe], axis=-1) # concatenate OHE channel-wise
    #-----------------------------------------------------------------------------------------------------------------------#
    # Transformer block 1
    x = TransformerBlock(embed_dim=33, num_heads=4, ff_dim=64)(x, mask=None)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Dense layer 1 Arrays
    x = Dense(units=16, 
              kernel_initializer='he_uniform', 
              name='Dense_1_geometries')(x)
    x = BatchNormalization()(x)
    x = activations.relu(x)
    x = Dropout(0.5)(x)
   #-----------------------------------------------------------------------------------------------------------------------#
    # Dense layer 2 arrays
    x = Dense(units=8, 
              kernel_initializer='he_uniform',
              name='Dense_2_geometries')(x)
    x = BatchNormalization()(x)
    x = activations.relu(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Dense layer 1 Params
    x_p = Dense(units=16, 
              kernel_initializer='he_uniform', 
              name='Dense_1_freqs')(input_tensor_freqs)
    x_p = BatchNormalization()(x_p)
    x_p = activations.relu(x_p)
    x_p = Dropout(0.5)(x_p)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Dense layer 2 Params
    x_p = Dense(units=8, 
              kernel_initializer='he_uniform', 
              name='Dense_2_freqs')(x_p)
    x_p = BatchNormalization()(x_p)
    x_p = activations.relu(x_p)
    x_p = Dropout(0.25)(x_p)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Gated Multimodal Unit
    x = GMU(x, x_p, HIDDEN_STATE_DIM=32)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Dense layer 1
    x = Dense(units=16, 
              kernel_initializer='he_uniform', 
              name='Dense_1')(x)
    x = BatchNormalization()(x)
    x = activations.relu(x)
    x = Dropout(0.25)(x)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Dense layer 2
    x = Dense(units=8, 
              kernel_initializer='he_uniform',
              kernel_regularizer=regularizers.l2(1e-3), # strong regularization
              name='Dense_2')(x)
    x = BatchNormalization()(x)
    x = activations.relu(x)
    x = Dropout(0.25)(x)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Predict
    predictions = Dense(1, # RC metric
                        name='Predictions')(x)
    #-----------------------------------------------------------------------------------------------------------------------#
    model = Model(inputs=[input_tensor_geometries,input_tensor_freqs], outputs=predictions, name='RC_ConvFormer')
    #-----------------------------------------------------------------------------------------------------------------------#
    model.compile(optimizer=adam, loss=tf.keras.losses.MeanSquaredError())
    #-----------------------------------------------------------------------------------------------------------------------#
    return model

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def train_models(split, epochs=1000, batch_size=32, lr=1e-3, patience=50, n_splits=5):
    '''----------------------------------------------------------------------------------------------------------------------------------
    Trains and saves results for RC ConvFormer models within each split
    ----------------------------------------------------------------------------------------------------------------------------------'''
    print('Begin split # ' +str(split+1) + ' of '+ str(n_splits))
    # Load train
    X_train_geometries_tmp = np.load('Training_Data/Split_'+str(split+1)+'/X_train_final.npy')
    X_train_freqs_tmp = np.load('Training_Data/Split_'+str(split+1)+'/F_train_final.npy')
    y_train_tmp = np.load('Training_Data/Split_'+str(split+1)+'/y_train_final.npy')
    # Load test
    X_test_geometries_tmp = np.load('Training_Data/Split_'+str(split+1)+'/X_val_final.npy')
    X_test_freqs_tmp = np.load('Training_Data/Split_'+str(split+1)+'/F_val_final.npy')
    y_test_tmp = np.load('Training_Data/Split_'+str(split+1)+'/y_val_final.npy')
    #-----------------------------------------------------------------------------------------------------------------------#
    # Build seismic model
    RC_model = build_RC_ConvFormer(X_train_geometries_tmp, X_train_freqs_tmp, lr=lr)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Callbacks - Define file path for saving the model in .keras format
    checkpoint_filepath = 'RC_Model_'+str(split+1)+'.keras'
    #-----------------------------------------------------------------------------------------------------------------------#
    # Create the ModelCheckpoint callback
    checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, # Filepath to save the model in .keras format
                                 monitor='val_loss', # Metric to monitor (e.g., validation loss)
                                 save_best_only=True, # Save only if the current epoch is the best
                                 mode='min', # 'min' for loss (lower is better), 'max' for metrics like accuracy
                                 verbose=1 # Print messages when saving
                                )
    #-----------------------------------------------------------------------------------------------------------------------#
    # Early stopping to reduce overfitting
    early_stopping = EarlyStopping(monitor='val_loss', # Metric to monitor
                                   patience=patience, # Number of epochs with no improvement before stopping
                                   mode='min', # 'min' because we are monitoring loss (lower is better)
                                   min_delta=0.00001, # minimum improvement to be considered an improvement
                                   restore_best_weights=True, # Restore the best weights after stopping
                                   verbose=1 # Print messages when stopping
                                  )
    #-----------------------------------------------------------------------------------------------------------------------#
    # Save training histories
    csv_logger = CSVLogger('training_log_'+str(split+1)+'.csv', append=True)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Train model
    print('Start training RC model')
    history = RC_model.fit([X_train_geometries_tmp, X_train_freqs_tmp], y_train_tmp, validation_data=([X_test_geometries_tmp, X_test_freqs_tmp], y_test_tmp), epochs=epochs, batch_size=batch_size,
                           callbacks=[checkpoint, early_stopping, csv_logger])