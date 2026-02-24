import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
os.environ["ABSL_LOGGING_MIN_LOG_LEVEL"] = "3"  # absl logs to stderr

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import warnings, sys, platform, logging, multiprocessing, psutil, cardinal, argparse, json, time

warnings.filterwarnings("ignore")
logging.getLogger('distributed.nanny').setLevel(logging.CRITICAL)

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from obspy import *

# Specify client for parallel computing
# Good starting point is n_workers = number of cores / 2...make sure n_workers * memory_limit is less than total memory available
from dask.distributed import Client

def main():
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    print(f"Python Platform: {platform.platform()}")
    print(f"Tensorflow Version: {tf.__version__}")
    print(f"Keras Version: {tf.keras.__version__}")
    print()
    print(f"Python {sys.version}")
    print()
    print(f"Number of cores available for processing: {multiprocessing.cpu_count()}")
    print(f"Total memory available: {psutil.virtual_memory().total / 1e9, 'GB'}")

    n_workers = 4
    client = Client(processes=True, threads_per_worker=1, n_workers=n_workers, memory_limit='3GB')
    try:
        ### Ingest command line args and param file ###
        params = cardinal.parse_args()

        print("Loaded parameters:")
        for k, v in params.items():
            print(f"  {k}: {v}")

        ### Setting variables from params ###

        ### Waveform type variables ###
        data_type = params['Waveform']['type']
        provider = params['Waveform']['source']['provider']
        network = params['Waveform']['source']['network']
        array = params['Waveform']['source']['array']
        channel = params['Waveform']['source']['channel']
        data_filepath = params['Waveform']['source']['datapath']
        array_coords_filepath = params['Waveform']['source']['sitepath']
        korea_arrays = params['Waveform']['source']['korea_arrays']
        remove_response = params['Waveform']['source']['remove_response']
        remove_stations = params['Waveform']['source']['remove_stations']
        convert_units = params['Waveform']['source']['convert_units']
        amp_units = params['Waveform']['source']['amp_units']

        ### Event variables ###
        use_event = params['Event']['use']
        event_time = params['Event']['time']
        source_lat = params['Event']['lat']
        source_lon = params['Event']['lon']

        ### Segmentor ###
        f_min = params['Segmentor']['f_min']
        f_max = params['Segmentor']['f_max']
        win_min = params['Segmentor']['win_min']
        win_max = params['Segmentor']['win_max']
        segm_type = params['Segmentor']['type']

        ### Array Processing ###
        process = params['ArrayProcessing']['process']
        adaptive_array = params['ArrayProcessing']['adaptive_array']
        signal_type = params['ArrayProcessing']['signal_type']
        semblance_threshold = params['ArrayProcessing']['Plotting']['semblance_threshold']
        clim_vtr = params['ArrayProcessing']['Plotting']['clim_vtr']
        clim_baz = params['ArrayProcessing']['Plotting']['clim_baz']
        bandpass = params['ArrayProcessing']['Plotting']['bandpass']
        normalize = params['ArrayProcessing']['Plotting']['normalize']
        scal_subplot = params['ArrayProcessing']['Plotting']['scal_subplot']
        spec_subplot = params['ArrayProcessing']['Plotting']['spec_subplot']
        if scal_subplot and spec_subplot:
            raise ValueError(
                "Invalid plotting configuration: "
                "'scal_subplot' and 'spec_subplot' cannot both be True. "
                "Set at most one of them to True (or both to False)."
            )

        ### Families ###
        dist_threshold = params['Families']['dist_threshold']
        min_pixels = params['Families']['min_pixels']
        sigma_t = params['Families']['sigma_t']
        sigma_f = params['Families']['sigma_f']
        sigma_b = params['Families']['sigma_b']
        p_threshold = params['Families']['p_threshold']
        family_grouping = params['Families']['family_grouping']

        starttime = UTCDateTime(params['starttime'])
        endtime = UTCDateTime(params['endtime'])
        plot_plots = params['Plot']
        user = params.get("user")
        password = params.get("password")

        ### Option to read form FDSN ###
        if data_type == 'FDSN':

            result = cardinal.get_waveforms(
                provider=provider,
                network=network,
                station=array,
                location="*",
                channel=channel,
                starttime=str(starttime).split('.')[0],
                endtime=str(endtime).split('.')[0],
                mseed_out=Path(data_filepath),
                site_out=Path(array_coords_filepath),
                korea_arrays=(korea_arrays, user, password),
                remove_response=remove_response, # defaults to velocity (set integrate to True in plot_array_data() if you want displacement)
                verbose=True,
            )

        ### Option to read from Oracle DB ###
        if data_type == 'DB':
            print('cheese')
            sitedf = cardinal.fetch_wfdisc_rows(array, channel, starttime, endtime, provider, data_filepath, array_coords_filepath)

        ### Read data ###
        if use_event:
            st, st_filt, delay_times, GT_baz, distance = cardinal.plot_array_data(
                data_filepath, array_coords_filepath, event_time=event_time, source_lat=source_lat, source_lon=source_lon,
                array=array, channel=channel, trim_stream=[starttime, endtime], remove_stations=remove_stations, convert_units=convert_units, amp_units=amp_units
            )

        elif not use_event:
            st, st_filt = cardinal.plot_array_data(
                data_filepath, array_coords_filepath,
                array=array, channel=channel, trim_stream=[starttime, endtime], remove_stations=remove_stations, convert_units=convert_units, amp_units=amp_units
            )
            delay_times = None
            GT_baz = None
            distance = None

        # Removed sensors
        if remove_stations:
            for lst_stn in remove_stations:
                print("Removed sensor " + str(lst_stn) + " for processing")

        ### Site dataframe ###
        sitedf = pd.read_csv(array_coords_filepath, delim_whitespace=True, header=None, names=["ELEM", "LAT", "LON", "ELEV"])
        sitedf = sitedf.iloc[0]

        ### Plot arry coordinates ###
        ref_station = st[0].stats.station
        X, stnm = cardinal.get_array_coords(st, ref_station, units='km')
        cardinal.plot_array_coords(X, stnm, units='km')
        plt.title(array + ' Coordinates')

        if process:
            ### Segmenting freqeuncy bands ###
            f_bands = cardinal.make_custom_fbands(f_min=f_min, f_max=f_max, win_min=win_min, win_max=win_max, type=segm_type)
            f_bands['fmax'].values[-1] = np.round(f_bands['fmax'].values[-1], 0)  # rounding so f_max becomes Nyquist
            print(f_bands)

            ### Adaptive array and array processing ###
            t0 = time.perf_counter()
            cache_dir = "./.cardinal_cache"  # or argparse option
            if adaptive_array: verbose_clusters = True
            else: verbose_clusters = False
            _, _, k = cardinal.clusters(st, plot=plot_plots, verbose=verbose_clusters)
            if adaptive_array and k >= 3: # run adaptive array if true and 3 or more clusters
                key = cardinal.make_adaptive_key(st, f_bands, k, array_type=signal_type)
                subarrays_stnms = cardinal.load_subarrays_cache(cache_dir, key)

                if subarrays_stnms is None:
                    _, subarrays_stnms = cardinal.adaptive_array(
                        st, f_bands,
                        array_type=signal_type,
                        plot=plot_plots,
                        n_clusters=k
                    )
                    cardinal.save_subarrays_cache(cache_dir, key, subarrays_stnms)
                    print(f"[adaptive_array] computed + cached (key={key[:8]})")
                else:
                    print(f"[adaptive_array] loaded from cache (key={key[:8]})")

                st_subarrays = cardinal.retrieve_subarray_data(st, subarrays_stnms)
                T, B, V, S = cardinal.sliding_time_array_fk_multifreq(
                    st_subarrays, f_bands, client,
                    signal_type=signal_type,
                    adaptive_array=True
                )
                title = array + " - Adaptive Array"

            else:
                T, B, V, S = cardinal.sliding_time_array_fk_multifreq(
                    st, f_bands, client,
                    signal_type=signal_type,
                    adaptive_array=False
                )
                title = array
            dt = time.perf_counter() - t0
            print(f"[array processing] elapsed time: {dt:.2f} s ({dt/60:.2f} min)")

            ### Plottting array processing results ###
            cardinal.plot_sliding_window_multifreq(
                st, f_bands, T, B, V, S, spec_subplot=spec_subplot, scal_subplot=scal_subplot, semblance_threshold=semblance_threshold, clim_vtr=clim_vtr, clim_baz=clim_baz, bandpass=bandpass, normalize=normalize,
                GT_baz=GT_baz, delay_times=delay_times, amp_units=amp_units, title=title
            )

            # Creating Families
            ref_time = st[0].stats.starttime.matplotlib_date
            ix, pixels_in_families, families = cardinal.make_families(
                T, B, V, S, f_bands, ref_time,
                dist_threshold=dist_threshold, min_pixels=min_pixels, sigma_t=sigma_t, sigma_f=sigma_f, sigma_b=sigma_b, p_threshold=p_threshold,
                family_grouping=family_grouping
            )

            df_dets = cardinal.df_families(ref_time, families)

            print(df_dets)

            cardinal.inputs_dets_to_infrapy_json(df_dets, array, channel, st[0].stats.starttime.timestamp, st, data_filepath, sitedf)

            # Plot results with families
            cardinal.plot_sliding_window_multifreq(
                st, f_bands, T, B, V, S, spec_subplot=spec_subplot, scal_subplot=scal_subplot, clim_vtr=clim_vtr, clim_baz=clim_baz, bandpass=bandpass, normalize=normalize,
                GT_baz=GT_baz, delay_times=delay_times, amp_units=amp_units, pixels_in_families=pixels_in_families, ix=ix, title=title + "- Aggregator"
            )

            if plot_plots:
                plt.show()
    finally:
        client.close()
    

if __name__ == "__main__":
    main()