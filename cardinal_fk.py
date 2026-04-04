"""
Cardinal F-K Analysis Module
=============================

High-performance frequency-wavenumber (F-K) beamforming for seismic and 
infrasound array processing.

This module provides optimized implementations of F-K analysis methods,
replacing legacy ObsPy-based code with JIT-compiled NumPy/Numba implementations.

Functions
---------
fk_analysis : Single-window F-K analysis
sliding_window_fk : Optimized sliding-window F-K analysis with Numba JIT

Theory
------
F-K analysis estimates the slowness vector s = (sx, sy) by beamforming in 
the frequency domain. For each slowness grid point, we compute:

    P(s) = ∫ |1/N Σᵢ Uᵢ(f) exp(i 2π f τᵢ(s))|² df

where:
    - Uᵢ(f) = FFT of station i
    - τᵢ(s) = xᵢsₓ + yᵢsᵧ (steering delay)
    - N = number of stations

Semblance (normalized beam power):
    Sem(s) = P(s) / P_incoherent

References
----------
.. [1] Rost, S., & Thomas, C. (2002). Array seismology: Methods and 
       applications. Reviews of Geophysics, 40(3), 1008.
.. [2] Schweitzer, J., et al. (2012). Seismic Arrays. In New Manual of 
       Seismological Observatory Practice 2 (NMSOP-2).

Author: The Cardinal Team

Portions of this code were developed with the assistance of a large language model (LLM).
All content has been reviewed and validated by the authors.
"""

import numpy as np
from numba import jit
from scipy import signal as scipy_signal
import matplotlib.pyplot as plt

__all__ = [
    'fk_analysis', 'sliding_window_fk',
    'get_array_coordinates', 'compute_arf',
    'plot_arf', 'plot_fk'
]


def get_array_coordinates(st, return_centroid=False):
    """
    Extract array station coordinates from ObsPy Stream.
    
    Converts geographic coordinates (lat/lon) to local Cartesian coordinates
    (x, y) in kilometers using equirectangular projection centered on the
    array centroid.
    
    Coordinate System
    -----------------
    The equirectangular projection (also called "plate carrée") provides:
    
        x = R_⊕ cos(φ₀) (λ - λ₀)
        y = R_⊕ (φ - φ₀)
    
    where:
        R_⊕ = 6371 km (mean Earth radius)
        (φ₀, λ₀) = array centroid (lat, lon in radians)
        (φ, λ) = station coordinates (lat, lon in radians)
        x = East-West position (km)
        y = North-South position (km)
    
    This is accurate for small arrays (< 100 km aperture) where Earth
    curvature effects are negligible.
    
    Parameters
    ----------
    st : obspy.Stream
        Stream object containing traces with SAC headers or stats.
        Requires each trace to have latitude/longitude in one of:
          - tr.stats.sac.stla / tr.stats.sac.stlo (SAC format)
          - tr.stats.coordinates.latitude / tr.stats.coordinates.longitude
    return_centroid : bool, optional
        If True, also return centroid coordinates (default: False)
    
    Returns
    -------
    x_km : ndarray, shape (N_stations,)
        East-West positions in km relative to centroid
        Positive = East, Negative = West
    y_km : ndarray, shape (N_stations,)
        North-South positions in km relative to centroid
        Positive = North, Negative = South
    lat0 : float (only if return_centroid=True)
        Centroid latitude in degrees
    lon0 : float (only if return_centroid=True)
        Centroid longitude in degrees
    
    Examples
    --------
    >>> from obspy import read
    >>> st = read()  # Assumes SAC headers with coordinates
    >>> x_km, y_km = get_array_coordinates(st)
    >>> 
    >>> # With centroid
    >>> x_km, y_km, lat0, lon0 = get_array_coordinates(st, return_centroid=True)
    >>> print(f"Array centered at {lat0:.4f}°N, {lon0:.4f}°E")
    
    Notes
    -----
    - Assumes all traces have valid coordinate information
    - Coordinate origin (0, 0) is at array centroid
    - For large aperture arrays (> 100 km), consider using proper geodetic
      transformations (e.g., UTM projection)
    """
    R_EARTH = 6371.0  # km
    
    # Extract lat/lon from each trace
    lats = []
    lons = []
    for tr in st:
        # Try SAC header format first
        if hasattr(tr.stats, 'sac'):
            lats.append(tr.stats.sac.stla)
            lons.append(tr.stats.sac.stlo)
        # Try ObsPy coordinates format
        elif hasattr(tr.stats, 'coordinates'):
            lats.append(tr.stats.coordinates.latitude)
            lons.append(tr.stats.coordinates.longitude)
        else:
            raise ValueError(
                f"Trace {tr.id} missing coordinate information. "
                "Requires SAC headers (stla/stlo) or stats.coordinates"
            )
    
    lats = np.array(lats)
    lons = np.array(lons)
    
    # Compute array centroid
    lat0 = lats.mean()
    lon0 = lons.mean()
    
    # Equirectangular projection
    x_km = R_EARTH * np.cos(np.radians(lat0)) * np.radians(lons - lon0)
    y_km = R_EARTH * np.radians(lats - lat0)
    
    if return_centroid:
        return x_km, y_km, lat0, lon0
    else:
        return x_km, y_km


def compute_arf(x_km, y_km, freq, smax, ngrid):
    """
    Compute Array Response Function (ARF) for a given frequency.
    
    Theory
    ------
    For a plane wave with slowness s = (sx, sy), the array response is:
    
        A(s, f) = |1/N Σᵢ exp(-i 2π f rᵢ·s)|²
    
    where:
        rᵢ = (xᵢ, yᵢ) = position of station i
        N = number of stations
        f = frequency
    
    The ARF is normalized so that A(0, f) = 1 (peak response at zero slowness).
    
    Key Features:
        - Main lobe width: Angular resolution (narrower = better)
        - Side lobes: False detection sensitivity (lower = better)
        - -3 dB contour: Half-power beamwidth
    
    Parameters
    ----------
    x_km : array_like, shape (N_stations,)
        East-West sensor positions in km
    y_km : array_like, shape (N_stations,)
        North-South sensor positions in km
    freq : float
        Frequency in Hz at which to compute ARF
    smax : float
        Maximum slowness magnitude in s/km
        Defines slowness grid: [-smax, +smax] × [-smax, +smax]
    ngrid : int
        Number of grid points per slowness axis
        Total grid size: ngrid × ngrid
    
    Returns
    -------
    ARF : ndarray, shape (ngrid, ngrid)
        Array response function (power), normalized to peak = 1
    ux_grid : ndarray, shape (ngrid,)
        Slowness grid in x-direction (E-W) in s/km
    uy_grid : ndarray, shape (ngrid,)
        Slowness grid in y-direction (N-S) in s/km
    
    Examples
    --------
    >>> # Compute ARF at 1 Hz
    >>> ARF, ux, uy = compute_arf(x_km, y_km, freq=1.0, 
    ...                            smax=0.5, ngrid=101)
    >>> 
    >>> # Find -3 dB beamwidth
    >>> ARF_db = 10 * np.log10(ARF)
    >>> beamwidth_mask = ARF_db >= -3
    >>> print(f"-3 dB beamwidth covers {beamwidth_mask.sum()} grid points")
    
    See Also
    --------
    plot_arf : Convenient plotting function for ARF
    
    Notes
    -----
    - Higher frequency → narrower main lobe (better resolution)
    - Larger array aperture → narrower main lobe
    - Irregular array geometry → lower side lobes
    """
    N = len(x_km)
    
    # Create slowness grid
    ux_grid = np.linspace(-smax, smax, ngrid)
    uy_grid = np.linspace(-smax, smax, ngrid)
    UX, UY = np.meshgrid(ux_grid, uy_grid)
    
    # Position vectors (N, 2) in km
    pos = np.column_stack([x_km, y_km])
    
    # Compute ARF for all grid points
    ARF = np.zeros((ngrid, ngrid))
    
    for i in range(ngrid):
        for j in range(ngrid):
            ux = UX[i, j]
            uy = UY[i, j]
            u_vec = np.array([ux, uy])  # s/km
            
            # Phase delay: r · u (in seconds, since r is in km and u is in s/km)
            phase_delays = pos @ u_vec  # (N,) in seconds
            
            # Complex beam response: (1/N) * sum(exp(-i * 2*pi*f * r·u))
            beam = np.mean(np.exp(-1j * 2 * np.pi * freq * phase_delays))
            
            # Power (squared magnitude)
            ARF[i, j] = np.abs(beam)**2
    
    # Normalize to peak = 1
    ARF /= ARF.max()
    
    return ARF, ux_grid, uy_grid


def plot_arf(x_km, y_km, freq_low=0.5, freq_high=4.0, freq_broadband=None,
             smax=3.6, ngrid=151, figsize=None, vmin_db=-8, vmax_db=0,
             cmap='inferno', show_contour=True, contour_db=-3, 
             show_all_frequencies=False):
    """
    Plot Array Response Function.
    
    By default, shows only the broadband ARF (averaged over frequency band).
    Optionally can show three panels: ARF at low frequency, high frequency,
    and broadband.
    
    Parameters
    ----------
    x_km : array_like, shape (N_stations,)
        East-West sensor positions in km
    y_km : array_like, shape (N_stations,)
        North-South sensor positions in km
    freq_low : float, optional
        Low frequency in Hz (default: 0.5)
    freq_high : float, optional
        High frequency in Hz (default: 4.0)
    freq_broadband : array_like or None, optional
        Frequencies to average for broadband ARF in Hz.
        If None, uses 15 frequencies linearly spaced between freq_low and freq_high
    smax : float, optional
        Maximum slowness in s/km (default: 3.6)
    ngrid : int, optional
        Grid resolution per axis (default: 151)
    figsize : tuple or None, optional
        Figure size (width, height) in inches. 
        If None, uses (7, 6) for single panel or (16, 5) for three panels
    vmin_db : float, optional
        Minimum power for colormap in dB (default: -8)
    vmax_db : float, optional
        Maximum power for colormap in dB (default: 0)
    cmap : str, optional
        Matplotlib colormap name (default: 'inferno')
    show_contour : bool, optional
        Whether to show contour lines (default: True)
    contour_db : float, optional
        Contour level in dB (default: -3 for half-power)
    show_all_frequencies : bool, optional
        If True, show three panels (low freq, high freq, broadband).
        If False, show only broadband panel (default: False)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    axes : matplotlib.axes.Axes or array of Axes
        Single axis (if show_all_frequencies=False) or 
        array of three axis objects [ax_low, ax_high, ax_broadband]
    
    Examples
    --------
    >>> # Show only broadband ARF (default)
    >>> fig, ax = plot_arf(x_km, y_km, freq_low=0.2, freq_high=4.0)
    >>> 
    >>> # Show all three frequency panels
    >>> fig, axes = plot_arf(x_km, y_km, show_all_frequencies=True)
    >>> plt.savefig('arf_analysis.png', dpi=300)
    """
    # Compute broadband ARF (average over frequency band)
    if freq_broadband is None:
        freq_broadband = np.linspace(freq_low, freq_high, 15)
    
    ARF_broadband = np.zeros_like(compute_arf(x_km, y_km, freq_broadband[0], smax, ngrid)[0])
    for f in freq_broadband:
        ARF_f, ux_grid, uy_grid = compute_arf(x_km, y_km, f, smax, ngrid)
        ARF_broadband += ARF_f
    ARF_broadband /= ARF_broadband.max()
    
    # Convert to dB scale
    ARF_broadband_db = 10 * np.log10(ARF_broadband + 1e-10)
    
    # Determine figure layout
    if show_all_frequencies:
        # Compute ARF at low and high frequencies for three-panel display
        ARF_low, ux_grid, uy_grid = compute_arf(x_km, y_km, freq_low, smax, ngrid)
        ARF_high, _, _ = compute_arf(x_km, y_km, freq_high, smax, ngrid)
        ARF_low_db = 10 * np.log10(ARF_low + 1e-10)
        ARF_high_db = 10 * np.log10(ARF_high + 1e-10)
        
        if figsize is None:
            figsize = (16, 5)
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        arf_list = [ARF_low_db, ARF_high_db, ARF_broadband_db]
        titles = [
            f'ARF at {freq_low} Hz',
            f'ARF at {freq_high} Hz',
            f'Broadband ARF ({freq_low}–{freq_high} Hz)'
        ]
    else:
        # Single panel: broadband only
        if figsize is None:
            figsize = (7, 6)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        axes = [ax]  # Make it iterable for the loop
        
        arf_list = [ARF_broadband_db]
        titles = [f'Broadband ARF ({freq_low}–{freq_high} Hz)']
    
    # Plot ARF panel(s)
    for ax, ARF_db, title in zip(axes, arf_list, titles):
        
        # Plot ARF
        im = ax.pcolormesh(ux_grid, uy_grid, ARF_db, cmap=cmap,
                          vmin=vmin_db, vmax=vmax_db, shading='auto')
        
        # Add contour if requested (only around main peak, not sidelobes)
        if show_contour:
            # Find peak location
            peak_idx = np.unravel_index(np.argmax(ARF_db), ARF_db.shape)
            peak_row, peak_col = peak_idx
            
            # Create a mask that only includes the connected region around the peak
            from scipy.ndimage import label, generate_binary_structure
            
            # Binary mask: 1 where ARF_db >= contour_db
            binary_mask = (ARF_db >= contour_db).astype(int)
            
            # Use 8-connectivity (includes diagonals) to properly connect the main lobe
            structure = generate_binary_structure(2, 2)  # 2D, connectivity=2 means 8-connected
            
            # Label connected components with 8-connectivity
            labeled_array, num_features = label(binary_mask, structure=structure)
            
            # Get the label of the peak region
            peak_label = labeled_array[peak_row, peak_col]
            
            # Create mask for only the peak region
            peak_region_mask = (labeled_array == peak_label)
            
            # Apply mask: set non-peak regions to below contour level
            ARF_db_masked = ARF_db.copy()
            ARF_db_masked[~peak_region_mask] = contour_db - 10  # Set well below contour
            
            # Now draw contour only around peak
            ax.contour(ux_grid, uy_grid, ARF_db_masked, levels=[contour_db],
                      colors='white', linewidths=[1.5], linestyles=['--'], 
                      alpha=0.8)
        
        ax.set_xlabel('Slowness $s_x$ (E–W) [s/km]', fontsize=10)
        ax.set_ylabel('Slowness $s_y$ (N–S) [s/km]', fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_aspect('equal')
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, linewidth=0.5)
        
        plt.colorbar(im, ax=ax, label='Power [dB]', pad=0.02)
    
    plt.tight_layout()
    
    # Return single axis or array of axes depending on mode
    if show_all_frequencies:
        return fig, axes
    else:
        return fig, axes[0]  # Return single axis, not array


def plot_fk(sx_vec, sy_vec, power_grid, semblance_grid, fmin, fmax,
            figsize=(14, 6), power_vmin_db=-10, power_vmax_db=0,
            semblance_vmin=0, semblance_vmax=1,
            power_cmap='inferno', semblance_cmap='hot_r',
            show_contour=True, contour_db=-3, show_peak=True):
    """
    Plot F-K analysis results (power and semblance).
    
    Creates a two-panel figure showing:
      - Left: F-K power spectrum in dB
      - Right: Semblance (normalized beam power)
    
    Parameters
    ----------
    sx_vec : ndarray, shape (ngrid,)
        Slowness grid in x-direction (E-W) in s/km
    sy_vec : ndarray, shape (ngrid,)
        Slowness grid in y-direction (N-S) in s/km
    power_grid : ndarray, shape (ngrid, ngrid)
        F-K power at each slowness grid point
    semblance_grid : ndarray, shape (ngrid, ngrid)
        Semblance at each grid point
    fmin : float
        Minimum frequency used in analysis (Hz)
    fmax : float
        Maximum frequency used in analysis (Hz)
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (14, 6))
    power_vmin_db : float, optional
        Minimum for power colormap in dB (default: -10)
    power_vmax_db : float, optional
        Maximum for power colormap in dB (default: 0)
    semblance_vmin : float, optional
        Minimum for semblance colormap (default: 0)
    semblance_vmax : float, optional
        Maximum for semblance colormap (default: 1)
    power_cmap : str, optional
        Colormap for power plot (default: 'inferno')
    semblance_cmap : str, optional
        Colormap for semblance plot (default: 'hot_r')
    show_contour : bool, optional
        Whether to show -3 dB contour on power plot (default: True)
    contour_db : float, optional
        Contour level in dB (default: -3)
    show_peak : bool, optional
        Whether to mark peak with crosshair and label (default: True)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    axes : array of matplotlib.axes.Axes
        Array of two axis objects [ax_power, ax_semblance]
    results : dict
        Dictionary containing:
          - 'sx_peak': Peak slowness in x-direction (s/km)
          - 'sy_peak': Peak slowness in y-direction (s/km)
          - 'slowness': Total slowness magnitude (s/km)
          - 'velocity': Apparent velocity (km/s)
          - 'backazimuth': Backazimuth in degrees [0, 360)
          - 'semblance_peak': Peak semblance value
    
    Examples
    --------
    >>> sx, sy, power, semb = fk_analysis(data, x_km, y_km, fs, 0.5, 4.0)
    >>> fig, axes, results = plot_fk(sx, sy, power, semb, 0.5, 4.0)
    >>> print(f"Backazimuth: {results['backazimuth']:.1f}°")
    >>> print(f"Velocity: {results['velocity']:.2f} km/s")
    """
    # Find peak
    peak_idx = np.unravel_index(np.argmax(power_grid), power_grid.shape)
    sx_peak = sx_vec[peak_idx[1]]
    sy_peak = sy_vec[peak_idx[0]]
    slowness = np.sqrt(sx_peak**2 + sy_peak**2)
    velocity = 1.0 / slowness if slowness > 1e-10 else np.inf
    backazimuth = (np.degrees(np.arctan2(sx_peak, sy_peak)) + 180) % 360
    semblance_peak = semblance_grid[peak_idx]
    
    # Convert power to dB
    power_db = 10 * np.log10(power_grid / power_grid.max() + 1e-30)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot power
    ax = axes[0]
    im1 = ax.pcolormesh(sx_vec, sy_vec, power_db, cmap=power_cmap,
                       vmin=power_vmin_db, vmax=power_vmax_db, shading='auto')
    
    # Add contour if requested (only around main peak, not sidelobes)
    if show_contour:
        from scipy.ndimage import label, generate_binary_structure
        
        # Binary mask: 1 where power_db >= contour_db
        binary_mask = (power_db >= contour_db).astype(int)
        
        # Use 8-connectivity (includes diagonals) to properly connect the main lobe
        structure = generate_binary_structure(2, 2)  # 2D, connectivity=2 means 8-connected
        
        # Label connected components with 8-connectivity
        labeled_array, num_features = label(binary_mask, structure=structure)
        
        # Get the label of the peak region (peak_idx was calculated earlier)
        peak_label = labeled_array[peak_idx]
        
        # Create mask for only the peak region
        peak_region_mask = (labeled_array == peak_label)
        
        # Apply mask: set non-peak regions to below contour level
        power_db_masked = power_db.copy()
        power_db_masked[~peak_region_mask] = contour_db - 10  # Set well below contour
        
        # Now draw contour only around peak
        ax.contour(sx_vec, sy_vec, power_db_masked, levels=[contour_db],
                  colors='white', linewidths=[1.5], linestyles=['--'], alpha=0.8)
    
    if show_peak:
        ax.plot(sx_peak, sy_peak, 'w+', ms=15, mew=2.5,
               label=f'Peak: {backazimuth:.1f}°, {velocity:.2f} km/s')
        ax.legend(loc='upper right', fontsize=9)
    
    ax.set_xlabel('Slowness $s_x$ (E–W) [s/km]', fontsize=11)
    ax.set_ylabel('Slowness $s_y$ (N–S) [s/km]', fontsize=11)
    ax.set_title(f'F-K Power [{fmin}–{fmax} Hz]', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    plt.colorbar(im1, ax=ax, label='Relative Power [dB]', pad=0.02)
    
    # Plot semblance
    ax = axes[1]
    im2 = ax.pcolormesh(sx_vec, sy_vec, semblance_grid, cmap=semblance_cmap,
                       vmin=semblance_vmin, vmax=semblance_vmax, shading='auto')
    
    if show_peak:
        ax.plot(sx_peak, sy_peak, 'c+', ms=15, mew=2.5,
               label=f'Peak: {semblance_peak:.3f}')
        ax.legend(loc='upper right', fontsize=9)
    
    ax.set_xlabel('Slowness $s_x$ (E–W) [s/km]', fontsize=11)
    ax.set_ylabel('Slowness $s_y$ (N–S) [s/km]', fontsize=11)
    ax.set_title(f'Semblance [{fmin}–{fmax} Hz]', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    plt.colorbar(im2, ax=ax, label='Semblance', pad=0.02)
    
    plt.tight_layout()
    
    # Prepare results dictionary
    results = {
        'sx_peak': sx_peak,
        'sy_peak': sy_peak,
        'slowness': slowness,
        'velocity': velocity,
        'backazimuth': backazimuth,
        'semblance_peak': semblance_peak
    }
    
    return fig, axes, results


def fk_analysis(data, x_km, y_km, fs, fmin, fmax, smax=0.4, ngrid=51):
    """
    Frequency-wavenumber (F-K) analysis for a single time window.
    
    Performs beamforming over a slowness grid to estimate the slowness vector
    that maximizes coherent beam power in the specified frequency band.
    
    Theory
    ------
    For each slowness s = (sx, sy), compute beam power:
    
        P(s) = ∫[fmin,fmax] |1/N Σᵢ U_i(f) exp(i 2π f τᵢ(s))|² df
    
    where τᵢ(s) = xᵢsₓ + yᵢsᵧ is the steering delay for station i.
    
    Semblance (normalized beam power):
        Sem(s) = P(s) / P_incoherent
        
    where P_incoherent = (1/N) Σᵢ ∫|Uᵢ(f)|² df
    
    Parameters
    ----------
    data : ndarray, shape (N_stations, N_samples)
        Waveform data from N_stations sensors
    x_km : array_like, shape (N_stations,)
        East-West sensor positions in km relative to array centroid
    y_km : array_like, shape (N_stations,)
        North-South sensor positions in km relative to array centroid
    fs : float
        Sampling rate in Hz
    fmin : float
        Minimum frequency for analysis in Hz
    fmax : float
        Maximum frequency for analysis in Hz
    smax : float, optional
        Maximum slowness magnitude in s/km (default: 0.4)
        Defines slowness grid as [-smax, +smax] in both dimensions
    ngrid : int, optional
        Number of grid points per slowness axis (default: 51)
        Total grid size will be ngrid × ngrid
    
    Returns
    -------
    sx_vec : ndarray, shape (ngrid,)
        Slowness grid points in x-direction (E-W) in s/km
    sy_vec : ndarray, shape (ngrid,)
        Slowness grid points in y-direction (N-S) in s/km
    power_grid : ndarray, shape (ngrid, ngrid)
        Beam power at each slowness grid point
    semblance_grid : ndarray, shape (ngrid, ngrid)
        Semblance (normalized beam power) at each grid point, range [0, 1]
    
    Notes
    -----
    - Data is automatically detrended (linear trend removal)
    - No tapering is applied; apply window before calling if needed
    - Backazimuth from peak: θ = atan2(sx_peak, sy_peak) + 180°
    - Apparent velocity from peak: v = 1 / |s_peak|
    
    Examples
    --------
    >>> # Analyze single time window
    >>> sx, sy, power, semb = fk_analysis(data, x_km, y_km, fs=20, 
    ...                                     fmin=0.5, fmax=4.0)
    >>> peak_idx = np.unravel_index(np.argmax(power), power.shape)
    >>> sx_peak = sx[peak_idx[1]]
    >>> sy_peak = sy[peak_idx[0]]
    >>> baz = (np.degrees(np.arctan2(sx_peak, sy_peak)) + 180) % 360
    >>> velocity = 1.0 / np.sqrt(sx_peak**2 + sy_peak**2)
    
    See Also
    --------
    sliding_window_fk : Optimized sliding window F-K analysis
    """
    N_sta, N_samp = data.shape
    
    # Detrend each trace (remove linear trend)
    data = data.copy()
    for i in range(N_sta):
        x = np.arange(N_samp)
        coeffs = np.polyfit(x, data[i], 1)
        data[i] -= np.polyval(coeffs, x)
    
    # FFT
    NFFT = N_samp
    freqs = np.fft.rfftfreq(NFFT, d=1.0/fs)
    U = np.fft.rfft(data, n=NFFT, axis=1)  # (N_sta, N_freq)
    
    # Frequency mask
    f_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs_sel = freqs[f_mask]
    U_sel = U[:, f_mask]
    df = freqs[1] - freqs[0]
    
    # Incoherent power (denominator for semblance)
    incoherent_power = np.sum(np.abs(U_sel)**2) * df / N_sta
    
    # Slowness grid
    sx_vec = np.linspace(-smax, smax, ngrid)
    sy_vec = np.linspace(-smax, smax, ngrid)
    SX, SY = np.meshgrid(sx_vec, sy_vec)
    
    # Flatten slowness grid for vectorization
    sx_flat = SX.ravel()
    sy_flat = SY.ravel()
    
    # Compute time delays: delay[sta, grid_point] = sx*x + sy*y
    x_km = np.asarray(x_km)
    y_km = np.asarray(y_km)
    delay = np.outer(x_km, sx_flat) + np.outer(y_km, sy_flat)  # (N_sta, ngrid²)
    
    beam_power_flat = np.zeros(ngrid * ngrid)
    
    # Loop over frequencies and beam
    for k, f in enumerate(freqs_sel):
        # Steering phase for all stations and grid points
        steer = np.exp(1j * 2 * np.pi * f * delay)  # (N_sta, ngrid²)
        # Beam: sum over stations
        beam = (U_sel[:, k, np.newaxis] * steer).sum(axis=0) / N_sta  # (ngrid²,)
        beam_power_flat += np.abs(beam)**2 * df
    
    power_grid = beam_power_flat.reshape(ngrid, ngrid)
    semblance_grid = power_grid / incoherent_power
    
    return sx_vec, sy_vec, power_grid, semblance_grid


@jit(nopython=True, cache=True, fastmath=True)
def _beamform_fk_numba(U_sel, steering_vectors, df, n_stations):
    """
    JIT-compiled beamforming using pre-computed steering vectors.
    
    This is an internal helper function optimized with Numba for speed.
    On first call, Numba compiles this to native machine code and caches
    it to disk. Subsequent calls are ~10-100x faster than pure Python.
    
    Parameters
    ----------
    U_sel : ndarray, shape (n_stations, n_freqs), dtype=complex128
        Frequency-domain data for selected frequency band
    steering_vectors : ndarray, shape (n_freqs, n_stations, ngrid²), dtype=complex128
        Pre-computed steering phase factors: exp(i 2π f τ)
    df : float
        Frequency resolution (Hz)
    n_stations : int
        Number of stations
    
    Returns
    -------
    beam_power_flat : ndarray, shape (ngrid²,)
        Beam power at each slowness grid point
        
    Notes
    -----
    Numba compilation happens once per function signature. The compiled
    code is cached to __pycache__/ and reloaded on subsequent runs.
    Recompilation only occurs if:
      - Function code changes
      - Input types change
      - Cache is manually deleted
    """
    n_freqs = U_sel.shape[1]
    ngrid_sq = steering_vectors.shape[2]
    beam_power_flat = np.zeros(ngrid_sq)
    
    for grid_idx in range(ngrid_sq):
        for k in range(n_freqs):
            # Accumulate beam for this grid point
            beam_real = 0.0
            beam_imag = 0.0
            
            for sta in range(n_stations):
                # Get pre-computed steering vector
                steer_real = steering_vectors[k, sta, grid_idx].real
                steer_imag = steering_vectors[k, sta, grid_idx].imag
                
                # Complex multiply: U * steer
                u_real = U_sel[sta, k].real
                u_imag = U_sel[sta, k].imag
                
                beam_real += u_real * steer_real - u_imag * steer_imag
                beam_imag += u_real * steer_imag + u_imag * steer_real
            
            # Normalize and accumulate power
            beam_real /= n_stations
            beam_imag /= n_stations
            beam_power_flat[grid_idx] += (beam_real**2 + beam_imag**2) * df
    
    return beam_power_flat


def sliding_window_fk(data, x_km, y_km, fs, fmin, fmax, 
                      window_length=10.0, overlap_percent=50.0,
                      smax=3.6, ngrid=101):
    """
    Optimized sliding-window F-K analysis with Numba JIT compilation.
    
    Applies F-K beamforming to overlapping time windows to track source
    parameters (backazimuth, velocity, semblance) over time.
    
    This implementation is optimized for speed using:
      - Pre-computed steering vectors (avoids repeated trig operations)
      - Numba JIT compilation for beamforming inner loop
      - Scipy's fast C-based detrending
      - Vectorized NumPy operations
    
    Typical speedup: 5-10x faster than naive implementation.
    
    Theory
    ------
    For each time window k centered at time t_k:
      1. Extract data window and apply taper
      2. Compute F-K power P_k(s) over slowness grid
      3. Find peak: s_k* = argmax P_k(s)
      4. Compute backazimuth: θ_k = atan2(sx*, sy*) + 180°
      5. Compute velocity: v_k = 1 / |s_k*|
      6. Record peak semblance: Sem_k
    
    Parameters
    ----------
    data : ndarray, shape (N_stations, N_samples)
        Waveform data from N_stations sensors
    x_km : array_like, shape (N_stations,)
        East-West sensor positions in km relative to array centroid
    y_km : array_like, shape (N_stations,)
        North-South sensor positions in km relative to array centroid
    fs : float
        Sampling rate in Hz
    fmin : float
        Minimum frequency for F-K analysis in Hz
    fmax : float
        Maximum frequency for F-K analysis in Hz
    window_length : float, optional
        Length of each time window in seconds (default: 10.0)
    overlap_percent : float, optional
        Overlap between consecutive windows as percentage (default: 50.0)
        Must be in range [0, 100)
    smax : float, optional
        Maximum slowness magnitude in s/km (default: 3.6)
        Defines slowness grid as [-smax, +smax] in both dimensions
    ngrid : int, optional
        Number of grid points per slowness axis (default: 101)
        Higher values give finer slowness resolution but slower computation
    
    Returns
    -------
    T : ndarray, shape (n_windows,)
        Center time of each window in seconds from start
    B : ndarray, shape (n_windows,)
        Backazimuth at each window in degrees [0, 360)
        Measured clockwise from North
    V : ndarray, shape (n_windows,)
        Apparent (phase) velocity at each window in km/s
    S : ndarray, shape (n_windows,)
        Peak semblance at each window, range [0, 1]
    
    Notes
    -----
    Performance:
      - First call is slower due to Numba JIT compilation (~1-2 seconds)
      - Compiled code is cached; subsequent runs are fast (~0.5 seconds typical)
      - Pre-computation of steering vectors is the key optimization
    
    Window processing:
      - Each window is linearly detrended before FFT
      - 10% Tukey (cosine) taper applied to reduce spectral leakage
      - Windows step by: Δt = window_length × (1 - overlap_percent/100)
    
    Examples
    --------
    >>> # Basic usage
    >>> T, B, V, S = sliding_window_fk(data, x_km, y_km, fs=20,
    ...                                 fmin=0.2, fmax=4.0,
    ...                                 window_length=20.0,
    ...                                 overlap_percent=50.0)
    >>> 
    >>> # Plot results
    >>> import matplotlib.pyplot as plt
    >>> fig, axes = plt.subplots(3, 1, sharex=True)
    >>> axes[0].scatter(T, B, c=S, cmap='hot_r')
    >>> axes[0].set_ylabel('Backazimuth [°]')
    >>> axes[1].scatter(T, V, c=S, cmap='hot_r')
    >>> axes[1].set_ylabel('Velocity [km/s]')
    >>> axes[2].plot(T, S, 'k-')
    >>> axes[2].set_ylabel('Semblance')
    >>> axes[2].set_xlabel('Time [s]')
    
    See Also
    --------
    fk_analysis : Single-window F-K analysis (non-optimized)
    """
    n_stations, n_samples = data.shape
    
    # Ensure arrays
    x_km = np.asarray(x_km, dtype=np.float64)
    y_km = np.asarray(y_km, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)
    
    # Window parameters
    window_samples = int(window_length * fs)
    window_step = window_length * (1.0 - overlap_percent / 100.0)
    step_samples = int(window_step * fs)
    
    # Calculate number of windows
    n_windows = int((n_samples - window_samples) / step_samples) + 1
    
    # Pre-allocate output arrays
    T = np.zeros(n_windows)
    B = np.zeros(n_windows)
    V = np.zeros(n_windows)
    S = np.zeros(n_windows)
    
    # Pre-compute slowness grid
    sx_vec = np.linspace(-smax, smax, ngrid)
    sy_vec = np.linspace(-smax, smax, ngrid)
    sx_flat = np.tile(sx_vec, ngrid)      # repeat entire sx_vec for each row
    sy_flat = np.repeat(sy_vec, ngrid)    # repeat each sy value ngrid times
    
    # Pre-compute time delays for all slowness grid points
    # delay[sta, grid_point] = sx*x + sy*y
    delay_grid = np.outer(x_km, sx_flat) + np.outer(y_km, sy_flat)  # (N_sta, ngrid²)
    
    # Pre-compute taper window (10% Tukey = cosine edges)
    taper_window = scipy_signal.windows.tukey(window_samples, alpha=0.1)
    
    # Pre-compute frequency parameters
    freqs = np.fft.rfftfreq(window_samples, d=1.0/fs)
    f_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs_sel = freqs[f_mask]
    n_freqs_sel = len(freqs_sel)
    df = freqs[1] - freqs[0]
    
    # Pre-compute steering vectors for all frequencies
    # This is the key optimization: compute exp(i 2π f τ) once for all windows
    # Shape: (n_freqs, n_stations, ngrid²)
    steering_vectors = np.zeros((n_freqs_sel, n_stations, ngrid * ngrid), 
                                dtype=np.complex128)
    for k, f in enumerate(freqs_sel):
        steering_vectors[k] = np.exp(1j * 2 * np.pi * f * delay_grid)
    
    # Sliding window loop
    for win_idx in range(n_windows):
        start_sample = win_idx * step_samples
        end_sample = start_sample + window_samples
        
        # Extract window data
        data_win = data[:, start_sample:end_sample]
        
        # Detrend (scipy C implementation is much faster than polyfit)
        data_win = scipy_signal.detrend(data_win, axis=1, type='linear')
        
        # Apply taper
        data_win = data_win * taper_window[np.newaxis, :]
        
        # FFT
        U = np.fft.rfft(data_win, n=window_samples, axis=1)  # (N_sta, N_freq)
        
        # Apply frequency mask
        U_sel = U[:, f_mask]
        
        # Incoherent power (for semblance normalization)
        incoherent_power = np.sum(np.abs(U_sel)**2) * df / n_stations
        
        # Beamforming (JIT-compiled, uses pre-computed steering vectors)
        beam_power_flat = _beamform_fk_numba(U_sel, steering_vectors, df, n_stations)
        
        # Reshape and compute semblance
        power_grid = beam_power_flat.reshape(ngrid, ngrid)
        semblance_grid = power_grid / incoherent_power
        
        # Find peak
        peak_idx = np.argmax(power_grid)
        peak_row, peak_col = np.unravel_index(peak_idx, (ngrid, ngrid))
        sx_peak = sx_vec[peak_col]
        sy_peak = sy_vec[peak_row]
        
        # Compute backazimuth and velocity
        slowness_peak = np.sqrt(sx_peak**2 + sy_peak**2)
        vapp_peak = 1.0 / slowness_peak if slowness_peak > 1e-10 else np.inf
        backazimuth_peak = (np.degrees(np.arctan2(sx_peak, sy_peak)) + 180) % 360
        
        # Store results
        T[win_idx] = (start_sample + window_samples / 2.0) / fs
        B[win_idx] = backazimuth_peak
        V[win_idx] = vapp_peak
        S[win_idx] = semblance_grid.flat[peak_idx]
    
    return T, B, V, S


# Convenience function for backward compatibility
def sliding_window_fk_fast(*args, **kwargs):
    """
    Alias for sliding_window_fk (for backward compatibility).
    
    See sliding_window_fk documentation for details.
    """
    return sliding_window_fk(*args, **kwargs)
