import numpy as np
from scipy.interpolate import CubicSpline

import pycbc
from pycbc import waveform
from scipy.interpolate import CubicSpline

from pycbc.waveform import utils as wfutils
from pycbc import pnutils

# =========================================================
# Helper: smooth taper function
# =========================================================

def make_tapered_spline(x_nodes, y_nodes, taper_frac=0.05):
    x = np.asarray(x_nodes, dtype=float)
    y = np.asarray(y_nodes, dtype=float)
    x0, xN = x[0], x[-1]
    span = xN - x0
    if span <= 0:
        return lambda freqs: np.zeros_like(np.asarray(freqs, dtype=float))

    eps = taper_frac * span
    x_ext = np.concatenate(([x0 - eps], x, [xN + eps]))
    y_ext = np.concatenate(([0.0], y, [0.0]))

    cs = CubicSpline(x_ext, y_ext, bc_type=((1, 0.0), (1, 0.0)), extrapolate=True)

    def taper_window(f):
        f = np.asarray(f, dtype=float)
        w = np.ones_like(f)
        left = (f >= x0 - eps) & (f < x0)
        if np.any(left):
            u = (f[left] - (x0 - eps)) / eps
            w[left] = 0.5 * (1 - np.cos(np.pi * u))
        right = (f > xN) & (f <= xN + eps)
        if np.any(right):
            u = ((xN + eps) - f[right]) / eps
            w[right] = 0.5 * (1 - np.cos(np.pi * u))
        w[f < x0 - eps] = 0.0
        w[f > xN + eps] = 0.0
        return w

    def callable_spline(freqs):
        arr = np.asarray(freqs, dtype=float)
        scalar = False
        if arr.ndim == 0:
            arr = arr[None]
            scalar = True
        vals = cs(arr) * taper_window(arr)
        if scalar:
            return float(vals[0])
        return vals

    return callable_spline



class WFModifierFD(object):
    """
    A class for waveform modification for PyCBC waveform plugin.

    Parameters
    ----------
    config : dict
        The dictionary of parameters passed from the PyCBC waveform generator.

    Notes
    -----
    This class evaluates all branching (if/else logic) *once* during initialization.
    The PE sampler then only calls the `apply()` method, which is fast.
    """

    def __init__(self, config):
        self.cfg = config

        # Validate error type once
        err = config["error_in_phase"]
        if err not in ("relative", "absolute"):
            raise ValueError("error_in_phase must be 'relative' or 'absolute'")
        self.error_in_phase = err

        self.two_pol = config.get("two_polarizations", False)

        # Select modification type
        modtype = config["modification_type"]

        if modtype == "cubic_spline":
            self._init_cubic_spline()

        elif modtype == "constant_shift":
            self._init_constant_shift()

        elif modtype == "cubic_spline_nodes":
            self._init_cubic_spline_nodes()

        else:
            raise TypeError(f"Unsupported modification_type '{modtype}'")

    # ---------------------------------------------------------
    # Initialization blocks for each modification model
    # ---------------------------------------------------------

    def _init_constant_shift(self):
        cfg = self.cfg
        self.modtype = "constant_shift"

        if self.two_pol:
            self.delta_amplitude_plus = lambda f: cfg["delta_amplitude_plus"]
            self.delta_phase_plus = lambda f: cfg["delta_phase_plus"]
            self.delta_amplitude_cross = lambda f: cfg["delta_amplitude_cross"]
            self.delta_phase_cross = lambda f: cfg["delta_phase_cross"]
        else:
            self.delta_amplitude_plus = lambda f: cfg["delta_amplitude"]
            self.delta_phase_plus = lambda f: cfg["delta_phase"]
            self.delta_amplitude_cross = lambda f: cfg["delta_amplitude"]
            self.delta_phase_cross = lambda f: cfg["delta_phase"]
            
    def _init_cubic_spline(self):
        cfg = self.cfg
        self.modtype = "cubic_spline"

        wf_nodal_points = cfg["nodal_points"]

        if self.two_pol:
            da_plus = cfg["delta_amplitude_plus"]
            dp_plus = cfg["delta_phase_plus"]
            self.delta_amplitude_plus = make_tapered_spline(wf_nodal_points, da_plus)
            self.delta_phase_plus = make_tapered_spline(wf_nodal_points, dp_plus)
            
            da_cross = cfg["delta_amplitude_cross"]
            dp_cross = cfg["delta_phase_cross"]
            self.delta_amplitude_cross = make_tapered_spline(wf_nodal_points, da_cross)
            self.delta_phase_cross = make_tapered_spline(wf_nodal_points, dp_cross)

        else:
            da = cfg["delta_amplitude"]
            dp = cfg["delta_phase"]

            self.delta_amplitude_plus = make_tapered_spline(wf_nodal_points, da)
            self.delta_phase_plus = make_tapered_spline(wf_nodal_points, dp)
            self.delta_amplitude_cross = make_tapered_spline(wf_nodal_points, da)
            self.delta_phase_cross = make_tapered_spline(wf_nodal_points, dp)

    def _init_cubic_spline_nodes(self):
        cfg = self.cfg
        self.modtype = "cubic_spline_nodes"

        f_lo = (cfg.get("f_lower_wferror", cfg["f_lower"]))
        f_hi = cfg["f_high_wferror"]
        
        n = int(cfg["n_nodes_wferror"])

        # Log-spaced nodes
        wf_nodal_points = np.logspace(np.log10(f_lo), np.log10(f_hi), n)

        if self.two_pol:
            da_plus = np.array([cfg[f"wferror_amplitude_plus_{i}"] for i in range(n)])
            dp_plus = np.array([cfg[f"wferror_phase_plus_{i}"] for i in range(n)])
            self.delta_amplitude_plus = make_tapered_spline(wf_nodal_points, da_plus)
            self.delta_phase_plus = make_tapered_spline(wf_nodal_points, dp_plus)
            
            da_cross = np.array([cfg[f"wferror_amplitude_cross_{i}"] for i in range(n)])
            dp_cross = np.array([cfg[f"wferror_phase_cross_{i}"] for i in range(n)])
            self.delta_amplitude_cross = make_tapered_spline(wf_nodal_points, da_cross)
            self.delta_phase_cross = make_tapered_spline(wf_nodal_points, dp_cross)

        else:
            da = np.array([cfg[f"wferror_amplitude_{i}"] for i in range(n)])
            dp = np.array([cfg[f"wferror_phase_{i}"] for i in range(n)])

            self.delta_amplitude_plus = make_tapered_spline(wf_nodal_points, da)
            self.delta_phase_plus = make_tapered_spline(wf_nodal_points, dp)

            self.delta_amplitude_cross = make_tapered_spline(wf_nodal_points, da)
            self.delta_phase_cross = make_tapered_spline(wf_nodal_points, dp)

    # ---------------------------------------------------------
    # Core method: Apply the modification to hp, hc
    # ---------------------------------------------------------
    def apply(self, hp, hc):
        """
        Apply the amplitude-phase modifications.

        Parameters
        ----------
        hp, hc : FrequencySeries
            Base waveform polarizations.

        Returns
        -------
        (hp, hc) : tuple of FrequencySeries
            Modified waveform polarizations.
        """

        freqs_p = hp.sample_frequencies
        freqs_c = hc.sample_frequencies

        # -----------------------------------------
        # Amplitude modification
        # -----------------------------------------
        da_p = self.delta_amplitude_plus(freqs_p)
        da_c = self.delta_amplitude_cross(freqs_c)
        amp_plus = waveform.amplitude_from_frequencyseries(hp) * (1 + da_p)
        amp_cross = waveform.amplitude_from_frequencyseries(hc) * (1 + da_c)

        # -----------------------------------------
        # Phase modification
        # -----------------------------------------
        ph_base_plus = waveform.phase_from_frequencyseries(hp, remove_start_phase=False)
        ph_base_cross = waveform.phase_from_frequencyseries(hc, remove_start_phase=False)

        dphi_p = self.delta_phase_plus(freqs_p)
        dphi_c = self.delta_phase_cross(freqs_c)

        if self.error_in_phase == "relative":
            ph_plus = ph_base_plus * (1 + dphi_p)
            ph_cross = ph_base_cross * (1 + dphi_c)
        elif self.error_in_phase == "absolute":
            ph_plus = ph_base_plus + dphi_p
            ph_cross = ph_base_cross + dphi_c
        else:
            raise ValueError('Only two type of errors in phase are supported')

        # -----------------------------------------
        # Convert back to complex frequency series
        # -----------------------------------------
        hp.data = amp_plus * np.exp(1j * ph_plus)
        hc.data = amp_cross * np.exp(1j * ph_cross)

        return hp, hc



# ================================================================
# Top-level function called by PyCBC
# ================================================================
def amplitude_phase_modification_fd(**kwds):
    """
    PyCBC entry point for the waveform plugin.

    This now acts as a thin wrapper that:
    1. Builds the baseline waveform
    2. Constructs a WFErrorModifierFD (freezes logic)
    3. Applies the modification
    """

    # Build baseline waveform once
    wf_params = kwds.copy()
    wf_params["approximant"] = kwds["baseline_approximant"]

    if 'NRSur' in wf_params['approximant']:
        hp, hc = get_fd_waveform_from_td(wf_params)
    else:
        hp, hc = waveform.get_fd_waveform(wf_params)

    # Modifier must be rebuilt each call (for PE)
    modifier = WFModifierFD(kwds)

    # Apply correction
    return modifier.apply(hp, hc)


def amplitude_phase_modification_both_polarization_fd(**kwds):
    """
    PyCBC entry point for the waveform plugin.

    This now acts as a thin wrapper that:
    1. Builds the baseline waveform
    2. Constructs a WFErrorModifierFD (freezes logic)
    3. Applies the modification
    """

    # Build baseline waveform once
    wf_params = kwds.copy()
    wf_params["approximant"] = kwds["baseline_approximant"]

    if 'NRSur' in wf_params['approximant']:
        hp, hc = get_fd_waveform_from_td(wf_params)
    else:
        hp, hc = waveform.get_fd_waveform(wf_params)

    kwds['two_polarizations']=True
    # Modifier must be rebuilt each call
    modifier = WFModifierFD(kwds)

    # Apply correction
    return modifier.apply(hp, hc)



#------------------------------------------
# This is a hack for NRSur model support
#-----------------------------------------

def get_fd_waveform_from_td(params):
    """
    Generate a frequency-domain waveform by first constructing
    a time-domain waveform and then Fourier transforming it.
    Based on same function in PyCBC

    The function automatically pads and tapers the waveform
    to avoid spectral leakage and wraparound artifacts.

    Parameters
    ----------
    params : dict
        Dictionary of waveform parameters. Must include:
        - 'approximant'
        - 'f_lower'
        - 'delta_f'

    Returns
    -------
    hp : pycbc.types.FrequencySeries
        Plus polarization frequency-domain waveform.
    hc : pycbc.types.FrequencySeries
        Cross polarization frequency-domain waveform.
    """
    # ------------------------------------------------------------------
    # Copy parameters defensively
    # ------------------------------------------------------------------
    base_params = params.copy()
    work_params = params.copy()

    # ------------------------------------------------------------------
    # Determine waveform duration
    # ------------------------------------------------------------------
    duration = waveform.get_waveform_filter_length_in_time(**work_params)

    # Fallback approximant if duration is unavailable
    if duration is None:
        work_params["approximant"] = "IMRPhenomPv2"
        duration = waveform.get_waveform_filter_length_in_time(**work_params)

    if duration is None:
        raise RuntimeError("Unable to determine waveform duration.")

    full_duration = duration

    # Ensure enough padding by lowering f_lower if necessary
    while full_duration < 1.5 * duration:
        work_params["f_lower"] -= 1.0
        if work_params["f_lower"] <= 0:
            raise ValueError("f_lower became non-positive while extending duration.")
        full_duration = waveform.get_waveform_filter_length_in_time(**work_params)

    # ------------------------------------------------------------------
    # Reference frequency
    # ------------------------------------------------------------------
    if "f_ref" not in work_params:
        work_params["f_ref"] = work_params["f_lower"]

    # ------------------------------------------------------------------
    # Determine time step
    # ------------------------------------------------------------------

    f_end = waveform.get_waveform_end_frequency(**work_params)

    if f_end is None or not np.isfinite(f_end):
        delta_t = 1.0 / 2048.0
    else:
        delta_t = 0.5 / pnutils.nearest_larger_binary_number(f_end)
    #try:
    #    f_end = waveform.get_waveform_end_frequency(**work_params)
    #    delta_t = 0.5 / pnutils.nearest_larger_binary_number(f_end)
    #except (AttributeError, ValueError):
    #    delta_t = 1.0 / 2048.0

    work_params["delta_t"] = delta_t

    # Restore requested approximant
    work_params["approximant"] = base_params["approximant"]

    # ------------------------------------------------------------------
    # Generate time-domain waveform
    # ------------------------------------------------------------------
    hp_td, hc_td = waveform.get_td_waveform(**work_params)

    # ------------------------------------------------------------------
    # Resize time series to match requested frequency spacing
    # ------------------------------------------------------------------
    tsamples = int(1.0 / base_params["delta_f"] / delta_t)

    if tsamples < len(hp_td):
        raise ValueError(
            f"delta_f = {base_params['delta_f']} is too small to "
            f"generate the {base_params['approximant']} waveform."
        )

    hp_td.resize(tsamples)
    hc_td.resize(tsamples)

    # ------------------------------------------------------------------
    # Taper the beginning of the waveform
    # ------------------------------------------------------------------
    taper_window = 0.9 * (full_duration - duration)

    hp_td = wfutils.td_taper(
        hp_td,
        hp_td.start_time,
        hp_td.start_time + taper_window,
    )
    hc_td = wfutils.td_taper(
        hc_td,
        hc_td.start_time,
        hc_td.start_time + taper_window,
    )

    # ------------------------------------------------------------------
    # Convert to frequency domain and avoid wraparound
    # ------------------------------------------------------------------
    hp_fd = hp_td.to_frequencyseries().cyclic_time_shift(hp_td.start_time)
    hc_fd = hc_td.to_frequencyseries().cyclic_time_shift(hc_td.start_time)

    # Enforce lower frequency cutoff
    mask = hp_fd.sample_frequencies < base_params["f_lower"]
    hp_fd.data[mask] = 0.0
    hc_fd.data[mask] = 0.0

    return hp_fd, hc_fd

