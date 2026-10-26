"""Preprocessing pipeline for fNIRS signals using MNE-NIRS."""

import mne
import mne_nirs


def preprocess_fnirs_data(
    raw: mne.io.Raw,
    l_freq: float = 0.01,
    h_freq: float = 0.2,
    apply_tddr: bool = True,
) -> mne.io.Raw:
    """Standard preprocessing pipeline for fNIRS data.

    1. Convert raw intensity to Optical Density (OD).
    2. Apply Motion Artifact Correction (TDDR).
    3. Convert OD to relative Hemoglobin Concentrations (HbO/HbR).
    4. Apply zero-phase Bandpass Filter to isolate hemodynamic response.

    Args:
        raw: MNE Raw object containing raw intensity channels.
        l_freq: High-pass filter cutoff frequency (Hz).
        h_freq: Low-pass filter cutoff frequency (Hz).
        apply_tddr: Whether to apply Temporal Derivative Distribution Repair.

    Returns:
        raw_haemo: Processed MNE Raw object containing HbO/HbR concentrations.
    """
    # 1. Convert intensity to optical density
    raw_od = mne.preprocessing.nirs.optical_density(raw)

    # 2. Motion artifact correction (TDDR)
    if apply_tddr:
        raw_od = mne_nirs.signal_processing.tddr(raw_od)

    # 3. Convert optical density to hemoglobin concentrations (MBLL)
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)

    # 4. Bandpass filter to remove cardiac (~1Hz) and low-frequency drift
    raw_haemo = raw_haemo.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="iir",
        iir_params=dict(order=4, ftype="butter"),
        verbose=False,
    )

    return raw_haemo


if __name__ == "__main__":
    print("fNIRS Preprocessing module successfully initialized.")
