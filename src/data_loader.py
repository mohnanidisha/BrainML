"""Data loader utility for loading fNIRS datasets using MNE-Python."""

from pathlib import Path
import mne
import mne_nirs


def load_fnirs_raw(file_path: str | Path, verbose: bool = False) -> mne.io.Raw:
    """Load a raw fNIRS file (.snirf, .fif, or dataset directory) into an MNE Raw object.

    Args:
        file_path: Path to the raw fNIRS data file.
        verbose: Print detailed MNE processing logs.

    Returns:
        raw: MNE Raw object containing raw intensity or optical density channels.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at: {file_path}")

    # Load based on extension
    if file_path.suffix == ".snirf":
        raw = mne.io.read_raw_snirf(file_path, preload=True, verbose=verbose)
    elif file_path.suffix in [".fif", ".gz"]:
        raw = mne.io.read_raw_fif(file_path, preload=True, verbose=verbose)
    else:
        raise ValueError(
            f"Unsupported file format '{file_path.suffix}'. Expected .snirf or .fif"
        )

    return raw


def get_dataset_summary(raw: mne.io.Raw) -> dict:
    """Extract baseline dataset specifications (sampling rate, duration, channels)."""
    return {
        "sampling_rate_hz": raw.info["sfreq"],
        "total_channels": len(raw.ch_names),
        "duration_seconds": raw.times[-1],
        "annotations_count": len(raw.annotations),
        "channel_types": list(set(raw.get_channel_types())),
    }


if __name__ == "__main__":
    print("fNIRS Data Loader module successfully initialized.")
