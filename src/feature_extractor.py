import numpy as np


def extract_features(signal):
    """
    Extract statistical and phase-based features
    from a complex baseband signal.
    """

    signal = np.asarray(signal, dtype=np.complex128)

    if signal.ndim != 1 or len(signal) < 2:
        raise ValueError("Signal must be a 1D array with at least 2 samples.")

    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal contains invalid values.")

    # 1. Calculate amplitude
    amplitude = np.abs(signal)

    # 2. Calculate signal power
    power = np.mean(amplitude ** 2)

    if power <= 0:
        raise ValueError("Signal power must be greater than zero.")

    # 3. Amplitude statistics
    amp_mean = np.mean(amplitude)
    amp_std = np.std(amplitude)

    if amp_std > 1e-12:
        centered = (amplitude - amp_mean) / amp_std
        amp_skewness = np.mean(centered ** 3)
        amp_kurtosis = np.mean(centered ** 4) - 3
    else:
        amp_skewness = 0.0
        amp_kurtosis = 0.0

    peak_amplitude = np.max(amplitude)

    papr = (peak_amplitude ** 2) / power

    # 4. Normalize signal to unit average power
    normalized_signal = signal / np.sqrt(power)

    # 5. Calculate phase differences between adjacent samples
    phase_differences = np.angle(
        normalized_signal[1:] * np.conj(normalized_signal[:-1])
    )

    phase_diff_sin_mean = np.mean(np.sin(phase_differences))
    phase_diff_cos_mean = np.mean(np.cos(phase_differences))
    phase_diff_std = np.std(phase_differences)

    # 6. Higher-order complex moments
    moment_2 = np.abs(np.mean(normalized_signal ** 2))
    moment_4 = np.abs(np.mean(normalized_signal ** 4))
    moment_8 = np.abs(np.mean(normalized_signal ** 8))

    # 7. Combine all features
    features = {
        "amplitude_mean": amp_mean,
        "amplitude_std": amp_std,
        "amplitude_skewness": amp_skewness,
        "amplitude_kurtosis": amp_kurtosis,
        "peak_amplitude": peak_amplitude,
        "papr": papr,
        "phase_diff_sin_mean": phase_diff_sin_mean,
        "phase_diff_cos_mean": phase_diff_cos_mean,
        "phase_diff_std": phase_diff_std,
        "moment_2": moment_2,
        "moment_4": moment_4,
        "moment_8": moment_8
    }

    return features