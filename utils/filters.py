import numpy as np
from scipy.signal import savgol_filter


def remove_outliers_iqr(data, multiplier=1.5):
    if len(data) < 4:
        return data
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    return np.clip(data, lower_bound, upper_bound)


def remove_outliers_zscore(data, threshold=3.0):
    if len(data) < 4:
        return data
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return data
    z_scores = np.abs((data - mean) / std)
    filtered_data = data.copy()
    filtered_data[z_scores > threshold] = mean
    return filtered_data


def median_filter(data, window_size=5):
    if len(data) < window_size:
        return data
    filtered = np.copy(data)
    half_window = window_size // 2
    for i in range(half_window, len(data) - half_window):
        window = data[i - half_window:i + half_window + 1]
        filtered[i] = np.median(window)
    return filtered


def smooth_data(data, window_length=5, polyorder=2):
    if len(data) < window_length:
        return data
    valid_window = min(window_length, len(data) - 1)
    if valid_window % 2 == 0:
        valid_window -= 1
    if valid_window < polyorder + 1:
        valid_window = polyorder + 2 if (polyorder + 2) <= len(data) else polyorder + 1
        if valid_window % 2 == 0: valid_window += 1

    try:
        if valid_window > polyorder and valid_window <= len(data):
            return savgol_filter(data, window_length=valid_window, polyorder=polyorder, mode='interp')
    except:
        pass
    return data


def comprehensive_filter(data, use_median=True, use_zscore=True, use_iqr=True, use_smooth=True,
                         median_window=5, zscore_threshold=3.0, iqr_multiplier=2.0, smooth_window=7):
    filtered = np.copy(data)
    if use_median:
        filtered = median_filter(filtered, window_size=median_window)
    if use_zscore:
        filtered = remove_outliers_zscore(filtered, threshold=zscore_threshold)
    if use_iqr:
        filtered = remove_outliers_iqr(filtered, multiplier=iqr_multiplier)
    if use_smooth:
        filtered = smooth_data(filtered, window_length=smooth_window, polyorder=2)
    return filtered