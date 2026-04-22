import numpy as np
import pandas as pd

# Winsorize to clip outlier at lower_q and upper_q percentiles. 
def winsorize_series(series, lower_q=0.01, upper_q=0.99):
    """Clip extreme scores before portfolio construction."""
    lower = series.quantile(lower_q)
    upper = series.quantile(upper_q)
    return series.clip(lower,upper)

# Standardize the score vectors so the scale is consistent.  
def zscore_series(series):
    """Standardize a score vector to mean zero and unit variance."""
    std = series.std(ddof=0)

    if std == 0 or np.isnan(std):
        return pd.Series(0.0,index=series.index)

    return (series - series.mean()) / std

# Build the alpha signal. Winsorize before standardizing so the outliers
# do not distort the mean and std. 
def build_alpha(score_series):
    """Turn a raw thematic score into a normalized alpha signal."""
    score_series = score_series.astype(float)
    score_series = winsorize_series(score_series)
    score_series = zscore_series(score_series)
    return score_series
