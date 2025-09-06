from typing import Optional

import numpy as np
import pandas as pd

from hyper_velocity_stars_detection.globular_clusters import GlobularClusterAnalysis
from hyper_velocity_stars_detection.sources.source import AstroMetricData, DataSample4


def get_hvs_candidates(
    gc_object: GlobularClusterAnalysis,
    v_kms_limit: float,
    astrometric_data: Optional[AstroMetricData] = None,
    sample_label: str = DataSample4.label,
    random_state: Optional[int] = None,
    ipd_gof_harmonic_amplitude: Optional[float] = 0.15,
    ipd_frac_multi_peak: Optional[int] = 10,
) -> pd.DataFrame:
    if astrometric_data is None:
        astrometric_data = gc_object.astro_data
    df_gc = gc_object.clustering_results.remove_outliers_gc(random_state=random_state).copy()
    parallax_mean = df_gc.parallax.mean()
    parallax_std = df_gc.parallax.std()
    df_candidates = astrometric_data.get_data(sample_label)
    df_candidates = df_candidates[
        (df_candidates.parallax >= parallax_mean - 2 * parallax_std)
        & (df_candidates.parallax <= parallax_mean + 2 * parallax_std)
        & (df_candidates.parallax >= 0)
        & (df_candidates.parallax_error < 0.3)
    ]
    if ipd_gof_harmonic_amplitude:
        df_candidates = df_candidates[
            df_candidates.ipd_gof_harmonic_amplitude < ipd_gof_harmonic_amplitude
        ]
    if ipd_frac_multi_peak:
        df_candidates = df_candidates[df_candidates.ipd_frac_multi_peak < ipd_frac_multi_peak]

    df_candidates["pmra_kms"] = df_candidates["pmra_kms"] - df_gc["pmra_kms"].mean()
    df_candidates["pmdec_kms"] = df_candidates["pmdec_kms"] - df_gc["pmdec_kms"].mean()
    df_candidates["v_tan"] = np.sqrt(df_candidates.pmra_kms**2 + df_candidates.pmdec_kms**2)
    return df_candidates[df_candidates["v_tan"] >= v_kms_limit]
