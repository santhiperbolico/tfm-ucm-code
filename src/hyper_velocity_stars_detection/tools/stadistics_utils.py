import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from attr import attrs
from matplotlib.ticker import LogLocator, NullFormatter
from pingouin import multivariate_normality
from tqdm import tqdm

MAX_SAMPLE = 5000


def hist_sample(sample: np.ndarray, xlabel: str, ylabel: str, title: str, logx: bool = True):
    """
    Función que formatea un hitograma.

    """
    x = np.asarray(sample)
    x = x[np.isfinite(x) & (x > 0.0)]
    if x.size == 0:
        raise ValueError("No hay datos positivos/finítos en imbh_sample.")

    q16, q50, q84 = np.percentile(x, [16, 50, 84])

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

    if logx:
        xmin, xmax = x.min(), x.max()
        xlog = np.log10(x)
        iqr = np.subtract(*np.percentile(xlog, [75, 25]))
        bw = 2 * iqr * (len(xlog) ** (-1 / 3))
        nb = max(10, int(np.clip((xlog.max() - xlog.min()) / max(bw, 1e-6), 10, 80)))

        edges = np.logspace(np.log10(xmin), np.log10(xmax), nb + 1)
        ax.hist(x, bins=edges, edgecolor="black", linewidth=0.5)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10.0))
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        bw = 2 * iqr * (len(x) ** (-1 / 3))
        nb = max(10, int(np.clip((np.ptp(x)) / max(bw, 1e-9), 10, 80)))
        ax.hist(x, bins=nb, edgecolor="black", linewidth=0.5)

    for val, lab in [(q16, "q16"), (q50, "mediana"), (q84, "q84")]:
        ax.axvline(val, linestyle="--", linewidth=1.2)
        ax.text(val, ax.get_ylim()[1] * 0.96, lab, rotation=90, va="top", ha="right", fontsize=9)

    # etiquetas, título, grid y cajita-resumen
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)

    resumen = rf"med={q50:,.0f}   16–84%=[{q16:,.0f}, {q84:,.0f}]"
    ax.annotate(
        resumen,
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="none"),
    )

    plt.tight_layout()
    return fig, ax


@attrs(auto_attribs=True)
class ResultMVN:
    """
    Resultados del test estadístico que comprueba la normalidad.

    Attributes
    ----------
    is_multivariate_normal: bool
        Indica si los datos siguen la normal multivariante.
    statistic_name: str
        Nombre del test estadístico
    statistic: float
        Valor del estadístico
    pval: float
        P-valor obtenido con el test.
    """

    is_multivariate_normal: bool
    statistic_name: str
    statistic: float
    pval: float

    @property
    def values(self) -> np.ndarray:
        return np.array([self.statistic, self.pval])

    def is_normal(self, alpha: float | None = None) -> bool:
        if alpha:
            return self.pval > alpha
        return self.is_multivariate_normal


def is_multivariate_normality_hz(
    df_data: pd.DataFrame | np.ndarray, alpha: float = 0.05
) -> ResultMVN:
    """
    Función que indica si los datos siguen una distribución normal multivariante, aplicando
    el test Henze-Zirkler.

    Parameters
    ----------
    df_data: pd.DataFrame | np.ndarray
        Datos a comprobar su distribción.
    alpha: float, default 0.05
        Nivel de significancia.

    Returns
    -------
    results: ResultMVN
        Reusltados del test estadístico
    """
    hz_results = multivariate_normality(df_data, alpha=alpha)
    results = dict(zip(["statistic", "pval", "is_multivariate_normal"], list(hz_results)))
    results.update({"statistic_name": "Henze-Zirkler"})
    return ResultMVN(**results)


def is_multivariate_normality(
    df_data: pd.DataFrame | np.ndarray, alpha: float = 0.05, max_sample: int = MAX_SAMPLE
) -> ResultMVN:
    """
    Función que indica si los datos siguen una distribución normal multivariante.

    Parameters
    ----------
    df_data: pd.DataFrame | np.ndarray
        Datos a comprobar su distribción.
    alpha: float, default 0.05
        Nivel de significancia.
    max_sample: int, default MAX_SAMPLE
        Muestra máxima que puede analizar d euna vez. Cuando supera este umbral aplica
        el método de monte carlo.

    Returns
    -------
    results: ResultMVN
        Reusltados del test estadístico
    """
    if df_data.shape[0] <= max_sample:
        return is_multivariate_normality_hz(df_data, alpha)

    n_simuls = max(10 * (df_data.shape[0] // max_sample + 1), 100)
    pval_array = np.zeros(n_simuls)
    statistics_array = np.zeros(n_simuls)
    for simul in tqdm(
        range(n_simuls), total=n_simuls, desc="Analizando la muestra", unit="simulación"
    ):
        hz_result = is_multivariate_normality_hz(df_data.sample(max_sample, replace=True))
        pval_array[simul] = hz_result.pval
        statistics_array[simul] = hz_result.statistic
    pval = np.quantile(pval_array, 1 - alpha)
    results = {
        "is_multivariate_normal": pval > alpha,
        "pval": pval,
        "statistic": statistics_array.mean(),
        "statistic_name": "MC - Henze-Zirkler",
    }
    return ResultMVN(**results)
