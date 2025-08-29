import gzip
import logging
import os
import shutil
from datetime import datetime, timedelta
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.timeseries.periodograms import LombScargle
from astroquery.esa.xmm_newton import XMMNewton
from attr import attrib, attrs
from scipy.optimize import curve_fit
from scipy.signal import welch


def get_obs_id(obs_ids: list[str | int]) -> list[str]:
    """
    Función que formatea los obs_id de XMMNewton.
    """
    list_ids = []
    for obs_id in obs_ids:
        obs_id = str(obs_id)
        n_id = len(obs_id)
        if n_id < 10:
            obs_id = "".join(["0"] * int(10 - n_id)) + obs_id
        list_ids.append(obs_id)
    return list_ids


def broken_power_law(f, A, f_break, alpha_low, alpha_high) -> float:
    """
    Ley de potencias usada para ajustar la densidad del espectro de potencias.
    P(f) = A * (f/f_break)**(-alpha_low) si f < f_break
    P(f) = A * (f/f_break)**(-alpha_high) si f >= f_break

    Parameters
    ----------
    f: float
        Frecuencia
    A: float
        Amplitud del espectro
    f_break: float
        Frecuencia de rotura
    alpha_low: float
        Exponente predominante en f < f_break
    alpha_high: float
        Exponente predominante en f > f_break

    Returns
    -------
    P(f): float
        Densidad del espectro de potencias.
    """
    return A * (f / f_break) ** (-alpha_low) * (f < f_break) + A * (f / f_break) ** (
        -alpha_high
    ) * (f >= f_break)


def get_psd(
    time: np.ndarray, rate: np.ndarray, method: str, params: Optional[dict[str, Any]] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Función que calcula la densidad del espectro de potencias sando diferentes
    métodos.

    Parameters
    ----------
    time: np.ndarray,
        Array de tiempos
    rate: np.ndarray,
        Array del conteo.
    method: str,
        Método a utilizar ("welch", "ls"). El método ls hace referencia LombScargle.
    params: Optional[dict[str, Any]] = None
        Parámetros adicionales del método.

    Returns
    -------
    frequency: np.ndarray
        Array con las frecuencias calculadas
    psd: np.array
        Array con el espectro de potencias.
    """
    if params is None:
        params = {}
    if method == "welch":
        return welch(rate, fs=1 / np.diff(time).mean(), **params)
    if method == "ls":
        return LombScargle(time, rate, **params).autopower()

    raise ValueError(
        f"El método {method} no estña implementado, prueba con 'welch' o 'ls', "
        f"donde 'ls' hace referencia al método LombScargle"
    )


def plot_light_curve(
    time: np.ndarray,
    rate: np.ndarray,
    rate_err: np.ndarray,
    back: np.ndarray,
    back_err: np.ndarray,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Función que dibuja la curva de luz junto con la emisión del fondo en rayos X.

    Parameters
    ----------
    time: np.ndarray,
        Array de tiempo
    rate: np.ndarray,
        Cuentas por segundo
    rate_err: np.ndarray,
        Error de las cuentas por segundo
    back: np.ndarray,
        Cuentas por segundo del fondo
    back_err: np.ndarray
        Error de las cuentas por segundo del fondo.

    Returns
    -------
    figure: plt.Figure
        Figura de las curvas de luz
    ax: plt.Axes
        Ejes de la figura.

    """
    # Graficar la curva de luz
    fig, ax = plt.subplots(2, 1, figsize=(20, 12))
    ax[0].errorbar(time, rate, yerr=rate_err, fmt=".k", alpha=0.5)
    ax[0].set_xlabel("Tiempo")
    ax[0].set_ylabel("Tasa de cuentas (cts/s)")
    ax[0].set_title("Curva de luz en rayos X")
    ax[0].grid()

    ax[1].errorbar(time, back, yerr=back_err, fmt=".k", alpha=0.5, label="Fondo")
    ax[1].set_xlabel("Tiempo")
    ax[1].set_ylabel("Tasa de cuentas (cts/s)")
    ax[1].set_title("Fondo de rayos X")
    ax[1].grid()
    return fig, ax


@attrs(auto_attribs=True)
class ObservationLC:
    file: str
    obs_id: str
    expid: str
    sensor: str
    date: datetime
    tsart: float
    tstop: float

    @classmethod
    def load_from_header(cls, file_path: str) -> "ObservationLC":
        """
        Método de clase que extrae los parámetros de un archivo fit desde
        el header.

        Parameters
        ----------
        file_path: str
            Archivo FIT

        Returns
        -------
        object: ObservationLC
            Observación
        """
        hdul = fits.open(file_path)
        header = hdul[1].header
        params = {
            "file": hdul.filename(),
            "obs_id": header["OBS_ID"],
            "expid": header["EXPIDSTR"],
            "sensor": header["INSTRUME"],
            "date": datetime.strptime(hdul[0].header["DATE"], "%Y-%m-%dT%H:%M:%S.%f"),
            "tsart": header["TSTART"],
            "tstop": header["TSTOP"],
        }
        return cls(**params)

    def get_header(self) -> dict[str, fits.Header]:
        """
        Método que extrae los headers

        Returns
        -------
        headers: dict[str, fits.Header]
            Headers del FITS
        """
        hdul = fits.open(self.file)
        return {hdul[0].name: hdul[0].header, hdul[1].name: hdul[1].header}

    def get_data(self) -> fits.fitsrec.FITS_rec:
        """
        Método que extrae los datos de un archivo FITS.

        Returns
        -------
        data: astropy.io.fits.fitsrec.FITS_rec

        """
        hdul = fits.open(self.file)
        return hdul["RATE"].data

    def plot(self) -> tuple[plt.Figure, plt.Axes]:
        """
        Método que dibuja la curva de luz.

        Returns
        -------
        figure: plt.Figure
            Figura de las curvas de luz
        ax: plt.Axes
            Ejes de la figura.
        """
        rate_field = "RATE"
        back_field = "BACKV"
        back_err_field = "BACKE"
        rate_err_field = "ERROR"
        data = self.get_data()

        # Extraer el tiempo y la tasa de conteo
        time = data["TIME"] - data["TIME"].min()
        rate = data[rate_field] - data[back_field]
        rate_err = np.abs(data[rate_err_field] - data[back_err_field])

        back = data[back_field]
        back_err = data[back_err_field]

        return plot_light_curve(time, rate, rate_err, back, back_err)


@attrs
class LightCurve:
    path_lightcurves = attrib(type=str, init=True)

    @property
    def instruments(self) -> list[str]:
        return list(self.get_fits_by_sensor().keys())

    def get_light_curves_path(self) -> list[str]:
        """
        Método que extrae los directorios de las curvas de luz para cada curva de luz.

        Returns
        -------
        list_paths: list[str]
            Lista de paths completos de cada observación.

        """
        list_paths = [
            os.path.join(self.path_lightcurves, path)
            for path in os.listdir(self.path_lightcurves)
            if os.path.isdir(os.path.join(self.path_lightcurves, path))
        ]
        return list_paths

    def get_fits(self) -> list[str]:
        """
        Método que extrae los directorios de las curvas de luz para cada curva de luz.

        Returns
        -------
        list_fits: list[str]
            Lista de los fits
        """
        list_paths = self.get_light_curves_path()
        list_fits = []
        # Extraemos todos los paths de curvas de luz de la fuente.
        for path in list_paths:
            list_files = os.listdir(path)
            for file in list_files:
                if file.endswith(".FITS"):
                    list_fits.append(os.path.join(path, file))
        return list_fits

    def get_fits_by_sensor(self) -> dict[str, list[ObservationLC]]:
        """
        Método que extrae los directorios de las curvas de luz para cada curva de luz.

        Returns
        -------
        sensors_files: dict[str, list[ObservationLC]]
            Diccionario con los fits separados por sensor, obs_id y

        """
        sensors_files = {}
        for file_path in self.get_fits():
            observation = ObservationLC.load_from_header(file_path)
            if observation.sensor not in sensors_files:
                sensors_files[observation.sensor] = []
            sensors_files[observation.sensor].append(observation)
        sensors_files.keys()

        return sensors_files

    def get_data(self, sensor: str) -> pd.DataFrame:
        """
        Método qeu carga los datos de todas las observaciones en una misma tabla

        Parameters
        ----------
        sensor: str
            Sensor a graficar.

        Returns
        -------
        data: pd.DataFrame
            Tabla de datos agregada
        """
        """
        Método que dibuja la curva de luz.
        """
        rate_field = "RATE"
        back_field = "BACKV"
        back_err_field = "BACKE"
        rate_err_field = "ERROR"

        observations = self.get_fits_by_sensor()[sensor]
        time = []
        obs_id = []
        rate = []
        rate_err = []
        rate_raw_err = []
        back = []
        back_err = []
        for obs in observations:
            data = obs.get_data()
            time_s = data["TIME"] - data["TIME"].min()
            time += [obs.date + timedelta(seconds=s) for s in time_s]
            rate += list(data[rate_field] - data[back_field])
            rate_err += list(np.abs(data[rate_err_field] - data[back_err_field]))
            rate_raw_err += list(np.abs(data[rate_err_field]))
            back += list(data[back_field])
            back_err += list(data[back_err_field])
            obs_id += [obs.obs_id] * data.shape[0]

        df_data = pd.DataFrame(
            {
                "time": np.array(time),
                "obs_id": np.array(obs_id),
                "rate": np.array(rate),
                "rate_err": np.array(rate_err),
                "rate_raw_err": np.array(rate_raw_err),
                "back": np.array(back),
                "back_err": np.array(back_err),
            }
        ).sort_values(["time"])
        df_data = df_data.drop_duplicates()
        df_data["time_s"] = (df_data["time"] - df_data["time"].min()).dt.total_seconds()
        return df_data

    def plot(self, sensor: str) -> tuple[plt.Figure, plt.Axes]:
        """
        Método que dibuja la curva de luzpara un sensor dado.

        Parameters
        ----------
        sensor: str
            Sensor a graficar.

        Returns
        -------
        figure: plt.Figure
            Figura de las curvas de luz
        ax: plt.Axes
            Ejes de la figura.
        """
        """
        Método que dibuja la curva de luz.
        """
        data = self.get_data(sensor)

        return plot_light_curve(data.time, data.rate, data.rate_err, data.back, data.back_err)

    def get_psd(
        self, sensor: str, method: str, nperseg: int = 258, n_simuls: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Método que calcula al densidad del espectro de potencias.

        Parameters
        ----------
        sensor: str
            Sensor a graficar.
        method: str,
            Método a utilizar ("welch", "ls"). El método ls hace referencia LombScargle.
        nperseg: int = 258

        n_simuls: int = None
            Simulaciones para estimar el error del espectro de potencias.

        Returns
        -------
        frequency: np.ndarray
            Frecuencias.
        psd: np.ndarray
            Array de la densidad del espectro de potencias
        """
        data = self.get_data(sensor)
        time = data.time_s
        rate = data.rate
        params = {}
        if method == "weltch":
            params["nperseg"] = nperseg

        frequencies, psd = get_psd(time, rate, method, params)
        n_rows = frequencies.size
        if n_simuls:
            logging.info("Calculamos las simulaciones para estimar el error del PSD.")
            rate_err = data.rate_err
            frequencies = np.zeros((n_rows, n_simuls))
            psd = np.zeros((n_rows, n_simuls))
            for simul in range(n_simuls):
                rate_simul = np.random.normal(loc=rate, scale=rate_err)
                (frequencies[:, simul], psd[:, simul]) = get_psd(time, rate_simul, method, params)
            return frequencies, psd

        frequencies = frequencies.reshape(-1, 1)
        psd = psd.reshape(-1, 1)
        return frequencies, psd

    def fit_power_law(
        self, sensor: str, method: str, nperseg: int = 258, n_simuls: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Método que calcula el tiempo de rotura, devolviendo los parámetros del ajuste.

        Parameters
        ----------
        sensor: str
            Sensor a graficar.
        method: str,
            Método a utilizar ("welch", "ls"). El método ls hace referencia LombScargle.
        nperseg: int = 258

        n_simuls: int = None
            Simulaciones para estimar el error del espectro de potencias.

        Returns
        -------
        frequency: np.ndarray
            Frecuencias.
        psd: np.ndarray
            Array de la densidad del espectro de potencias
        popt:: np.ndarray
            amplitud: float
                Amplitud de la ley de potencias
            f_break: float
                Frecuencia de rotura
            alpha_low: flow
                exponent lower
            alpha_hig: flow
                exponent high
        """
        frequencies_simuls, psd_simuls = self.get_psd(sensor, method, nperseg, n_simuls)
        frequencies = frequencies_simuls.mean(axis=1)
        psd = psd_simuls.mean(axis=1)
        psd_err = 0
        if n_simuls:
            psd_err = psd_simuls.std(axis=1)

        popt = np.zeros((psd_simuls.shape[1], 4))
        for simul in range(psd_simuls.shape[1]):
            psd_simul = np.random.normal(loc=psd, scale=psd_err)
            p0 = [psd_simul.mean(), frequencies[1], 1, 2]
            bounds = (
                (0, frequencies[1], 0, 0),
                (psd_simul[1:].max(), 2 * frequencies.max(), 2, 5),
            )
            popt[simul, :], _ = curve_fit(
                f=broken_power_law,
                xdata=frequencies[1:],
                ydata=psd_simul[1:],
                p0=p0,
                maxfev=100000,
                bounds=bounds,
                full_output=False,
            )
        return frequencies_simuls, psd_simuls, popt

    def plot_power_law(
        self, sensor: str, method: str, nperseg: int = 258, n_simuls: int = None
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Método que calcula el tiempo de rotura, devolviendo los parámetros del ajuste.

        Parameters
        ----------
        sensor: str
            Sensor a graficar.
        method: str,
            Método a utilizar ("welch", "ls"). El método ls hace referencia LombScargle.
        nperseg: int = 258

        n_simuls: int = None
            Simulaciones para estimar el error del espectro de potencias.

        Returns
        -------
        figure: plt.Figure
            Figura de las curvas de luz
        ax: plt.Axes
            Ejes de la figura.
        """
        frequencies_simul, psd_simul, popt_simul = self.fit_power_law(
            sensor, method, nperseg, n_simuls
        )
        frequencies = frequencies_simul.mean(axis=1)
        psd = psd_simul.mean(axis=1)
        popt = popt_simul.mean(axis=0)
        # Extraer el break timescale (T_B = 1 / f_B)
        time_break = 1 / popt[1]
        # Graficar el PSD y el ajuste
        fig, ax = plt.subplots(figsize=(8, 6))
        if not n_simuls:
            ax.plot(frequencies, psd, label="PSD Data")
        else:
            psd_err = psd_simul.std(axis=1)
            ax.errorbar(frequencies, psd, yerr=psd_err, alpha=0.5)
        ax.loglog(
            frequencies,
            broken_power_law(frequencies, *popt),
            linestyle="--",
            label="Broken Power Law Fit",
        )
        ax.axvline(popt[1], color="r", linestyle=":", label=f"Break Frequency: {popt[1]:.4f} Hz")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power Spectral Density")
        ax.legend()
        ax.set_title(f"Break Timescale: {time_break:.2f} s")
        return fig, ax

    def download_light_curve(self, obs_id: str, cache: bool) -> None:
        """
        Método que descarga las curvas de luz de la observación obs_id.
        Las observaciones se guardan en self.path_lightcurves/obs_id en formato
        FITS.

        Parameters
        ----------
        obs_id: str
            Id de la observación a descargar.
        cache: bool, default False
            Indica si se quiere descargar usando la cache.
        """
        file_tar = f"xmm_data_{obs_id}.tar"
        XMMNewton.download_data(obs_id, extension="FTZ", filename=file_tar, cache=cache)

        os.rename(file_tar, os.path.join(self.path_lightcurves, file_tar))
        file_tar = os.path.join(self.path_lightcurves, file_tar)

        extract_dir = os.path.join(self.path_lightcurves, obs_id)

        os.makedirs(extract_dir, exist_ok=True)

        dic_data = True
        iteration = 1
        while dic_data and iteration < 1000:
            dic = XMMNewton.get_epic_lightcurve(file_tar, iteration)
            dic_data = len(dic) > 0
            for key, fits_list in dic.items():
                for l_file in fits_list:
                    lc_ftz_file = l_file
                    lc_fits_file = lc_ftz_file.replace(".FTZ", ".FITS")
                    file = lc_fits_file.split("/")[-1]
                    with gzip.open(lc_ftz_file, "rb") as f_in:
                        with open(lc_fits_file, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(lc_ftz_file)
                    os.rename(lc_fits_file, os.path.join(extract_dir, file))
            iteration += 1
        try:
            os.removedirs(os.path.join(obs_id, "pps"))
            logging.info(f"\t - Datos de la curva de luz de {obs_id} desargadas.")
        except FileNotFoundError:
            logging.info(f"\t - No hay datos de la curva de luz de {obs_id}.")
