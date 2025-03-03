import logging
import os
from typing import Optional

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.table.table import Table
from attr import attrib, attrs

from hyper_velocity_stars_detection.data_storage import (
    ContainerSerializerZip,
    StorageObjectPandasCSV,
)
from hyper_velocity_stars_detection.sources.catalogs import Catalog, CatalogsType
from hyper_velocity_stars_detection.sources.metrics import (
    convert_mas_yr_in_km_s,
    get_l_b_velocities,
)
from hyper_velocity_stars_detection.sources.ruwe_tools.dr2.ruwetools import U0Interpolator
from hyper_velocity_stars_detection.sources.utils import get_object_from_simbad, get_skycoords

SAMPLE_DESCRIPTION = [
    "Todas las estrellas seleccionadas",
    "Las estrellas con errores de paralaje y pm menores al 10%",
    "Las estrellas con un error de paralaje menor del 30% y de pm menores al 10%",
    "Las estrellas con un error de paralaje menor del 10% y de pm menores al 20%.",
]


class InvalidFileFormat(RuntimeError):
    """
    Error que se lanza cuando un archivo no tiene el formato esperado.
    """

    pass


def get_radio(coords: Table, radio_scale: float, radius_type: Optional[str] = None) -> float:
    """
    Función que devulve el radio asociado a la búsqueda
    Parameters
    ----------
    radio_scale
    radius_type
    coords

    Returns
    -------

    """
    radius = None
    if radius_type is None:
        radius_type = "vision_fold_radius"

    if radius_type == "vision_fold_radius":
        radius = radio_scale * coords["ANGULAR_SIZE"][0] / 60
    if radius_type == "core_radius":
        radius = radio_scale * coords["CORE_RADIUS"][0]
    if radius_type == "half_light_radius":
        radius = radio_scale * coords["HALF_LIGHT_RADIUS"][0]

    if radius is None:
        raise ValueError(
            'El tipo de radio no es correcto, pruebe con "core_radius", '
            '"half_light_radius" o "vision_fold_radius"'
        )
    return radius


@attrs
class AstroObject:
    name = attrib(init=True, type=str)
    info = attrib(init=True, type=Table)
    coord = attrib(init=True, type=SkyCoord)
    data = attrib(init=False, type=pd.DataFrame)
    file = attrib(init=False, type=str)

    @classmethod
    def get_object(cls, name: str) -> "AstroObject":
        """
        Método de clase que dado el nombre del objeto astronómico obtiene las coordenadas
        del objeto.

        Parameters
        ----------
        name: str
            Nombre del objeto astronómico

        Returns
        -------
        object: AstroObject
            Elemento AstroObject instanciado.
        """
        result_object = get_object_from_simbad(name)
        coord = get_skycoords(result_object, u.deg, u.deg)
        return cls(name, result_object, coord)

    def download_object(
        self,
        catalog_name: CatalogsType,
        radius_scale: float = 1,
        radius_type: Optional[str] = None,
        filter_parallax_min: Optional[float] = None,
        filter_parallax_max: Optional[float] = None,
        filter_parallax_error: Optional[float] = None,
        path: str = ".",
    ) -> pd.DataFrame:
        """
         Método que descarga los datos del objeto astronómico de catalog_name.
         Parameters
         ----------
         catalog_name: CatalogsType
             Nombre del tipo del catálogo.
         radius_scale: float, default 1
             Escala del radio de búsqueda
         radius_type: Optional[str], default None
             Tipo del radio de búsqueda "core_radius", "half_light_radius",
             "vision_fold_radius". Por defecto es "vision_fold_radius".
         filter_parallax_min: Optional[float], default None
             Paralaje mínimo que buscar.
         filter_parallax_max: Optional[float], default None
             Paralaje máximo que buscar.
         filter_parallax_error: Optional[float], default None
             Máximo error en el parale a buscar.
         path: str, default "."
            Ruta donde se quiere guardar el archivo

        Returns
         -------
         results: pd.DataFrame
             Tabla con los datos del catálogo descargados.
        """
        ra = self.coord.ra.value
        dec = self.coord.dec.value

        radius = get_radio(self.info, radius_scale, radius_type)
        catalog = Catalog.get_catalog(catalog_name)
        r_scale = f"{radius_scale:.0f}"
        self.file = os.path.join(path, f"{catalog_name}_{self.name}_r{r_scale}.vot")
        results = catalog.download_results(
            ra=ra,
            dec=dec,
            radius=radius,
            output_file=self.file,
            filter_parallax_min=filter_parallax_min,
            filter_parallax_max=filter_parallax_max,
            filter_parallax_error=filter_parallax_error,
        )
        logging.info(f"Se encontraron {len(results)} fuentes en el radio de búsqueda:")
        self.data = results
        return self.data

    def read_object(
        self,
        path: str,
        file_to_read: Optional[str] = None,
        catalog_name: Optional[CatalogsType] = None,
        radius_scale: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Método que lee los datos del objeto astronómico de un archivo del catálogo.

        Parameters
        ----------
        path: str
            Ruta donde se encuentra el archivo.
        file_to_read: Optional[str], default None.
            Nombre del archivo. Si es None se tiene que indicar el resto de parámetros-
        catalog_name: Optional[CatalogsType], default None
            Nombre del catálogo. Se utiliza para construir el nombre dle archivo
        radius_scale: Optional[float], default None
            Escala del radio de búsqueda. Se utiliza para construir el nombre dle archivo.

        Returns
        -------
        results: pd.DataFrame
            Tabla con los datos del catálogo descargados.

        """

        if file_to_read is None:
            r_scale = f"r{radius_scale:.0f}"
            file_to_read = f"{catalog_name}_{self.name}_{r_scale}.vot"
        self.file = os.path.join(path, file_to_read)
        results = Catalog.read_catalog(self.file)
        self.data = results
        return self.data

    def set_extra_metrics(self) -> pd.DataFrame:
        """
        Método que copia los datos originales y calcula métricas extra como los movimientos
        propios en km por segundo o el movimiento propio en coordenadas galacticas.

        Returns
        -------
        df_r: pd.DataFrame
            Datos copiados con las métricas extra.
        """
        df_r = self.data
        df_r["pmra_kms"] = convert_mas_yr_in_km_s(df_r["parallax"].values, df_r["pmra"].values)
        df_r["pmdec_kms"] = convert_mas_yr_in_km_s(df_r["parallax"].values, df_r["pmdec"].values)
        df_r["pm_kms"] = np.sqrt(df_r.pmra_kms**2 + df_r.pmdec_kms**2)
        df_r["pm"] = np.sqrt(df_r.pmra**2 + df_r.pmdec**2)
        df_r["pm_l"], df_r["pm_b"] = get_l_b_velocities(
            df_r.ra.values, df_r.dec.values, df_r.pmra.values, df_r.pmdec.values
        )

        u0_object = U0Interpolator(n_p=5)
        # Métricas de cualificación de la muestra
        df_r["uwe"] = u0_object.get_uwe_from_gaia(df_r)
        df_r["ruwe"] = df_r["uwe"] / u0_object.get_u0(df_r["phot_g_mean_mag"], df_r["bp_rp"])
        return df_r

    def qualify_data(
        self,
        max_ruwe: float = 1.4,
        pmra_kms_min: Optional[float] = None,
        pmdec_kms_min: Optional[float] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Método que devuelve cuatro muestras de datos siguiendo los siguientes criterios. En todos
        ello se ha aplicado un filtro de calidad donde el RUWE es menor que max_ruwe
        - Todas las estrellas seleccionadas.
        - Las estrellas con errores de paralaje y pm menores al 10%.
        - Las estrellas con un error de paralaje menor del 30% y de pm menores al 10%.
        - Las estrellas con un error de paralaje menor del 10% y de pm menores al 20%.

        Parameters
        ----------
        max_ruwe: float
            Máximo RUWE aceptado en la muestra.
        pmra_kms_min: Optional[float], default None
            Filtro opcional. Mímimo valor para pmra en km/s.
        pmdec_kms_min: Optional[float], default None
            Filtro opcional. Mímimo valor para pmra en km/s.

        Returns
        -------
        dfr_c1: pd.DataFrame
            Todas las estrellas seleccionadas.
        dfr_c2: pd.DataFrame
            Las estrellas con errores de paralaje y pm menores al 10%.
        dfr_c3: pd.DataFrame
            Las estrellas con un error de paralaje menor del 30% y de pm menores al 10%.
        dfr_c4: pd.DataFrame
            Las estrellas con un error de paralaje menor del 10% y de pm menores al 20%.
        """
        dfr_c1 = self.set_extra_metrics()
        mask_filter = dfr_c1.ruwe < max_ruwe
        if pmra_kms_min:
            mask_filter = mask_filter & (dfr_c1.pmra_kms > pmra_kms_min)
        if pmdec_kms_min:
            mask_filter = mask_filter & (dfr_c1.pmdec_kms > pmdec_kms_min)
        dfr_c1 = dfr_c1[mask_filter]

        dfr_c2 = dfr_c1[
            (dfr_c1.pmra_error < 0.10)
            & (dfr_c1.pmdec_error < 0.10)
            & (dfr_c1.parallax_error < 0.10)
        ]
        dfr_c3 = dfr_c1[
            (dfr_c1.pmra_error < 0.10)
            & (dfr_c1.pmdec_error < 0.10)
            & (dfr_c1.parallax_error < 0.30)
        ]
        dfr_c4 = dfr_c1[
            (dfr_c1.pmra_error < 0.30)
            & (dfr_c1.pmdec_error < 0.30)
            & (dfr_c1.parallax_error < 0.10)
        ]
        return dfr_c1, dfr_c2, dfr_c3, dfr_c4


@attrs
class AstroObjectData:
    """
    Clase que recoge cuatro muestras de datos de una misma consulta de datos de un objeto
    astronómico.
    """

    name = attrib(type=str, init=True)
    radio_scale = attrib(type=float, init=True)
    data = attrib(type=dict[str, pd.DataFrame], init=True)

    def __str__(self):
        """
        Método que imprime los tamaños de las muestras asociadas al objeto astronómico.
        """
        description = (
            f"Muestras seleccionadas del objeto astronómico {self.name} "
            f"con radio {self.radio_scale}:\n"
        )
        for key, value in self.data.items():
            item = int(key.split("_")[-1].replace("c", ""))
            description += f"\t - {key} - {SAMPLE_DESCRIPTION[item]}: {value.shape[0]}.\n"
        return description

    @property
    def data_name(self) -> str:
        """
        Nombre asociado al conjunto de muestras de los datos asociados.
        """
        return f"{self.name}_r_{int(self.radio_scale)}"

    @classmethod
    def load_data_from_object(
        cls,
        astro_object: AstroObject,
        radio_scale: float,
        max_ruwe: float = 1.4,
        pmra_kms_min: Optional[float] = None,
        pmdec_kms_min: Optional[float] = None,
    ) -> "AstroObjectData":
        """
        Método que genera cuatro muestras de datos siguiendo los siguientes criterios. En todos
        ello se ha aplicado un filtro de calidad donde el RUWE es menor que max_ruwe
        - Todas las estrellas seleccionadas.
        - Las estrellas con errores de paralaje y pm menores al 10%.
        - Las estrellas con un error de paralaje menor del 30% y de pm menores al 10%.
        - Las estrellas con un error de paralaje menor del 10% y de pm menores al 20%.

        Parameters
        ----------
        astro_object: AstroObject
            Consutla de datos asociada al objeto astronómico de estudio.
        radio_scale:
            Escala del radio de búsqueda.
        max_ruwe: float
            Máximo RUWE aceptado en la muestra.
        pmra_kms_min: Optional[float], default None
            Filtro opcional. Mímimo valor para pmra en km/s.
        pmdec_kms_min: Optional[float], default None
            Filtro opcional. Mímimo valor para pmra en km/s.

        Returns
        -------
        object_data: AstroObjectData
            Clase que recoge las cuatro muestras de la consulta.
        """
        df_name = f"df_{int(radio_scale)}"
        result_data = astro_object.qualify_data(max_ruwe, pmra_kms_min, pmdec_kms_min)
        data = {f"{df_name}_c{i}": result_data[i] for i in range(len(result_data))}
        return cls(astro_object.name, radio_scale, data)

    @classmethod
    def load(cls, filepath: str) -> "AstroObjectData":
        """
        Método de clase que lee los datos desde un archivo zip dado.

        Parameters
        ----------
        filepath: str
            Ruta del archivo zip. Este archivo zip debe seguir la siguiente
            nomenclatura: <object_name>_r_<radius_scale>.zip. Dentro de este
            archivo zip debe solo incluirse archivos <key>.csv

        Returns
        -------
        object_data: AstroObjectData
            Datos cargados con las muestras del objeto astronómico

        """
        filename = filepath.split("/")[-1].replace(".zip", "")
        data_name = filename.split("_")[0]
        radius_scale = int(filename.split("_")[-1].replace("r", "").split(".")[0])
        data = ContainerSerializerZip.load_files(filepath, StorageObjectPandasCSV())
        return cls(data_name, radius_scale, data)

    def save(self, path: str) -> None:
        """
        Método que guarda las muestras del catálogo en un archivo zip.

        Parameters
        ----------
        path: str
            Ruta donde se va a guardar las muestras del catálogo.
        """
        path_name = os.path.join(path, self.data_name)
        serielizer = [StorageObjectPandasCSV] * len(self.data)
        container = ContainerSerializerZip(dict(zip(self.data.keys(), serielizer)))
        container.save(path_name, self.data)

    def get_data(self, g_name: str) -> pd.DataFrame:
        """
        Método que extrae una copia de los datos asociados a la muestra g_name.

        Parameters
        ----------
        g_name: str
            Nombre del grupo de la muestra de datos que se quiere extraer

        Returns
        -------
        df_g: pd.DataFrame
            Copia de los datos asociadso a la muestra g_name.
        """
        try:
            return self.data[g_name].copy()
        except KeyError:
            raise ValueError(
                f"El catalogo de los datos {g_name} es erroneo. "
                f"Prueba con : {list(self.data.keys())}"
            )
