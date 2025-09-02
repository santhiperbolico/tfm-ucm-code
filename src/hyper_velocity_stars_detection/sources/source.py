import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.table.table import Table
from attr import attrib, attrs

from hyper_velocity_stars_detection.data_storage import (
    ContainerSerializerZip,
    StorageCustomObject,
    StorageObjectPandasCSV,
    StorageObjectPickle,
    StorageObjectTableVotable,
)
from hyper_velocity_stars_detection.sources.catalogs import GaiaCatalog, get_catalog
from hyper_velocity_stars_detection.sources.metrics import (
    convert_mas_yr_in_km_s,
    get_l_b_velocities,
)
from hyper_velocity_stars_detection.sources.utils import get_object_from_simbad, get_skycoords
from hyper_velocity_stars_detection.variables_names import (
    ASTRO_OBJECT,
    ASTRO_OBJECT_DATA,
    PARALLAX,
    PARALLAX_ERROR,
    PM,
    PM_DEC,
    PM_DEC_ERROR,
    PM_DEC_KMS,
    PM_G_LATITUDE,
    PM_G_LONGITUDE,
    PM_KMS,
    PM_RA,
    PM_RA_ERROR,
    PM_RA_KMS,
)

INFO_JSON_NAME = "info_json"
INFO_VOT_NAME = "info_vot"

CATALOG_NAME = "catalog"
RADIO_SCALE = "radio_scale"
DATA = "data"


class DataSample(ABC):
    label = "sample_label"
    description = "description"

    @staticmethod
    @abstractmethod
    def get_sample(df_data: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class DataSample1(DataSample):
    label = "df_c1"
    description = "Todas las estrellas seleccionadas"

    @staticmethod
    def get_sample(df_data: pd.DataFrame) -> np.ndarray:
        """
        Método que devuelve una mascara con una muestra de datos siguiendo el criterio:
        - Todas las estrellas seleccionadas

        Parameters
        ----------
        df_data: pd.DataFrame,
            Datos originales
        Returns
        -------
        mask_data: np.ndarray
            Array de booleanos que devuelve la muestra seleccionada.
        """
        mask_data = np.ones(df_data.shape[0]).astype(bool)
        return mask_data


class DataSample2(DataSample):
    label = "df_c2"
    description = "Las estrellas con errores de paralaje y pm menores al 10%"

    @staticmethod
    def get_sample(df_data: pd.DataFrame) -> np.ndarray:
        """
        Método que devuelve una mascara con una muestra de datos siguiendo el criterio:
        - Las estrellas con errores de paralaje y pm menores al 10%

        Parameters
        ----------
        df_data: pd.DataFrame,
            Datos originales
        Returns
        -------
        mask_data: np.ndarray
            Array de booleanos que devuelve la muestra seleccionada.
        """
        mask_data = (
            (df_data[PM_RA_ERROR] < 0.10)
            & (df_data[PM_DEC_ERROR] < 0.10)
            & (df_data[PARALLAX_ERROR] < 0.10)
        )
        return mask_data


class DataSample3(DataSample):
    label = "df_c3"
    description = "Las estrellas con un error de paralaje menor del 30% y de pm menores al 10%"

    @staticmethod
    def get_sample(df_data: pd.DataFrame) -> np.ndarray:
        """
        Método que devuelve una mascara con una muestra de datos siguiendo el criterio:
        - Las estrellas con un error de paralaje menor del 30% y de pm menores al 10%

        Parameters
        ----------
        df_data: pd.DataFrame,
            Datos originales
        Returns
        -------
        mask_data: np.ndarray
            Array de booleanos que devuelve la muestra seleccionada.
        """
        mask_data = (
            (df_data[PM_RA_ERROR] < 0.10)
            & (df_data[PM_DEC_ERROR] < 0.10)
            & (df_data[PARALLAX_ERROR] < 0.30)
        )
        return mask_data


class DataSample4(DataSample):
    label = "df_c4"
    description = "Las estrellas con un error de paralaje menor del 10% y de pm menores al 20%."

    @staticmethod
    def get_sample(df_data: pd.DataFrame) -> np.ndarray:
        """
        Método que devuelve una mascara con una muestra de datos siguiendo el criterio:
        - Las estrellas con un error de paralaje menor del 10% y de pm menores al 20%.

        Parameters
        ----------
        df_data: pd.DataFrame,
            Datos originales
        Returns
        -------
        mask_data: np.ndarray
            Array de booleanos que devuelve la muestra seleccionada.
        """

        mask_data = (
            (df_data[PM_RA_ERROR] < 0.30)
            & (df_data[PM_DEC_ERROR] < 0.30)
            & (df_data[PARALLAX_ERROR] < 0.10)
        )
        return mask_data


def get_data_sample(
    data: pd.DataFrame, sample_type: str, data_sample_types: list[DataSample]
) -> np.ndarray:
    """
    Método que devuelve una mascara con una muestra  de datos siguiendo los siguientes
    criterios y el tipo de muestra seleccionada.
    - DATA_SAMPLE_1: Todas las estrellas seleccionadas.
    - DATA_SAMPLE_2: Las estrellas con errores de paralaje y pm menores al 10%.
    - DATA_SAMPLE_3: Las estrellas con un error de paralaje menor del 30% y de pm menores al 10%.
    - DATA_SAMPLE_4: Las estrellas con un error de paralaje menor del 10% y de pm menores al 20%.

    Parameters
    ----------
    data: pd.DataFrame,
        Datos originales
    sample_type: str
        Nombre de los datos donde termina con el sufico que indica el tipo de muestra.
    data_sample_types: list[DataSample]
        Lista de las muestras de viables a analizar.

    Returns
    -------
    mask_data: np.ndarray
        Array de booleanos que devuelve la muestra seleccionada.
    """
    for data_sample in data_sample_types:
        if sample_type == data_sample.label:
            return data_sample.get_sample(data)

    raise ValueError(
        "El tipo de muestra %s que se quiere seleccionar no está implementada." % sample_type
    )


@attrs
class AstroObject:
    name = attrib(init=True, type=str)
    main_id = attrib(init=True, type=str)
    info = attrib(init=True, type=Table)
    coord = attrib(init=True, type=SkyCoord)
    _storage = ContainerSerializerZip(
        serializers={
            INFO_JSON_NAME: StorageObjectPickle(),
            INFO_VOT_NAME: StorageObjectTableVotable(),
        }
    )

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
        main_id = "_".join(result_object["MAIN_ID"][0].split())
        coord = get_skycoords(result_object, u.deg, u.deg)
        return cls(name, main_id, result_object, coord)

    def download_data(self, catalog_name: str, radius: float, **filter_params) -> pd.DataFrame:
        """
         Método que descarga los datos del objeto astronómico de catalog_name.
         Parameters
         ----------
         catalog_name: CatalogsType
             Nombre del tipo del catálogo a utilizar.
         radius: float
            Radio de búsqueda, en grados.

        Returns
         -------
         results: pd.DataFrame
             Tabla con los datos del catálogo descargados.
        """
        ra = self.coord.ra.value
        dec = self.coord.dec.value
        catalog = get_catalog(catalog_name)

        data = catalog.download_data(
            ra=ra, dec=dec, radius=radius, output_file=None, **filter_params
        )
        return data

    def save(self, path: str) -> None:
        """
        Método que guarda el objeto en un archivo zip con la información del elemento.

        Parameters
        ----------
        path: str
            Ruta donde se encuentra el archivo.
        """
        info_json = {
            "name": self.name,
            "main_id": self.main_id,
            "coord": self.coord.to_string(precision=16).split(),
        }

        container = {INFO_JSON_NAME: info_json, INFO_VOT_NAME: self.info}
        filename = f"{ASTRO_OBJECT}_{self.main_id}"
        file_path = os.path.join(path, filename)
        self._storage.save(file_path, container)

    @classmethod
    def load(cls, path: str) -> "AstroObject":
        """
        Método que carga el objeto en un archivo zip con la información del elemento.

        Parameters
        ----------
        path: str
            Ruta donde se encuentra el archivo.
        """
        file = path
        if not path.endswith(".zip"):
            file = file + ".zip"
        container = cls._storage.load(file)
        info_json = container[INFO_JSON_NAME]
        info = container[INFO_VOT_NAME]
        coords = info_json.get("coord")
        info_json["coord"] = SkyCoord(
            ra=float(coords[0]), dec=float(coords[1]), unit=(u.deg, u.deg)
        )
        return cls(info=info, **info_json)


@attrs
class AstroMetricData:
    """
    Clase que recoge cuatro muestras de datos de una misma consulta de datos de un objeto
    astronómico.
    """

    astro_object = attrib(type=AstroObject, init=True)
    catalog = attrib(type=GaiaCatalog, init=True)
    radio_scale = attrib(type=float, init=True)
    data = attrib(type=pd.DataFrame, init=True, default=pd.DataFrame())

    _storage = ContainerSerializerZip(
        serializers={
            ASTRO_OBJECT: StorageCustomObject(custom_class=AstroObject, prefix=ASTRO_OBJECT),
            CATALOG_NAME: StorageObjectPickle(),
            RADIO_SCALE: StorageObjectPickle(),
            DATA: StorageObjectPandasCSV(),
        }
    )
    _data_samples = [DataSample1(), DataSample2(), DataSample3(), DataSample4()]

    def __str__(self):
        """
        Método que imprime los tamaños de las muestras asociadas al objeto astronómico.
        """
        description = (
            f"Muestras seleccionadas del objeto astronómico {self.astro_object.name} "
            f"con radio {self.radio_scale}:\n"
        )
        for data_sample in self._data_samples:
            value = get_data_sample(self.data, data_sample.label, self._data_samples).sum()
            description += f"\t - {data_sample.label} - {data_sample.description}: {value}.\n"
        return description

    @property
    def data_name(self) -> str:
        """
        Nombre asociado al conjunto de muestras de los datos asociados.
        """
        return (
            f"{self.catalog.catalog_name}_{self.astro_object.name}_" f"r_{int(self.radio_scale)}"
        )

    @classmethod
    def load_data(
        cls,
        name: str,
        catalog_name: str,
        radius: Optional[float] = None,
        radius_scale: float = 1.0,
        **filter_params,
    ) -> "AstroMetricData":
        """
        Método que genera el objeto astro data con lso datos filtrados del objeto astronómico
        a estudiar, descargando los datos correspondientes a la escala del radio de
        visión utilizado.

        Parameters
        ----------
        name: str
            Nombre del objeto.
        catalog_name: str
            Nombre del catálogo utilizado para descargarlso datos
        radius: Optional[float] = None
            Radio de búsqueda en grados. SI no se indica se utilizará radius_scale.
        radius_scale: float = 1.0
            Escala del radio de búsqueda en comparación con el campo de visión del objeto.
            El campo de visión es extraido de la base de datos de Simbad.
            Solo se utiliza si no se indica el radius

        Returns
        -------
        object_data: AstroMetricData
            Clase que recoge las cuatro muestras de la consulta.
        """
        logging.info("-- Cargando objeto astronómico %s" % name)
        astro_object = AstroObject.get_object(name)
        if radius is None:
            radius = radius_scale * astro_object.info["ANGULAR_SIZE"][0] / 60

        catalog = get_catalog(catalog_name=catalog_name)
        if not isinstance(catalog, GaiaCatalog):
            raise ValueError(
                "El catalogo seleccionado no se corresponde con uno válido para" "astrometría."
            )

        logging.info("-- Descargando datos principales. Aplicando los siguientes filtros:")
        for key, value in filter_params.items():
            logging.info("\t %s: %s" % (key, value))

        data = astro_object.download_data(
            catalog_name=catalog_name, radius=radius, **filter_params
        )
        astro_data = cls(astro_object, catalog, radius_scale, data)

        logging.info("-- Preprocesando datos.")
        astro_data.fix_parallax()
        astro_data.calculate_pm_to_kms()
        return astro_data

    @classmethod
    def load(cls, filepath: str) -> "AstroMetricData":
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
        object_data: AstroMetricData
            Datos cargados con las muestras del objeto astronómico

        """
        path = filepath
        if not filepath.endswith(".zip"):
            path = path + ".zip"
        container = cls._storage.load(path)
        container[CATALOG_NAME] = get_catalog(container[CATALOG_NAME])
        return cls(**container)

    def save(self, path: str) -> None:
        """
        Método que guarda las muestras del catálogo en un archivo zip.

        Parameters
        ----------
        path: str
            Ruta donde se va a guardar las muestras del catálogo.
        """
        container = {
            ASTRO_OBJECT: self.astro_object,
            CATALOG_NAME: self.catalog.catalog_name,
            RADIO_SCALE: self.radio_scale,
            DATA: self.data,
        }
        filename = f"{ASTRO_OBJECT_DATA}_{self.data_name}"
        file = os.path.join(path, filename)
        self._storage.save(file, container)

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
            mask_data = get_data_sample(self.data, g_name, self._data_samples)
        except ValueError:
            raise ValueError(
                f"El catalogo de los datos {g_name} es erroneo. Selecciona"
                f"una muestra de: {list(ds.label for ds in self._data_samples)}"
            )
        return self.data[mask_data].copy()

    def get_data_max_samples(self, max_sample: int) -> str:
        """
        Método que extrae una copia de los datos asociados a la muestra con un tamaño
        máximo que no sobre pase max_sample. SI no hay ninguna que cumpla esa condición
        devuelve la muestra con menor tamaño.

        Parameters
        ----------
        max_sample: int
            Volumen máximo de muestra.

        Returns
        -------
        sample_label: str
            Nombre de la muestra seleccionada.
        """
        size_valid_samples = np.zeros(len(self._data_samples))
        size_samples = np.zeros(len(self._data_samples))
        for pos, sample in enumerate(self._data_samples):
            size = get_data_sample(self.data, sample.label, self._data_samples).sum()
            size_samples[pos] = size
            if size < max_sample:
                size_valid_samples[pos] = size

        pos = int(np.argmax(size_valid_samples))
        if size_valid_samples.sum() == 0:
            pos = int(np.argmin(size_samples))
        sample_label = self._data_samples[pos].label
        return sample_label

    def fix_parallax(self) -> None:
        """
        Método que modifica el atributo data modificando la columna parallax con el ajuste
        implementado por el catálogo y guardando lso datos originales de paralaje en la columna
        parallax_orig. Si el catálogo no tiene implementado el método de corrección no hace nada.
        """
        if PARALLAX + "_orig" in self.data:
            raise RuntimeError(
                "Ya existe una columna ajustada, revisa tus datos y restablece"
                "la columna 'parallax_corrected' en caso de querer volver a "
                "ejecutar este método."
            )
        parallax_corrected = None
        try:
            parallax_corrected = self.catalog.fix_parallax_zero_point(self.data)
        except NotImplementedError:
            logging.info(
                "No se puede ajustar el paralaje porque no está implementado "
                "en el catálogo %s" % self.catalog.catalog_name
            )
        if isinstance(parallax_corrected, np.ndarray):
            self.data[PARALLAX + "_orig"] = self.data.parallax.values
            self.data[PARALLAX] = parallax_corrected

    def calculate_pm_to_kms(self) -> None:
        """
        Método que calcula el movimiento propio en km por segundo. Modifica el atributo data.
        """
        self.data[PM_RA_KMS] = convert_mas_yr_in_km_s(
            self.data[PARALLAX].values, self.data[PM_RA].values
        )
        self.data[PM_DEC_KMS] = convert_mas_yr_in_km_s(
            self.data[PARALLAX].values, self.data[PM_DEC].values
        )
        self.data[PM_KMS] = np.sqrt(self.data.pmra_kms**2 + self.data.pmdec_kms**2)
        self.data[PM] = np.sqrt(self.data.pmra**2 + self.data.pmdec**2)
        self.data[PM_G_LONGITUDE], self.data[PM_G_LATITUDE] = get_l_b_velocities(
            self.data.ra.values,
            self.data.dec.values,
            self.data.pmra.values,
            self.data.pmdec.values,
        )
