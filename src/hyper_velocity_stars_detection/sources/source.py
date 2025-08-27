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
    StorageCustomObject,
    StorageObjectPandasCSV,
    StorageObjectPickle,
    StorageObjectTableVotable,
)
from hyper_velocity_stars_detection.sources.catalogs import Catalog, get_catalog
from hyper_velocity_stars_detection.sources.metrics import (
    convert_mas_yr_in_km_s,
    get_l_b_velocities,
)
from hyper_velocity_stars_detection.sources.utils import get_object_from_simbad, get_skycoords

INFO_JSON_NAME = "info_json"
INFO_VOT_NAME = "info_vot"

ASTRO_OBJECT = "astro_object"
ASTRO_OBJECT_DATA = "astro_data"
CATALOG_NAME = "catalog"
RADIO_SCALE = "radio_scale"
DATA = "data"


class DataSampleType:
    DATA_SAMPLE_1 = "_c1"
    DATA_SAMPLE_2 = "_c2"
    DATA_SAMPLE_3 = "_c3"
    DATA_SAMPLE_4 = "_c4"


SAMPLE_DESCRIPTION = {
    DataSampleType.DATA_SAMPLE_1: "Todas las estrellas seleccionadas",
    DataSampleType.DATA_SAMPLE_2: "Las estrellas con errores de paralaje y pm menores al 10%",
    DataSampleType.DATA_SAMPLE_3: "Las estrellas con un error de paralaje menor del 30% y de pm "
    "menores al 10%",
    DataSampleType.DATA_SAMPLE_4: "Las estrellas con un error de paralaje menor del 10% y de pm "
    "menores al 20%.",
}


def get_data_sample(data: pd.DataFrame, sample_type: str) -> np.ndarray:
    """ "
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

    Returns
    -------
    mask_data: np.ndarray
        Array de booleanos que devuelve la muestra seleccionada.
    """
    mask_data = np.ones(data.shape[0]).astype(bool)
    if sample_type.endswith(DataSampleType.DATA_SAMPLE_1):
        return mask_data

    if sample_type.endswith(DataSampleType.DATA_SAMPLE_2):
        mask_data = (
            (data.pmra_error < 0.10) & (data.pmdec_error < 0.10) & (data.parallax_error < 0.10)
        )
        return mask_data

    if sample_type.endswith(DataSampleType.DATA_SAMPLE_3):
        mask_data = (
            (data.pmra_error < 0.10) & (data.pmdec_error < 0.10) & (data.parallax_error < 0.30)
        )
        return mask_data

    if sample_type.endswith(DataSampleType.DATA_SAMPLE_4):
        mask_data = (
            (data.pmra_error < 0.30) & (data.pmdec_error < 0.30) & (data.parallax_error < 0.10)
        )
        return mask_data

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
         path: str, default "."
            Ruta donde se quiere guardar el archivo

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
class AstroObjectData:
    """
    Clase que recoge cuatro muestras de datos de una misma consulta de datos de un objeto
    astronómico.
    """

    astro_object = attrib(type=AstroObject, init=True)
    catalog = attrib(type=Catalog, init=True)
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

    def __str__(self):
        """
        Método que imprime los tamaños de las muestras asociadas al objeto astronómico.
        """
        description = (
            f"Muestras seleccionadas del objeto astronómico {self.astro_object.name} "
            f"con radio {self.radio_scale}:\n"
        )
        for key, description in SAMPLE_DESCRIPTION.items():
            value = get_data_sample(self.data, key).sum()
            description += f"\t - {key} - {description}: {value}.\n"
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
    def load_astro_data(
        cls,
        name: str,
        catalog_name: str,
        radius: Optional[float] = None,
        radius_scale: float = 1.0,
        **filter_params,
    ) -> "AstroObjectData":
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
        object_data: AstroObjectData
            Clase que recoge las cuatro muestras de la consulta.
        """
        logging.info("-- Cargando objeto astronómico %s" % name)
        astro_object = AstroObject.get_object(name)
        if radius is None:
            radius = radius_scale * astro_object.info["ANGULAR_SIZE"][0] / 60
        logging.info("-- Descargando datos principales. Aplicando los siguientes:")
        for key, value in filter_params:
            logging.info("\t %s: %s" % (key, value))

        data = astro_object.download_data(
            catalog_name=catalog_name, radius=radius, **filter_params
        )
        catalog = get_catalog(catalog_name=catalog_name)
        astro_data = cls(astro_object, catalog, radius_scale, data)

        logging.info("-- Preprocesando datos.")
        astro_data.fix_parallax()
        astro_data.calculate_pm_to_kms()
        return astro_data

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
            mask_data = get_data_sample(self.data, g_name)
        except ValueError:
            raise ValueError(f"El catalogo de los datos {g_name} es erroneo. ")
        return self.data[mask_data].copy()

    def fix_parallax(self) -> None:
        """
        Método que modifica el atributo data modificando la columna parallax con el ajuste
        implementado por el catálogo y guardando lso datos originales de paralaje en la columna
        parallax_orig. Si el catálogo no tiene implementado el método de corrección no hace nada.
        """
        if "parallax_orig" in self.data:
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
            self.data["parallax_orig"] = self.data.parallax.values
            self.data["parallax"] = parallax_corrected

    def calculate_pm_to_kms(self) -> None:
        """
        Método que calcula el movimiento propio en km por segundo. Modifica el atributo data.
        """
        self.data["pmra_kms"] = convert_mas_yr_in_km_s(
            self.data["parallax"].values, self.data["pmra"].values
        )
        self.data["pmdec_kms"] = convert_mas_yr_in_km_s(
            self.data["parallax"].values, self.data["pmdec"].values
        )
        self.data["pm_kms"] = np.sqrt(self.data.pmra_kms**2 + self.data.pmdec_kms**2)
        self.data["pm"] = np.sqrt(self.data.pmra**2 + self.data.pmdec**2)
        self.data["pm_l"], self.data["pm_b"] = get_l_b_velocities(
            self.data.ra.values,
            self.data.dec.values,
            self.data.pmra.values,
            self.data.pmdec.values,
        )
