import logging
import os
from typing import Optional

import pandas as pd
from astroquery.simbad import Simbad
from attr import attrib, attrs

from hyper_velocity_stars_detection.data_storage import (
    ContainerSerializerZip,
    StorageCustomObject,
    StorageObjectPandasCSV,
    StorageObjectPickle,
)
from hyper_velocity_stars_detection.sources.catalogs import Chandra, XMMNewton, XRSCatalog
from hyper_velocity_stars_detection.sources.source import (
    CATALOG_NAME,
    DATA,
    RADIO_SCALE,
    AstroObject,
)
from hyper_velocity_stars_detection.variables_names import ASTRO_OBJECT, XRSOURCE

Simbad.TIMEOUT = 300
Simbad.ROW_LIMIT = 1


def get_xrs_catalog(catalog_name: str | list[str]) -> list[XRSCatalog]:
    """
    Selector de catalogos, devuelve una lista de catálogos que usar.
    """

    if not isinstance(catalog_name, list):
        catalog_name = [catalog_name]

    dic_catalogs = {
        XMMNewton.catalog_name: XMMNewton,
        Chandra.catalog_name: Chandra,
    }

    list_catalogs = []
    for cat_name in catalog_name:
        try:
            cat = dic_catalogs[cat_name]()
        except KeyError:
            raise ValueError("El catálogo %s no está implementado" % cat_name)
        list_catalogs.append(cat)
    return list_catalogs


@attrs
class XRSourceData:
    """
    Elemento encargado en descargar y gestionar la fuente de rayos X desde el XMNNewton.
    """

    astro_object = attrib(type=AstroObject, init=True)
    catalog = attrib(type=XRSCatalog | list[XRSCatalog], init=True)
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

    def __attrs_post_init__(self):
        if not isinstance(self.catalog, list):
            self.catalog = [self.catalog]

    def __str__(self):
        """
        Método que imprime los tamaños de las muestras asociadas al objeto astronómico.
        """
        description = f"Fuentes de rayos X detectadas en el entorno de {self.astro_object.name}:\n"
        for catalog in self.catalog:
            data = self.get_data(catalog.catalog_name)
            value = data.shape[0]
            description += f"\t - {catalog.catalog_name} - {catalog.catalog_table}: {value}.\n"
        return description

    @property
    def data_name(self) -> str:
        """
        Nombre asociado al conjunto de muestras de los datos asociados.
        """
        catalogs = "-".join([cat.catalog_name for cat in self.catalog])
        return f"{catalogs}_{self.astro_object.name}_" f"r_{int(self.radio_scale)}"

    @classmethod
    def load_data(
        cls,
        name: str,
        catalog_name: str | list[str] = None,
        radius: Optional[float] = None,
        radius_scale: float = 1.0,
        **filter_params,
    ) -> "XRSourceData":
        """
        Método que genera el objeto astro data con lso datos filtrados del objeto astronómico
        a estudiar, descargando los datos correspondientes a la escala del radio de
        visión utilizado.

        Parameters
        ----------
        name: str
            Nombre del objeto.
        catalog_name: str| list[str] = None,
            Nombre o nombres del catálogo utilizado para descargarlso datos. Si no se indica
            coge Chandra y XMMNewton.
        radius: Optional[float] = None
            Radio de búsqueda en grados. Si no se indica se utilizará radius_scale.
        radius_scale: float = 1.0
            Escala del radio de búsqueda en comparación con el campo de visión del objeto.
            El campo de visión es extraido de la base de datos de Simbad.
            Solo se utiliza si no se indica el radius

        Returns
        -------
        object_data: AstroMetricData
            Clase que recoge las cuatro muestras de la consulta.
        """
        if catalog_name is None:
            catalog_name = [XMMNewton.catalog_name, Chandra.catalog_name]

        if isinstance(catalog_name, str):
            catalog_name = [catalog_name]

        logging.info("-- Cargando objeto astronómico %s" % name)
        astro_object = AstroObject.get_object(name)
        if radius is None:
            radius = radius_scale * astro_object.info["ANGULAR_SIZE"][0] / 60
        logging.info("-- Descargando datos principales. Aplicando los siguientes:")
        for key, value in filter_params:
            logging.info("\t %s: %s" % (key, value))

        result = pd.DataFrame()
        for cat_name in catalog_name:
            data = astro_object.download_data(
                catalog_name=cat_name, radius=radius, **filter_params
            )
            result = pd.concat((result, data))
        result = result.reset_index(drop=True)
        catalog = get_xrs_catalog(catalog_name=catalog_name)
        xrs_data = cls(astro_object, catalog, radius_scale, result)
        return xrs_data

    @classmethod
    def load(cls, filepath: str) -> "XRSourceData":
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
        container[CATALOG_NAME] = get_xrs_catalog(container[CATALOG_NAME])
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
            CATALOG_NAME: [cat.catalog_name for cat in self.catalog],
            RADIO_SCALE: self.radio_scale,
            DATA: self.data,
        }
        filename = f"{XRSOURCE}_{self.data_name}"
        file = os.path.join(path, filename)
        self._storage.save(file, container)

    def get_data(self, catalog_name: str) -> pd.DataFrame:
        """
        Método que extrae una copia de los datos asociados al catálogo seleccionado.

        Parameters
        ----------
        catalog_name: str
            Nombre del catalogo que queremos filtrar

        Returns
        -------
        result: pd.DataFrame
            Copia de los datos asociadso al catálogo.
        """
        catalog = get_xrs_catalog(catalog_name)[0]
        result = self.data[self.data["mission"] == catalog.catalog_table].reset_index(drop=True)
        return result
