import gc
import logging
import os
from typing import Optional

import pandas as pd

from hyper_velocity_stars_detection.astrobjects import (
    AstroObject,
    AstroObjectData,
    AstroObjectProject,
)
from hyper_velocity_stars_detection.jobs.project_vars import PATH, PM_KMS_MIN, SELECTED_CLUSTERS
from hyper_velocity_stars_detection.sources.catalogs import CatalogsType


def download_cluster(
    astro_object: AstroObject,
    read_from_cache: bool,
    path: str,
    radio_scale: float,
    catalog_name: CatalogsType = CatalogsType.GAIA_DR3,
    filter_parallax_min: Optional[float] = None,
    filter_parallax_max: Optional[float] = None,
    filter_parallax_error: float = 0.30,
) -> AstroObject:
    """
    Función que descarga los datos del cluster.

    Parameters
    ----------
    astro_object: AstroObject
        Cluster.
    read_from_cache: bool
        Indica si queremos buscar los datos desde la cache.
    path: str
        Ruta donde se quiere guardar los archivos.
    radio_scale: float, default 1
        Escala del radio de búsqueda
    catalog_name: CatalogsType
             Nombre del tipo del catálogo.
    filter_parallax_min: Optional[float], default None
        Paralaje mínimo que buscar.
    filter_parallax_max: Optional[float], default None
        Paralaje máximo que buscar.
    filter_parallax_error: Optional[float], default None
        Máximo error en el parale a buscar.

    Returns
    -------
    astro_object: AstroObject
        Cluster con los datos descargados.

    """
    result = None
    if read_from_cache:
        try:
            result = astro_object.read_object(
                catalog_name=catalog_name, radius_scale=radio_scale, path=path
            )
            logging.info("\t - Archivo cargado desde cache.")
        except FileNotFoundError:
            logging.info(
                "\t - No hay archivos para cargar en la cache. Se van a descargar los datos."
            )

    if not isinstance(result, pd.DataFrame):
        _ = astro_object.download_object(
            catalog_name=catalog_name,
            radius_scale=radio_scale,
            filter_parallax_min=filter_parallax_min,
            filter_parallax_max=filter_parallax_max,
            filter_parallax_error=filter_parallax_error,
            path=path,
        )
    return astro_object


def download_data(
    cluster_name: str,
    read_from_cache: bool,
    path: str,
    radio_scale: float,
    catalog_name: CatalogsType = CatalogsType.GAIA_DR3,
    filter_parallax_min: float = 0.0,
    filter_parallax_max: Optional[float] = None,
    filter_parallax_error: float = 0.30,
    max_ruwe: float = 1.4,
    pmra_kms_min: Optional[float] = None,
    pmdec_kms_min: Optional[float] = None,
) -> None:
    """
    Función que descarga o carga desde la cache los datos limpios en un proyecto.

    Parameters
    ----------
    cluster_name: str,
        Nombre del cluster
    read_from_cache: bool
        Indica si queremos buscar los datos desde la cache.
    path: str
        Ruta donde se quiere guardar los archivos.
    radio_scale: float, default 1
        Escala del radio de búsqueda
    catalog_name: CatalogsType
             Nombre del tipo del catálogo.
    filter_parallax_min: Optional[float], default None
        Paralaje mínimo que buscar.
    filter_parallax_max: Optional[float], default None
        Paralaje máximo que buscar.
    filter_parallax_error: Optional[float], default None
        Máximo error en el parale a buscar.
    max_ruwe: float
        Máximo RUWE aceptado en la muestra.
    pmra_kms_min: Optional[float], default None
        Filtro opcional. Mímimo valor para pmra en km/s.
    pmdec_kms_min: Optional[float], default None
        Filtro opcional. Mímimo valor para pmra en km/s.
    """
    logging.info("Descargando primer catálogo")
    astro_object_1 = AstroObject.get_object(cluster_name)
    logging.info("Objeto seleccionado: \n" + astro_object_1.info.to_pandas().to_string())

    astro_object_1 = download_cluster(astro_object_1, read_from_cache, path, 1, catalog_name)
    astro_data_1 = AstroObjectData.load_data_from_object(astro_object_1, 1, max_ruwe)
    logging.info(str(astro_data_1))

    logging.info("Descargando segundo catálogo")
    astro_object_r = AstroObject.get_object(cluster_name)
    astro_object_r = download_cluster(
        astro_object_r,
        read_from_cache,
        path,
        radio_scale,
        catalog_name,
        filter_parallax_min,
        filter_parallax_max,
        filter_parallax_error,
    )
    astro_data_r = AstroObjectData.load_data_from_object(
        astro_object_r, radio_scale, max_ruwe, pmra_kms_min, pmdec_kms_min
    )
    logging.info(str(astro_data_r))
    project = AstroObjectProject(cluster_name, path, [astro_data_1, astro_data_r])
    project.save_project()
    gc.collect()


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [%(asctime)s] %(message)s")
    for cluster in SELECTED_CLUSTERS:
        download_data(
            cluster_name=cluster.name,
            read_from_cache=True,
            path=PATH,
            radio_scale=cluster.radio_scale,
            filter_parallax_max=cluster.filter_parallax_max,
            pmra_kms_min=PM_KMS_MIN,
            pmdec_kms_min=PM_KMS_MIN,
        )
