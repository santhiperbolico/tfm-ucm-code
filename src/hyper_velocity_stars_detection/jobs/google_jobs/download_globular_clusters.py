import argparse
import logging
import os
import sys
from tempfile import TemporaryDirectory
from typing import Optional

import pandas as pd
from tqdm import tqdm

from hyper_velocity_stars_detection.globular_clusters import GlobularClusterAnalysis
from hyper_velocity_stars_detection.jobs.google_jobs.utils import upload_folder_to_gcs
from hyper_velocity_stars_detection.jobs.utils import (
    read_baumgardt_catalog,
    read_clusters_harris_catalog,
)
from hyper_velocity_stars_detection.sources.catalogs import GaiaDR3
from hyper_velocity_stars_detection.variables_names import PARALLAX

RADIUS_SCALE = 1
CATALOG = GaiaDR3.catalog_name
FILTERS = {"ast_params_solved": 3, "ruwe": 1.4, "v_periods_used": 10, "min_parallax": 0}
MAX_STARS_SAMPLE = 5000


def get_params(argv: list[str]) -> argparse.Namespace:
    """
    Función de preprocesado de los argumentos del job de MMM para devolver un
    objeto MarketingMixProject con los parámetros del proceso.

    Parameters
    ----------
    argv: list[str]
        Lista de argumentos del job.

    Returns
    -------
    args: argparse.Namespace
        Argumentos.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="report/gc_clusters/")
    parser.add_argument("--init_pos", default=0, type=int)
    args = parser.parse_args(argv)
    return args


def download_globular_cluster(
    cluster_name: str,
    max_stars_to_clus: int = MAX_STARS_SAMPLE,
    radius: Optional[float] = None,
    radio_scale: Optional[float] = None,
    catalog_name: str = GaiaDR3.catalog_name,
    reference_data: Optional[pd.DataFrame] = None,
    **filters_params,
) -> GlobularClusterAnalysis:
    """
    Función que descarga lso datos de un cluster, implementa la detección del cluster y lo guarda
    en un directorio.

    Parameters
    ----------
    cluster_name: str,
        Nombre del cluster
    max_stars_to_clus: int= MAX_STARS_SAMPLE,
        Número máximo de estrellas a utilizar en la clusterización
    radius: Optional[float] = None
        Radio en grados de búsqueda.
    radio_scale: Optional[float] = None
        Escala del radio de búsqueda.
    catalog_name: CatalogsType = GaiaDR3.catalog_name
             Nombre del tipo del catálogo.
    reference_data: Optional[pd.DataFrame] = None,
        Catálogo de referencia si se quiere utilizar.
    **filters_params
        Parámetros de filtro a implementar.

    Returns
    -------
    gc_object: GlobularClusterAnalysis
        Cúmulo globular donde se ha implementaod un clustering.
    """
    logging.info("-- %s: Descargando datos." % cluster_name)
    gc_object = GlobularClusterAnalysis.load_globular_cluster(
        name=cluster_name,
        radius=radius,
        radius_scale=radio_scale,
        catalog_name=catalog_name,
        **filters_params,
    )

    logging.info("-- %s: Seleccionando estrellas del cluster" % cluster_name)
    reference_cluster = None
    if isinstance(reference_data, pd.DataFrame):
        df_ref = reference_data.loc[reference_data.MAIN_ID == gc_object.name, [PARALLAX]]
        if not df_ref.empty:
            reference_cluster = df_ref.iloc[0]

    gc_object.cluster_star_detection(
        max_stars_to_clus=max_stars_to_clus, reference_cluster=reference_cluster, n_trials=10
    )
    return gc_object


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [%(asctime)s] %(message)s")
    args = get_params(sys.argv[1:])
    selected_clusters = read_clusters_harris_catalog()
    cluster_dr2 = read_baumgardt_catalog()
    project_id = os.getenv("PROJECT_ID")
    bucket_name = os.getenv("BUCKET")

    for cluster in tqdm(selected_clusters, desc="Procesando elementos", unit="item"):
        logging.info(f"Procesando elemento {cluster.name}")
        gc_object = None
        try:
            gc_object = download_globular_cluster(
                cluster_name=cluster.name,
                radius=cluster.radius,
                radio_scale=cluster.radio_scale,
                catalog_name=CATALOG,
                reference_data=cluster_dr2,
                **FILTERS,
            )
        except Exception as e:
            logging.info("!!! %s: Ha fallado la descarga por %s" % (cluster.name, e))

        if gc_object is not None:
            logging.info("-- %s: Guardando resultado." % cluster.name)
            try:
                with TemporaryDirectory() as temp_path:
                    gc_object.save(temp_path)
                    upload_folder_to_gcs(project_id, bucket_name, temp_path, args.path)
            except Exception as e:
                logging.info("!!! %s: Ha fallado el guardado por %s" % (cluster.name, e))
