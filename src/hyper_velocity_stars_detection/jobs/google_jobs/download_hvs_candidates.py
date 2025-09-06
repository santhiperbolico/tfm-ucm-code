import argparse
import logging
import os
import sys
from tempfile import TemporaryDirectory
from typing import Optional

from tqdm import tqdm

from hyper_velocity_stars_detection.jobs.google_jobs.utils import upload_folder_to_gcs
from hyper_velocity_stars_detection.jobs.utils import (
    read_baumgardt_catalog,
    read_clusters_harris_catalog,
)
from hyper_velocity_stars_detection.sources.catalogs import GaiaDR3
from hyper_velocity_stars_detection.sources.source import AstroMetricData

RADIUS_SCALE = 6
CATALOG = GaiaDR3.catalog_name
FILTERS = {
    "ast_params_solved": 3,
    "ruwe": 1.4,
    "v_periods_used": 10,
    "min_parallax": 0,
    "max_parallax": 0.80,
    "parallax_error": 0.30,
}


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
    parser.add_argument("--path", default="report/hvs_candidates/")
    parser.add_argument("--init_pos", default=0, type=int)
    args = parser.parse_args(argv)
    return args


def download_astrometric_data(
    cluster_name: str,
    radius: Optional[float] = None,
    radio_scale: Optional[float] = None,
    catalog_name: str = GaiaDR3.catalog_name,
    **filters_params,
) -> AstroMetricData:
    """
    Función que descarga lso datos cadindatos a HVS de un cluster.

    Parameters
    ----------
    cluster_name: str,
        Nombre del cluster
    radius: Optional[float] = None
        Radio en grados de búsqueda.
    radio_scale: Optional[float] = None
        Escala del radio de búsqueda.
    catalog_name: CatalogsType = GaiaDR3.catalog_name
             Nombre del tipo del catálogo.
    **filters_params
        Parámetros de filtro a implementar.

    Returns
    -------
    astro_data: AstroMetricData
        Cúmulo globular donde se ha implementaod un clustering.
    """
    logging.info("-- %s: Descargando datos." % cluster_name)
    astro_data = AstroMetricData.load_data(
        name=cluster_name,
        radius=radius,
        radius_scale=radio_scale,
        catalog_name=catalog_name,
        **filters_params,
    )
    return astro_data


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [%(asctime)s] %(message)s")
    args = get_params(sys.argv[1:])
    selected_clusters = read_clusters_harris_catalog()
    cluster_dr2 = read_baumgardt_catalog()
    project_id = os.getenv("PROJECT_ID")
    bucket_name = os.getenv("BUCKET")
    init_pos = args.init_pos

    for cluster in tqdm(selected_clusters[init_pos:], desc="Procesando elementos", unit="item"):
        logging.info(f"Procesando elemento {cluster.name}")
        astro_data = None
        try:
            astro_data = download_astrometric_data(
                cluster_name=cluster.name,
                radius=None,
                radio_scale=RADIUS_SCALE,
                catalog_name=CATALOG,
                **FILTERS,
            )
        except Exception as e:
            logging.info("!!! %s: Ha fallado la descarga por %s" % (cluster.name, e))

        if astro_data is not None:
            logging.info("-- %s: Guardando resultado." % cluster.name)
            try:
                with TemporaryDirectory() as temp_path:
                    astro_data.save(temp_path)
                    upload_folder_to_gcs(project_id, bucket_name, temp_path, args.path)
            except Exception as e:
                logging.info("!!! %s: Ha fallado el guardado por %s" % (cluster.name, e))
