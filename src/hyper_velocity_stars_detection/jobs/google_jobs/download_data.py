import argparse
import logging
import os
import sys
from tempfile import TemporaryDirectory

from astroquery.simbad import Simbad
from tqdm import tqdm

from hyper_velocity_stars_detection.jobs.google_jobs.utils import (
    download_from_gcs,
    upload_folder_to_gcs,
)
from hyper_velocity_stars_detection.jobs.utils import download_astro_data, read_catalog_file

Simbad.SIMBAD_URL = "http://simbad.u-strasbg.fr/simbad/sim-id"


def get_params(argv: list[str]) -> argparse.Namespace:
    """
    Función que genera los argumentos de download_data.

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
    parser.add_argument("--path", default="data/globular_clusters/")
    parser.add_argument("--catalog", default="gaiadr3")
    parser.add_argument("--cluster_name", default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [%(asctime)s] %(message)s")
    args = get_params(sys.argv[1:])

    # Autenticación en Google Cloud
    project_id = os.getenv("PROJECT_ID")
    bucket_name = os.getenv("BUCKET")
    cluster_catalog = download_from_gcs(project_id, bucket_name, "mwgc.dat.txt", args.path)
    selected_clusters = read_catalog_file(cluster_catalog)
    if args.cluster_name:
        all_clusters = selected_clusters
        selected_clusters = [
            cluster for cluster in all_clusters if cluster.name == args.cluster_name
        ]

    for cluster in tqdm(selected_clusters, desc="Procesando elementos", unit="item"):
        logging.info(f"Procesando elemento {cluster.name}")
        try:
            with TemporaryDirectory() as temp_path:
                download_astro_data(
                    cluster_name=cluster.name,
                    catalog_name=args.catalog,
                    read_from_cache=True,
                    path=temp_path,
                    radio_scale=cluster.radio_scale,
                    filter_parallax_max=cluster.filter_parallax_max,
                )
                upload_folder_to_gcs(project_id, bucket_name, temp_path, "raw_data")
        except Exception as e:
            logging.info(f"Ha fallado {cluster.name} por {e}")
