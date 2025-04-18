import argparse
import logging
import os
import sys

from google.cloud import storage
from tqdm import tqdm

from hyper_velocity_stars_detection.jobs.google_jobs.utils import (
    download_from_gcs,
    load_save_project,
)
from hyper_velocity_stars_detection.jobs.utils import ProjectDontExist, read_catalog_file


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
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction)
    return parser.parse_args(argv)


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [%(asctime)s] %(message)s")
    args = get_params(sys.argv[1:])

    project_id = os.getenv("PROJECT_ID")
    bucket_name = os.getenv("BUCKET")

    cluster_catalog = download_from_gcs(project_id, bucket_name, "mwgc.dat.txt", args.path)
    all_clusters = read_catalog_file(cluster_catalog)
    selected_clusters = all_clusters
    if args.cache:
        selected_clusters = []
        logging.info("Uso de la Cache")
        logging.info("Se van a descargar los datos de los clusters no descargados.")
        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)
        for cluster in all_clusters:
            blob = bucket.blob(f"{cluster.name}.zip")
            if not blob.exists():
                selected_clusters.append(cluster)
        logging.info("Se van a descargar %i de %i" % (len(selected_clusters), len(all_clusters)))

    for cluster in tqdm(selected_clusters, desc="Procesando elementos", unit="item"):
        try:
            path_zip = load_save_project(cluster.name, project_id, bucket_name)
            logging.info(f"{cluster.name} procesado.\n")
        except ProjectDontExist as error:
            logging.info(error)
