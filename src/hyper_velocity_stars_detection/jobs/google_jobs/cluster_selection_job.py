import argparse
import logging
import os
import sys
from tempfile import TemporaryDirectory

import pandas as pd
from google.cloud import storage
from tqdm import tqdm

from hyper_velocity_stars_detection.jobs.google_jobs.utils import download_from_gcs, load_project
from hyper_velocity_stars_detection.jobs.utils import (
    DefaultParamsClusteringDetection,
    ProjectDontExist,
    read_catalog_file,
)
from hyper_velocity_stars_detection.sources.clusters_catalogs import get_clusters_dr2
from hyper_velocity_stars_detection.sources.utils import get_main_id


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
    parser.add_argument("--init_pos", default=0, type=int)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction)
    return parser.parse_args(argv)


def get_reference_cluster() -> pd.DataFrame:
    """
    Función que descarga los datos del catálogo de referencia.

    Returns
    -------
    clusters_dr2: pd.DataFrame
        Catálogo de referencia
    """
    clusters_dr2 = get_clusters_dr2()
    clusters_dr2 = clusters_dr2.rename(
        columns={"pmRA_": "pmra", "pmDE": "pmdec", "parallax": "parallax_corrected"}
    )
    clusters_dr2["MAIN_ID"] = clusters_dr2["Name"].apply(get_main_id)
    return clusters_dr2


def load_save_project(
    cluster_name: str, project_id: str, bucket_name: str, clusters_dr2: pd.DataFrame
) -> str:
    """
    Función que descarga o carga desde la cache los datos filtrados en un proyecto e implementa
    la selección de clustering y descarga de los datos de Rayos X.

    Parameters
    ----------
    cluster_name: str,
        Nombre del cluster
    project_id: str
        ID del proyecto de GCP
    bucket_name: str
        Nombre del bucket de Storage.
    clusters_dr2: pd.DataFrame
        Catálogo de referencia

    Returns
    -------
    path_zip: str
        Path con el archivo ZIP
    """
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)

    logging.info(f"Procesando elemento {cluster_name}")
    with TemporaryDirectory() as temp_path:
        try:
            project = load_project(cluster_name, project_id, bucket_name, temp_path)
        except ProjectDontExist:
            raise ProjectDontExist("No hay datos descargados del proyecto")

        logging.info("\t - Calculando cluster por defecto.")
        params = DefaultParamsClusteringDetection().params.copy()
        columns = params.get("columns")
        reference = clusters_dr2.loc[clusters_dr2.MAIN_ID == project.name, columns]
        if not reference.empty:
            params["reference_cluster"] = reference.iloc[0]
        if project.get_data("df_1_c2").shape[0] < 16000:
            params["data_name"] = "df_1_c0"

        _ = project.optimize_cluster_detection(**params)
        project.save_project(to_zip=True)
        blob_path = cluster_name + ".zip"
        path_zip = os.path.join(temp_path, blob_path)
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(path_zip)
        return path_zip


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [%(asctime)s] %(message)s")
    args = get_params(sys.argv[1:])

    project_id = os.getenv("PROJECT_ID")
    bucket_name = os.getenv("BUCKET")

    cluster_catalog = download_from_gcs(project_id, bucket_name, "mwgc.dat.txt", args.path)
    all_clusters = read_catalog_file(cluster_catalog)
    selected_clusters = all_clusters[args.init_pos :]
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

    clusters_dr2 = get_reference_cluster()
    for cluster in tqdm(selected_clusters, desc="Procesando elementos", unit="item"):
        try:
            path_zip = load_save_project(cluster.name, project_id, bucket_name, clusters_dr2)
            logging.info(f"{cluster.name} procesado.\n")
        except ProjectDontExist as error:
            logging.info(error)
