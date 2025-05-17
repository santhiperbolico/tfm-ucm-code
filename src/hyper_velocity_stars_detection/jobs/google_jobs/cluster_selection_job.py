import argparse
import logging
import os
import sys
from tempfile import TemporaryDirectory

from google.cloud import storage
from tqdm import tqdm

from hyper_velocity_stars_detection.astrobjects import AstroObjectProject
from hyper_velocity_stars_detection.jobs.google_jobs.utils import (
    DefaultParamsClusteringDetection,
    download_from_gcs,
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


def load_save_project(cluster_name: str, project_id: str, bucket_name: str) -> str:
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

    Returns
    -------
    project: AstroObjectProject
        Proyecto donde se guardan los resultados.
    """
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=cluster_name)
    logging.info(f"Procesando elemento {cluster_name}")

    with TemporaryDirectory() as temp_path:
        path_project = os.path.join(temp_path, cluster_name)
        os.makedirs(path_project, exist_ok=True)
        zip_f = []
        for blob in blobs:
            file_name = blob.name.split("/")[-1]
            if blob.name.endswith(".zip") and f"{cluster_name}_r" in file_name:
                local_path = os.path.join(path_project, file_name)
                logging.info(f"\t - Descargando {file_name} desde {cluster_name}...")
                blob.download_to_filename(local_path)
                zip_f.append(local_path)

        if len(zip_f) > 0:
            project = AstroObjectProject.load_project(cluster_name, temp_path)
            logging.info("\t - Calculando cluster por defecto.")
            params = DefaultParamsClusteringDetection().params.copy()
            if project.get_data("df_1_c2").shape[0] < 16000:
                params["data_name"] = "df_1_c0"

            _ = project.cluster_detection(**params)
            project.save_project(to_zip=True)
            blob_path = cluster_name + ".zip"
            path_zip = os.path.join(temp_path, blob_path)
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(path_zip)
            return path_zip
    raise ProjectDontExist("No hay datos descargados del proyecto")


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
