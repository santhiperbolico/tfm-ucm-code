import logging
import os
from tempfile import TemporaryDirectory

from google.cloud import storage

from hyper_velocity_stars_detection.astrobjects import AstroObjectProject
from hyper_velocity_stars_detection.jobs.utils import (
    DefaultParamsClusteringDetection,
    ProjectDontExist,
)


def upload_folder_to_gcs(project_id, bucket_name, temp_path, destination_folder):
    """
    Sube todos los archivos de una carpeta local a una carpeta en Google Cloud Storage.

    Parameters
    ----------
    project_id : str
        ID del proyecto en Google Cloud.
    bucket_name : str
        Nombre del bucket en Google Cloud Storage.
    temp_path : str
        Ruta local de la carpeta con los archivos a subir.
    destination_folder : str
        Carpeta de destino en el bucket donde se almacenarán los archivos.

    Returns
    -------
    None
        La función no devuelve ningún valor. Sube los archivos al bucket de GCS.
    """
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)

    # Recorrer todos los archivos en la carpeta temporal
    for file_name in os.listdir(temp_path):
        file_path = os.path.join(temp_path, file_name)

        # Verificar que sea un archivo y no un directorio
        if os.path.isfile(file_path):
            # Definir la ruta en el bucket
            blob_path = f"{destination_folder}/{file_name}"
            blob = bucket.blob(blob_path)
            # Subir el archivo
            blob.upload_from_filename(file_path)


def download_from_gcs(project_id: str, bucket_name: str, file_path: str, path: str) -> str:
    """
    Descarga un archivo desde Google Cloud Storage a una ruta local.

    Parameters
    ----------
    project_id : str
        ID del proyecto en Google Cloud.
    bucket_name : str
        Nombre del bucket en Google Cloud Storage.
    file_path : str
        Ruta del archivo dentro del bucket de GCS.
    path : str
        Ruta local donde se guardará el archivo descargado.

    Returns
    -------
    str
        Ruta local del archivo descargado.
    """
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name=file_path)
    file_name = blob.name.split("/")[-1]
    local_path = os.path.join(path, file_name)
    blob.download_to_filename(local_path)
    return local_path


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
            _ = project.cluster_detection(**DefaultParamsClusteringDetection().params)
            project.save_project(to_zip=True)
            blob_path = cluster_name + ".zip"
            path_zip = os.path.join(temp_path, blob_path)
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(path_zip)
            return path_zip
    raise ProjectDontExist("No hay datos descargados del proyecto")
