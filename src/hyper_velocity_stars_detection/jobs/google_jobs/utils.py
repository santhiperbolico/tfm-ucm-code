import logging
import os
from tempfile import TemporaryDirectory

from google.cloud import storage

from hyper_velocity_stars_detection.globular_clusters import GlobularClusterAnalysis
from hyper_velocity_stars_detection.jobs.utils import ProjectDontExist
from hyper_velocity_stars_detection.sources.utils import get_main_id
from hyper_velocity_stars_detection.variables_names import GLOBULAR_CLUSTER_ANALYSIS


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
            blob_path = os.path.join(destination_folder, file_name)
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


def load_globular_cluster(
    cluster_name: str, project_id: str, bucket_name: str, path: str
) -> GlobularClusterAnalysis:
    """
    Función que descarga o carga desde la cache los datos limpios en un proyecto.

    Parameters
    ----------
    cluster_name: str,
        Nombre del cluster
    project_id: str
        ID del proyecto de GCP
    bucket_name: str
        Nombre del bucket de Storage.
    path: str
        Directorio donde se encuentra el objeto en storage.

    Returns
    -------
    gc_object: GlobularClusterAnalysis
        Proyecto donde se guardan los resultados.
    """
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    main_id = get_main_id(cluster_name)

    logging.info(f"Cargando objeto {main_id}")
    file_name = f"{GLOBULAR_CLUSTER_ANALYSIS}_{main_id}.zip"
    file_path = os.path.join(path, file_name)
    blob = bucket.blob(blob_name=file_path)
    if blob.exists():
        with TemporaryDirectory() as temp_path:
            temp_file = os.path.join(temp_path, file_name)
            blob.download_to_filename(temp_file)
            gc_object = GlobularClusterAnalysis.load(temp_file)
        return gc_object

    raise ProjectDontExist("No hay datos descargados del proyecto")
