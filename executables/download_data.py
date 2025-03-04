import logging
import os
from tempfile import TemporaryDirectory

from google.cloud import storage
from project_vars import BUCKET, PM_KMS_MIN, PROJECT_ID, SELECTED_CLUSTERS
from tqdm import tqdm

from hyper_velocity_stars_detection.utils import download_astro_data

# Autenticación en Google Cloud
project_id = PROJECT_ID
bucket_name = BUCKET


def upload_folder_to_gcs(project_id, bucket_name, temp_path, destination_folder):
    """Sube todos los archivos de temp_path a una carpeta en Google Cloud Storage"""
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


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [%(asctime)s] %(message)s")
    for cluster in tqdm(SELECTED_CLUSTERS, desc="Procesando elementos", unit="item"):
        logging.info(f"Procesando elemento {cluster.name}")
        try:
            with TemporaryDirectory() as temp_path:
                download_astro_data(
                    cluster_name=cluster.name,
                    read_from_cache=True,
                    path=temp_path,
                    radio_scale=cluster.radio_scale,
                    filter_parallax_max=cluster.filter_parallax_max,
                    pmra_kms_min=PM_KMS_MIN,
                    pmdec_kms_min=PM_KMS_MIN,
                )
                path = f"{temp_path}/{cluster.name}/"
                upload_folder_to_gcs(project_id, bucket_name, path, cluster.name)
        except Exception as e:
            logging.info(f"Ha fallado {cluster.name} por {e}")
