import logging
import os
import sys
from tempfile import TemporaryDirectory

from astroquery.simbad import Simbad
from google.cloud import storage
from project_vars import get_params
from tqdm import tqdm

from hyper_velocity_stars_detection.utils import download_astro_data, read_catalog_file

Simbad.SIMBAD_URL = "http://simbad.u-strasbg.fr/simbad/sim-id"


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
    args = get_params(sys.argv[1:])

    # Autenticación en Google Cloud
    project_id = os.getenv("PROJECT_ID")
    bucket_name = os.getenv("BUCKET")

    selected_clusters = read_catalog_file(os.path.join(args.path, "mwgc.dat.txt"))

    for cluster in tqdm(selected_clusters, desc="Procesando elementos", unit="item"):
        logging.info(f"Procesando elemento {cluster.name}")
        try:
            with TemporaryDirectory() as temp_path:
                download_astro_data(
                    cluster_name=cluster.name,
                    read_from_cache=True,
                    path=temp_path,
                    radio_scale=cluster.radio_scale,
                    filter_parallax_max=cluster.filter_parallax_max,
                )
                upload_folder_to_gcs(project_id, bucket_name, temp_path, "raw_data")
        except Exception as e:
            logging.info(f"Ha fallado {cluster.name} por {e}")
