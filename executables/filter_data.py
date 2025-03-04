import logging
import os
import sys
from tempfile import TemporaryDirectory

from astroquery.simbad import Simbad
from google.cloud import storage
from project_vars import get_params
from tqdm import tqdm
from utils import upload_folder_to_gcs

from hyper_velocity_stars_detection.sources.source import AstroObject, AstroObjectData

Simbad.SIMBAD_URL = "http://simbad.u-strasbg.fr/simbad/sim-id"


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [%(asctime)s] %(message)s")
    args = get_params(sys.argv[1:])

    # Autenticación en Google Cloud
    project_id = os.getenv("PROJECT_ID")
    bucket_name = os.getenv("BUCKET")
    folder = "raw_data/"
    pm_kms = args.pm_kms
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=folder)
    for blob in tqdm(blobs, desc="Procesando elementos", unit="item"):
        file_name = blob.name.split("/")[-1]
        logging.info(f"Procesando elemento {file_name}")
        # Saltar si es un "subdirectorio"
        if blob.name.endswith("/"):
            continue

        with TemporaryDirectory() as temp_path:
            local_path = os.path.join(temp_path, file_name)
            logging.info(f"Descargando {file_name} desde {folder}...")
            blob.download_to_filename(local_path)
            name = file_name.split("_")[1]
            radio_scale = int(file_name.split("_")[2].replace("r", "").replace(".vot", ""))
            astro_object = AstroObject.get_object(name)
            _ = astro_object.read_object(temp_path, file_name)
            pm_kms = None
            if radio_scale > 1:
                pm_kms = args.pm_kms
            astro_data = AstroObjectData.load_data_from_object(
                astro_object, radio_scale, pmra_kms_min=pm_kms, pmdec_kms_min=pm_kms
            )
            path_temp = os.path.join(temp_path, name)
            astro_data.save(path_temp)
            upload_folder_to_gcs(project_id, bucket_name, path_temp, name)
            os.remove(local_path)  # Borrar archivo local

        logging.info(f"Borrando {file_name} del bucket...")
        blob.delete()
        logging.info(f"{file_name} procesado y eliminado.\n")
