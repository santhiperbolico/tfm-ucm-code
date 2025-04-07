import logging
import os
import sys

from astroquery.simbad import Simbad
from tqdm import tqdm

from hyper_velocity_stars_detection.jobs.google_jobs.utils import (
    download_from_gcs,
    load_save_project,
)
from hyper_velocity_stars_detection.jobs.utils import (
    ProjectDontExist,
    get_params,
    read_catalog_file,
)

Simbad.SIMBAD_URL = "http://simbad.u-strasbg.fr/simbad/sim-id"


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [%(asctime)s] %(message)s")
    args = get_params(sys.argv[1:])

    project_id = os.getenv("PROJECT_ID")
    bucket_name = os.getenv("BUCKET")
    pm_kms = args.pm_kms

    cluster_catalog = download_from_gcs(project_id, bucket_name, "mwgc.dat.txt", args.path)
    selected_clusters = read_catalog_file(cluster_catalog)

    for cluster in tqdm(selected_clusters, desc="Procesando elementos", unit="item"):
        try:
            path_zip = load_save_project(cluster.name, project_id, bucket_name)
            logging.info(f"{cluster.name} procesado.\n")
        except ProjectDontExist as error:
            logging.info(error)
