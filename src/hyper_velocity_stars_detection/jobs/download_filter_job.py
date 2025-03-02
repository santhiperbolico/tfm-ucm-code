import logging
import os

from hyper_velocity_stars_detection.jobs.project_vars import PATH, PM_KMS_MIN, SELECTED_CLUSTERS
from hyper_velocity_stars_detection.utils import download_astro_data

if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [%(asctime)s] %(message)s")
    for cluster in SELECTED_CLUSTERS:
        download_astro_data(
            cluster_name=cluster.name,
            read_from_cache=True,
            path=PATH,
            radio_scale=cluster.radio_scale,
            filter_parallax_max=cluster.filter_parallax_max,
            pmra_kms_min=PM_KMS_MIN,
            pmdec_kms_min=PM_KMS_MIN,
        )
