import logging
import os

from project_vars import PATH, SELECTED_CLUSTERS
from tqdm import tqdm

from hyper_velocity_stars_detection.utils import download_astro_data

if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [%(asctime)s] %(message)s")
    for cluster in tqdm(SELECTED_CLUSTERS, desc="Procesando elementos", unit="item"):
        logging.info(f"Procesando elemento {cluster.name}")
        try:
            download_astro_data(
                cluster_name=cluster.name,
                read_from_cache=True,
                path=PATH,
                radio_scale=cluster.radio_scale,
                filter_parallax_max=cluster.filter_parallax_max,
            )

        except Exception as e:
            logging.info(f"Ha fallado {cluster.name} por {e}")
