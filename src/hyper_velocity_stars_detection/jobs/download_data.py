import logging
import os
import sys

from tqdm import tqdm

from hyper_velocity_stars_detection.jobs.utils import (
    download_astro_data,
    get_params,
    read_catalog_file,
)

if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [%(asctime)s] %(message)s")
    args = get_params(sys.argv[1:])
    selected_clusters = read_catalog_file(os.path.join(args.path, "mwgc.dat.txt"))

    for cluster in tqdm(selected_clusters, desc="Procesando elementos", unit="item"):
        logging.info(f"Procesando elemento {cluster.name}")
        try:
            download_astro_data(
                cluster_name=cluster.name,
                read_from_cache=True,
                path=args.path,
                radio_scale=cluster.radio_scale,
                filter_parallax_max=cluster.filter_parallax_max,
            )

        except Exception as e:
            logging.info(f"Ha fallado {cluster.name} por {e}")
