import argparse
import logging
import os
import sys

from tqdm import tqdm

from hyper_velocity_stars_detection.jobs.utils import (
    DefaultParamsClusteringDetection,
    ProjectDontExist,
    load_project,
    read_catalog_file,
)


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


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [%(asctime)s] %(message)s")
    args = get_params(sys.argv[1:])

    all_clusters = read_catalog_file(os.path.join(args.path, "mwgc.dat.txt"))
    selected_clusters = all_clusters
    if args.cache:
        selected_clusters = []
        logging.info("Uso de la Cache")
        logging.info("Se van a descargar los datos de los clusters no descargados.")
        for cluster in all_clusters:
            file = os.path.join(args.path, f"{cluster.name}.zip")
            if not os.path.exists(file):
                selected_clusters.append(cluster)
        logging.info("Se van a descargar %i de %i" % (len(selected_clusters), len(all_clusters)))

    for cluster in tqdm(selected_clusters, desc="Procesando elementos", unit="item"):
        try:
            project = load_project(cluster.name, args.path)
            logging.info("\t - Calculando cluster por defecto.")
            params = DefaultParamsClusteringDetection().params.copy()
            if project.get_data("df_1_c2").shape[0] < 16000:
                params["data_name"] = "df_1_c0"

            df_6 = "df_6_c1"
            if project.get_data("df_6_c1").shape[0] < 18000:
                df_6 = "df_6_c0"

            _ = project.cluster_detection(**params)
            project.save_project(args.path)
            fig, ax = project.plot_cmd(
                hvs_candidates_name=df_6, factor_sigma=2.0, hvs_pm=150, legend=True
            )
            logging.info(f"{cluster.name} procesado.\n")
        except ProjectDontExist as error:
            logging.info(error)
