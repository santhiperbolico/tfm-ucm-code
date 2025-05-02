import argparse
import logging
import os
import sys

import pandas as pd
from tqdm import tqdm

from hyper_velocity_stars_detection.jobs.google_jobs.utils import download_from_gcs, load_project
from hyper_velocity_stars_detection.jobs.utils import ProjectDontExist, read_catalog_file
from hyper_velocity_stars_detection.sources.clusters_catalogs import get_all_cluster_data


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
    return parser.parse_args(argv)


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [%(asctime)s] %(message)s")
    args = get_params(sys.argv[1:])

    project_id = os.getenv("PROJECT_ID")
    bucket_name = os.getenv("BUCKET")

    cluster_catalog = download_from_gcs(project_id, bucket_name, "mwgc.dat.txt", args.path)
    selected_clusters = read_catalog_file(cluster_catalog)
    df_clusters_dr2 = get_all_cluster_data()

    df_results = pd.DataFrame(
        columns=[
            "Name",
            "parallax",
            "e_parallax",
            "pmRA_",
            "e_pmRA_",
            "pmDE",
            "e_pmDE",
            "RV",
            "e_RV",
        ]
    )
    selected_clusters = [cl.name for cl in selected_clusters]
    drs_clusters_name = df_clusters_dr2.Name.unique()
    index = 0

    project_dont_exist = []

    other_error = []

    for cluster_name in tqdm(
        drs_clusters_name, desc="Procesando clusters", unit="item", total=drs_clusters_name.size
    ):
        gc = None
        if cluster_name in selected_clusters:
            project = None
            try:
                project = load_project(
                    cluster_name=cluster_name,
                    project_id=os.getenv("PROJECT_ID"),
                    bucket_name=os.getenv("BUCKET"),
                    path=args.path,
                )
            except ProjectDontExist:
                project_dont_exist.append(cluster_name)
            if project is not None:
                try:
                    gc = project.clustering_results.gc
                except Exception as error:
                    other_error.append({"cluster_name": cluster_name, "error": error})

        if isinstance(gc, pd.DataFrame):
            df_results.loc[index] = (
                cluster_name,
                gc.parallax.mean(),
                gc.parallax.std(),
                gc.pmra.mean(),
                gc.pmra.std(),
                gc.pmdec.mean(),
                gc.pmdec.std(),
                gc.radial_velocity.mean(),
                gc.radial_velocity.std(),
            )
            index += 1

    logging.info("Guardando los resultados")
    df_results.to_csv(os.path.join(args.path, "raw_results_dr3.csv"))

    # Aplicamos filtros para quedarnos con los clusters cuyos resultados en DR2 estén dentro del
    # margen de error.
    df_results = pd.merge(df_results, df_clusters_dr2, on="Name", suffixes=["", "_dr2"])
    mask_p = (df_results.parallax_dr2 >= df_results.parallax - df_results.e_parallax) & (
        df_results.parallax_dr2 <= df_results.parallax + df_results.e_parallax
    )
    mask_pmra = (df_results.pmRA__dr2 >= df_results.pmRA_ - df_results.e_pmRA_) & (
        df_results.pmRA__dr2 <= df_results.pmRA_ + df_results.e_pmRA_
    )
    mask_pmde = (df_results.pmDE_dr2 >= df_results.pmDE - df_results.e_pmDE) & (
        df_results.pmDE_dr2 <= df_results.pmDE + df_results.e_pmDE
    )

    mask_ml = ~df_results.M_L.isna()
    df_results = df_results[mask_p & mask_pmra & mask_pmde & mask_ml]
    df_results.to_csv(os.path.join(args.path, "clean_results_dr3.csv"))

    logging.info("Datos Guardados.")
