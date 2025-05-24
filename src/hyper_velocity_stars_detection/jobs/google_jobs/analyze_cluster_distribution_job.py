import argparse
import logging
import os
import sys
from tempfile import TemporaryDirectory

import pandas as pd
from google.cloud import storage
from tqdm import tqdm

from hyper_velocity_stars_detection.astrobjects import AstroObjectProject
from hyper_velocity_stars_detection.jobs.google_jobs.utils import download_from_gcs, load_project
from hyper_velocity_stars_detection.jobs.utils import ProjectDontExist, read_catalog_file
from hyper_velocity_stars_detection.tools.stadistics_utils import is_multivariate_normality


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


def get_project_from_cluster(cluster_name: str, path: str) -> AstroObjectProject | None:
    """
    Función que extrae en el caso de que corresponda las estrellas del cúmulo globular
    encontrado por el método.

    Parameters
    ----------
    cluster_name: str
        Nombre del cúmulo globular
    path: str
        Ruta donde descargar los datos.

    Returns
    -------
    project: AstroObjectProject | None
        Si se detecta que el cúmulo globular ha sido estudiado se devuelve el
        proyecto seleccionado. En caso contrario devuelve None.
    """
    project = None
    if cluster_name in selected_clusters:
        try:
            project = load_project(
                cluster_name=cluster_name,
                project_id=os.getenv("PROJECT_ID"),
                bucket_name=os.getenv("BUCKET"),
                path=path,
            )
        except ProjectDontExist:
            return None
    return project


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [%(asctime)s] %(message)s")
    args = get_params(sys.argv[1:])

    project_id = os.getenv("PROJECT_ID")
    bucket_name = os.getenv("BUCKET")

    cluster_catalog = download_from_gcs(project_id, bucket_name, "mwgc.dat.txt", args.path)
    selected_clusters = read_catalog_file(cluster_catalog)
    selected_clusters = [cl.name for cl in selected_clusters]

    cluster_analyze = pd.DataFrame(
        columns=["name", "columns", "statistic_name", "statistic", "pval", "is_normal"]
    )

    columns_params = {
        "5p": ["ra", "dec", "pmra", "pmdec", "parallax_corrected"],
        "3p": ["pmra", "pmdec", "parallax_corrected"],
        "5p_color": ["pmra", "pmdec", "parallax_corrected", "bp_rp", "phot_g_mean_flux"],
        "7p": ["ra", "dec", "pmra", "pmdec", "parallax_corrected", "bp_rp", "phot_g_mean_flux"],
    }

    pos_table = 0
    for cluster_name in tqdm(
        selected_clusters, desc="Procesando clusters", unit="item", total=len(selected_clusters)
    ):
        with TemporaryDirectory() as temp_path:
            project = get_project_from_cluster(cluster_name, temp_path)
            if project is None:
                logging.info("El proyecto %s no existe." % cluster_name)
            else:
                data_name = "df_1_c2"
                if project.get_data(data_name).shape[0] < 16000:
                    data_name = "df_1_c0"
                df_stars = project.data_list[0].data[data_name]
                df_stars = df_stars[~df_stars[columns_params.get("7p")].isna()]

                for key, columns in columns_params.items():
                    logging.info("-- Procesando %s: %s." % (cluster_name, key))
                    columns_to_clus = columns_params.get(key)
                    results = is_multivariate_normality(df_stars[columns_to_clus], max_sample=5000)
                    cluster_analyze.loc[pos_table] = pd.Series(
                        {
                            "name": cluster_name,
                            "columns": key,
                            "statistic_name": results.statistic_name,
                            "statistic": results.statistic,
                            "pval": results.pval,
                            "is_normal": results.is_multivariate_normal,
                        }
                    )
                    pos_table += 1

    with TemporaryDirectory() as temp_path:
        file_name = "cluster_analyze_distribution.csv"
        file_path = os.path.join(temp_path, file_name)
        logging.info("-- Guardando los datos en  %s." % file_name)
        cluster_analyze.to_csv(file_path, index=False)
        logging.info("-- Subiendo datos a Storage %s." % file_name)
        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        blob.upload_from_filename(file_path)
        logging.info("-- Archivo %s subido a Storage ." % file_name)
    logging.info("Proceso finalizado")
