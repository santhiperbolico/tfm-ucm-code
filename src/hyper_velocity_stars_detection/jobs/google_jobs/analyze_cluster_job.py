import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
from google.cloud import storage
from tqdm import tqdm

from hyper_velocity_stars_detection.astrobjects import AstroObject, AstroObjectProject
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


def extract_globular_cluster(
    cluster_name: str, selected_clusters: list[str]
) -> AstroObjectProject | None:
    """
    Función que extrae en el caso de que corresponda las estrellas del cúmulo globular
    encontrado por el método.

    Parameters
    ----------
    cluster_name: str
        Nombre del cúmulo globular
    selected_clusters: list[str]
        Lista de cúmulos a analizar.

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
                path=args.path,
            )
        except ProjectDontExist:
            return None
        if project is not None:
            try:
                _ = project.clustering_results.gc
            except Exception:
                return None
    return project


def get_clean_results(df_results: pd.DataFrame, df_clusters_dr2: pd.DataFrame) -> pd.DataFrame:
    """
    Función que combina los resultados con el DR2 y aplicamos filtros para quedarnos con
    los clusters cuyos resultados en DR2 estén dentro del margen de error.

    Parameters
    ----------
    df_results: pd.DataFrame
        Tabla resumen con lso datos del DR3.
    df_clusters_dr2: pd.DataFrame
        Datos del DR2 descargados.

    Returns
    -------
    clean_data: pd.DataFrame
        Datos filtrados y combiandos con el DR2.

    """

    clean_data = pd.merge(df_results, df_clusters_dr2, on="Name", suffixes=["", "_dr2"])
    mask_p = (clean_data.parallax_dr2 >= clean_data.parallax - clean_data.e_parallax) & (
        clean_data.parallax_dr2 <= clean_data.parallax + clean_data.e_parallax
    )
    mask_pmra = (clean_data.pmRA__dr2 >= clean_data.pmRA_ - clean_data.e_pmRA_) & (
        clean_data.pmRA__dr2 <= clean_data.pmRA_ + clean_data.e_pmRA_
    )
    mask_pmde = (clean_data.pmDE_dr2 >= clean_data.pmDE - clean_data.e_pmDE) & (
        clean_data.pmDE_dr2 <= clean_data.pmDE + clean_data.e_pmDE
    )

    mask_ml = ~clean_data.M_L.isna()
    clean_data = clean_data[mask_p & mask_pmra & mask_pmde & mask_ml]
    return clean_data


def get_distance_from_center(astro_object: AstroObject, data: pd.DataFrame) -> float:
    """
    Función que calcula la distancia media de los objetos de data del centro.

    Parameters
    ----------
    astro_object: AstroObject
        Objeto astronómico.
    data: pd.DataFrame
        Elementos a medir, deben tener las columnas ra y dec.

    Returns
    -------
    distance: float
        Distancia media de los elementos data del centro de astro_object en mas.
    """
    ra_center = astro_object.coord.ra.value
    dec_center = astro_object.coord.dec.value
    distance = np.sqrt((ra_center - data.ra.values) ** 2 + (dec_center - data.dec.values) ** 2)
    return distance.mean()


def save_results(
    file_name: str, path: str, path_bucket: str, project_id: str, bucket_name: str
) -> None:
    """
    Función que guarda los resultados en formato csv en Storage.

    Parameters
    ----------
    file_name: str
        Nombre del archivo
    path: str
        Ruta del archivo en local.
    path_bucket: str
        Ruta de guardado en el bucket.
    project_id: str
        ID del proyecto de GCP
    bucket_name: str
        Nombre del bucket.
    """
    logging.info("Subiendo datos a Storage %s." % file_name)
    file_path = os.path.join(path, file_name)
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    blob_path = f"{path_bucket}/{file_name}"
    blob = bucket.blob(blob_path)
    # Subir el archivo
    blob.upload_from_filename(file_path)


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
            "N_count",
            "parallax",
            "e_parallax",
            "pmRA_",
            "e_pmRA_",
            "pmDE",
            "e_pmDE",
            "RV",
            "e_RV",
            "HVS_count",
            "HVS_pmRA_",
            "e_HVS_pmRA_",
            "HVS_pmDE",
            "e_HVS_pmDE",
            "HVS_Distance",
            "HVS_cosTheta",
            "HVS_2_count",
            "HVS_2_pmRA_",
            "e_HVS_2_pmRA_",
            "HVS_2_pmDE",
            "e_HVS_2_pmDE",
            "HVS_2_Distance",
            "HVS_2_cosTheta",
            "XS_count",
            "XS_Distance",
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
        project = extract_globular_cluster(cluster_name, selected_clusters)
        if project is not None:
            gc = project.clustering_results.remove_outliers_gc()
            gc_pm = np.array([gc.pmra.mean(), gc.pmdec.mean()])
            hvs = project.clustering_results.selected_hvs(
                df_hvs_candidates=project.get_data("df_6_c1"),
                factor_sigma=1.0,
                hvs_pm=150,
            )
            hvs_2 = project.clustering_results.selected_hvs(
                df_hvs_candidates=project.get_data("df_6_c1"),
                factor_sigma=2.0,
                hvs_pm=150,
            )
            hvs_distance = get_distance_from_center(project.astro_object, hvs)
            hvs_distance_2 = get_distance_from_center(project.astro_object, hvs_2)
            hvs_pm = hvs[["pmra", "pmdec"]].values
            hvs_pm_2 = hvs_2[["pmra", "pmdec"]].values
            xsource = project.xsource.results[project.xsource.results.main_id == project.name]
            xsource_distance = get_distance_from_center(project.astro_object, xsource)

            hvs_cos_theta = (gc_pm * hvs_pm).sum(axis=1) / (
                np.linalg.norm(hvs_pm, axis=1) * np.linalg.norm(gc_pm)
            )
            hvs_cos_theta_2 = (gc_pm * hvs_pm_2).sum(axis=1) / (
                np.linalg.norm(hvs_pm_2, axis=1) * np.linalg.norm(gc_pm)
            )
            df_results.loc[index] = (
                cluster_name,
                gc.shape[0],
                gc.parallax_corrected.mean(),
                gc.parallax.std(),
                gc.pmra.mean(),
                gc.pmra.std(),
                gc.pmdec.mean(),
                gc.pmdec.std(),
                gc.radial_velocity.mean(),
                gc.radial_velocity.std(),
                hvs.shape[0],
                hvs.pmra.mean(),
                hvs.pmra.std(),
                hvs.pmdec.mean(),
                hvs.pmdec.std(),
                hvs_distance,
                hvs_cos_theta.mean(),
                hvs_2.shape[0],
                hvs_2.pmra.mean(),
                hvs_2.pmra.std(),
                hvs_2.pmdec.mean(),
                hvs_2.pmdec.std(),
                hvs_distance_2,
                hvs_cos_theta_2.mean(),
                xsource.shape[0],
                xsource_distance,
            )
            _ = project.plot_cmd(
                hvs_candidates_name="df_6_c1", factor_sigma=1.0, hvs_pm=150, legend=True
            )

            _ = project.plot_cluster(
                hvs_candidates_name="df_6_c1",
                factor_sigma=1,
                hvs_pm=150,
                legend=True,
                factor_size=50,
            )
            _ = project.plot_cmd(
                hvs_candidates_name="df_6_c1", factor_sigma=2.0, hvs_pm=150, legend=True
            )

            _ = project.plot_cluster(
                hvs_candidates_name="df_6_c1",
                factor_sigma=2,
                hvs_pm=150,
                legend=True,
                factor_size=50,
            )
            files = [
                "cluster_df_6_c1_hvs_150_sigma_1.png",
                "cluster_df_6_c1_hvs_150_sigma_2.png",
                "cmd_hvs_150_sigma_1.png",
                "cmd_hvs_150_sigma_2.png",
            ]
            for file_name in files:
                save_results(
                    file_name=file_name,
                    path=os.path.join(args.path, cluster_name),
                    path_bucket=cluster_name,
                    project_id=project_id,
                    bucket_name=bucket_name,
                )
            index += 1

    logging.info("Guardando los resultados")
    df_results.to_csv(os.path.join(args.path, "raw_results_dr3.csv"))
    clean_results = get_clean_results(df_results, df_clusters_dr2)
    clean_results.to_csv(os.path.join(args.path, "clean_results_dr3.csv"))

    for file_name in ["raw_results_dr3.csv", "clean_results_dr3.csv"]:
        save_results(
            file_name=file_name,
            path=args.path,
            path_bucket="results",
            project_id=project_id,
            bucket_name=bucket_name,
        )

    logging.info("Datos Guardados.")
