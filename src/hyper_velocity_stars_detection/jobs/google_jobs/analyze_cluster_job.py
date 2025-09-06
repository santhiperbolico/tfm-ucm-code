import argparse
import logging
import os
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from google.cloud import storage
from tqdm import tqdm

from hyper_velocity_stars_detection.globular_clusters import GlobularClusterAnalysis
from hyper_velocity_stars_detection.jobs.google_jobs.utils import load_globular_cluster
from hyper_velocity_stars_detection.jobs.utils import (
    ProjectDontExist,
    read_clusters_harris_catalog,
)
from hyper_velocity_stars_detection.sources.source import AstroObject
from hyper_velocity_stars_detection.tools.hvs_estimation import get_hvs_candidates
from hyper_velocity_stars_detection.tools.imbh_mass_estimation import v_ejections_sample


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
    parser.add_argument("--data_path", default="report/gc_clusters/")
    parser.add_argument("--path", default="report/analysis/")
    return parser.parse_args(argv)


def get_v_ejection_limit(
    df_data: pd.DataFrame,
    quantil: float = 0.5,
    a_space: tuple[float, float] = (0.05, 0.5),
    m_bin_space: tuple[float, float] = (0.6, 1.7),
    sigma_f: float = 0.3,
    size: int = 20000,
    seed: int = 123,
) -> float:
    """
    Función que calcula el límite de velocidad de eyección.

    Parameters
    ----------
    df_data: pd.DataFrame,
        Tabla con los datos del cluster
    quantil: float = 0.5,
        Cuantil usado para extraer la velocidad límite de la muestra generada.
    a_space: tuple[float, float] = (0.05,0.5),
        Intervalo de búsqueda en el eje semimayo del sistema binario.
    m_bin_space: tuple[float, float] = (0.6, 1.7),
        Intervalo de búsqueda en la masa total del sistema binario.
    sigma_f: float = 0.3,
        Proporción de la desviación según el trabajo de Fragione 2018.
    size: int = 20000
        Tamaño de la muestra
    seed: int = 123
    Semilla para generar la muestra.

    Returns
    -------
    v_ejection_limit: float
        Límite de la velocidad de eyección.
    """
    df_gc = df_data.copy()
    df_gc["pmra_kms"] = df_gc["pmra_kms"] - df_gc["pmra_kms"].mean()
    df_gc["pmdec_kms"] = df_gc["pmdec_kms"] - df_gc["pmdec_kms"].mean()
    df_gc["radial_velocity"] = df_gc["radial_velocity"] - df_gc["radial_velocity"].mean()

    df_gc["v_tan"] = np.sqrt(df_gc.pmra_kms**2 + df_gc.pmdec_kms**2)
    df_gc["v_3d"] = np.sqrt(df_gc.v_tan**2 + df_gc.radial_velocity**2)
    sigma_v = df_gc["v_3d"].std()

    v_ej_sample = v_ejections_sample(
        a_space=a_space,
        m_bin_space=m_bin_space,
        sigma_kms=sigma_v,
        sigma_f=sigma_f,
        size=size,
        seed=seed,
    )
    return np.quantile(v_ej_sample, quantil)


def extract_globular_cluster(
    cluster_name: str, selected_clusters: list[str]
) -> GlobularClusterAnalysis | None:
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
    project: GlobularClusterAnalysis | None
        Si se detecta que el cúmulo globular ha sido estudiado se devuelve el
        proyecto seleccionado. En caso contrario devuelve None.
    """
    project = None
    if cluster_name in selected_clusters:
        try:
            project = load_globular_cluster(
                cluster_name=cluster_name,
                project_id=os.getenv("PROJECT_ID"),
                bucket_name=os.getenv("BUCKET"),
                path=args.data_path,
            )
        except ProjectDontExist:
            return None
        if project is not None:
            try:
                _ = project.clustering_results.gc
            except Exception:
                return None
    return project


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
    blob_path = os.path.join(path_bucket, file_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(file_path)


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [%(asctime)s] %(message)s")
    args = get_params(sys.argv[1:])

    project_id = os.getenv("PROJECT_ID")
    bucket_name = os.getenv("BUCKET")

    selected_clusters = read_clusters_harris_catalog()

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
            "XS_count",
            "XS_Distance",
        ]
    )
    selected_clusters = [cl.name for cl in selected_clusters]
    index = 0

    project_dont_exist = []

    other_error = []

    with TemporaryDirectory() as temp:
        for cluster_name in tqdm(
            selected_clusters,
            desc="Procesando clusters",
            unit="item",
            total=len(selected_clusters),
        ):
            gc_object = extract_globular_cluster(cluster_name, selected_clusters)
            if gc_object is not None:
                df_gc = gc_object.clustering_results.remove_outliers_gc()
                gc_pm = np.array([df_gc.pmra.mean(), df_gc.pmdec.mean()])
                v_limit = get_v_ejection_limit(df_gc)
                sample_label = gc_object.astro_data.get_data_max_samples(5000)
                hvs = get_hvs_candidates(
                    gc_object=gc_object, v_kms_limit=v_limit, sample_label=sample_label
                )

                hvs_distance = get_distance_from_center(gc_object.astro_data.astro_object, hvs)
                hvs_pm = hvs[["pmra", "pmdec"]].values
                xsource = gc_object.xrsource.data[
                    gc_object.xrsource.data.main_id == gc_object.name
                ]
                xsource_distance = get_distance_from_center(
                    gc_object.astro_data.astro_object, xsource
                )

                hvs_cos_theta = (gc_pm * hvs_pm).sum(axis=1) / (
                    np.linalg.norm(hvs_pm, axis=1) * np.linalg.norm(gc_pm)
                )

                df_results.loc[index] = (
                    cluster_name,
                    df_gc.shape[0],
                    df_gc.parallax.mean(),
                    df_gc.parallax.std(),
                    df_gc.pmra.mean(),
                    df_gc.pmra.std(),
                    df_gc.pmdec.mean(),
                    df_gc.pmdec.std(),
                    df_gc.radial_velocity.mean(),
                    df_gc.radial_velocity.std(),
                    hvs.shape[0],
                    hvs.pmra.mean(),
                    hvs.pmra.std(),
                    hvs.pmdec.mean(),
                    hvs.pmdec.std(),
                    hvs_distance,
                    hvs_cos_theta.mean(),
                    xsource.shape[0],
                    xsource_distance,
                )
                _ = gc_object.plot_cmd(
                    highlight_stars=hvs,
                    legend=True,
                    clusters=gc_object.clustering_results.main_label,
                    path=temp,
                )

                _ = gc_object.plot_cluster(
                    highlight_stars=hvs, random_state=123, legend=True, factor_size=50, path=temp
                )
                catalog_name = gc_object.astro_data.catalog.catalog_name
                main_id = gc_object.name
                files = [
                    f"cluster_{catalog_name}_{main_id}_r_1.png",
                    f"cmd_{catalog_name}_{main_id}_r_1.png",
                ]
                for file_name in files:
                    save_results(
                        file_name=file_name,
                        path=temp,
                        path_bucket=args.path,
                        project_id=project_id,
                        bucket_name=bucket_name,
                    )
                index += 1

        logging.info("Guardando los resultados")
        df_results.to_csv(os.path.join(temp, "raw_results_dr3.csv"))

        for file_name in ["raw_results_dr3.csv"]:
            save_results(
                file_name=file_name,
                path=temp,
                path_bucket=args.path,
                project_id=project_id,
                bucket_name=bucket_name,
            )

    logging.info("Datos Guardados.")
