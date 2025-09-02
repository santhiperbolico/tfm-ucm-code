import argparse
import gc
import logging
import os.path
from typing import Optional

import pandas as pd
from attr import attrs

from hyper_velocity_stars_detection.globular_clusters import GlobularClusterAnalysis
from hyper_velocity_stars_detection.sources.catalogs import GaiaDR3
from hyper_velocity_stars_detection.sources.source import AstroObject
from hyper_velocity_stars_detection.sources.utils import get_main_id, get_vizier_catalog


class ProjectDontExist(Exception):
    pass


class DefaultParamsClusteringDetection:
    data_name = "df_1_c2"
    columns = ["pmra", "pmdec", "parallax_corrected"]
    columns_to_clus = ["pmra", "pmdec", "parallax_corrected", "bp_rp", "phot_g_mean_mag"]
    max_cluster = 5
    n_trials = 500

    @property
    def params(self) -> dict[str, str | list[str] | int]:
        return {
            "data_name": self.data_name,
            "columns": self.columns,
            "columns_to_clus": self.columns_to_clus,
            "max_cluster": self.max_cluster,
            "n_trials": self.n_trials,
        }


@attrs(auto_attribs=True)
class Cluster:
    name: str
    radius: float
    radio_scale: float


def load_project(cluster_name: str, path: str, from_zip: bool = False) -> GlobularClusterAnalysis:
    """
    Función que descarga o carga desde la cache los datos limpios en un proyecto.

    Parameters
    ----------
    cluster_name: str,
        Nombre del cluster
    path: str
        Ruta donde se quiere guardar los archivos.

    Returns
    -------
    project: GlobularClusterAnalysis
        Proyecto donde se guardan los resultados.
    """
    path_project = os.path.join(path, cluster_name)
    if from_zip and os.path.exists(path_project + ".zip"):
        project = GlobularClusterAnalysis.load(path)
        logging.info("Cargando primer catálogo")
        logging.info(str(project))
        return project

    if os.path.exists(path_project):
        zip_f = [file for file in os.listdir(path_project) if ".zip" == file[-4:]]
        if len(zip_f) > 0:
            project = GlobularClusterAnalysis.load(path)
            logging.info("Cargando primer catálogo")
            logging.info(str(project))
            return project

    raise ProjectDontExist("No hay datos descargados del proyecto")


def download_object(
    astro_object: AstroObject,
    read_from_cache: bool,
    path: str,
    radio_scale: float,
    catalog_name: str = GaiaDR3.catalog_name,
    filter_parallax_min: Optional[float] = None,
    filter_parallax_max: Optional[float] = None,
    filter_parallax_error: float = 0.30,
) -> AstroObject:
    """
    Función que descarga los datos del cluster.

    Parameters
    ----------
    astro_object: AstroObject
        Cluster.
    read_from_cache: bool
        Indica si queremos buscar los datos desde la cache.
    path: str
        Ruta donde se quiere guardar los archivos.
    radio_scale: float, default 1
        Escala del radio de búsqueda
    catalog_name: CatalogsType
             Nombre del tipo del catálogo.
    filter_parallax_min: Optional[float], default None
        Paralaje mínimo que buscar.
    filter_parallax_max: Optional[float], default None
        Paralaje máximo que buscar.
    filter_parallax_error: Optional[float], default None
        Máximo error en el parale a buscar.

    Returns
    -------
    astro_object: AstroObject
        Cluster con los datos descargados.

    """
    result = None
    if read_from_cache:
        r_scale = f"r{radio_scale:.0f}"
        file_to_read = f"{catalog_name}_{astro_object.name}_{r_scale}.vot"
        file = os.path.join(path, file_to_read)
        if os.path.isfile(file):
            logging.info(
                "\t - El archivo está descargado, si quieres descargarlo de "
                "nuevo borra el antiguo."
            )
            return astro_object
        logging.info("\t - No hay archivos para cargar en la cache. Se van a descargar los datos.")

    if not isinstance(result, pd.DataFrame):
        _ = astro_object.download_data(
            catalog_name=catalog_name,
            radius_scale=radio_scale,
            path=path,
            filter_parallax_min=filter_parallax_min,
            filter_parallax_max=filter_parallax_max,
            filter_parallax_error=filter_parallax_error,
            return_data=False,
        )
    return astro_object


def download_astro_data(
    cluster_name: str,
    read_from_cache: bool,
    path: str,
    radio_scale: float | list[float],
    catalog_name: str = GaiaDR3.catalog_name,
    filter_parallax_min: float = None,
    filter_parallax_max: Optional[float] = None,
    filter_parallax_error: float = 0.30,
) -> None:
    """
    Función que descarga o carga desde la cache los datos limpios en un proyecto.

    Parameters
    ----------
    cluster_name: str,
        Nombre del cluster
    read_from_cache: bool
        Indica si queremos buscar los datos desde la cache.
    path: str
        Ruta donde se quiere guardar los archivos.
    radio_scale: float | list[float], default 1
        Escala del radio de búsqueda. Si se pasa una lista se descarga la lista de radios
        indicado.
    catalog_name: CatalogsType
             Nombre del tipo del catálogo.
    filter_parallax_min: Optional[float], default None
        Paralaje mínimo que buscar.
    filter_parallax_max: Optional[float], default None
        Paralaje máximo que buscar.
    filter_parallax_error: Optional[float], default None
        Máximo error en el parale a buscar.
    max_ruwe: float
        Máximo RUWE aceptado en la muestra.
    pmra_kms_min: Optional[float], default None
        Filtro opcional. Mímimo valor para pmra en km/s.
    pmdec_kms_min: Optional[float], default None
        Filtro opcional. Mímimo valor para pmra en km/s.
    """
    radio_list = [1, radio_scale]
    if isinstance(radio_scale, list):
        radio_list = radio_scale

    params_download_cluster = {
        "filter_parallax_min": filter_parallax_min,
        "filter_parallax_max": filter_parallax_max,
        "filter_parallax_error": filter_parallax_error,
    }
    params_default = {
        "read_from_cache": read_from_cache,
        "path": path,
        "catalog_name": catalog_name,
    }

    for radio in radio_list:
        logging.info(f"Descargando {cluster_name} para r_scale {radio}")
        astro_object = AstroObject.get_object(cluster_name)
        logging.info("Objeto seleccionado: \n" + astro_object.info.to_pandas().to_string())
        params_download = {"astro_object": astro_object, "radio_scale": radio}
        params_download.update(params_default)
        if radio > 1:
            params_download.update(params_download_cluster)
        _ = download_object(**params_download)
    gc.collect()


def read_harris_catalog() -> pd.DataFrame:
    """
    Función que lee "CATALOG OF PARAMETERS FOR MILKY WAY GLOBULAR CLUSTERS"
    de Harris, W.E. 1996, AJ, 112, 1487

    Returns
    -------
    df_catalog: pd.DataFrame
        Datos del catálogo de Harris, W.E. 1996, AJ, 112, 1487
    """
    df_catalog = get_vizier_catalog("VII/202/catalog")[0]
    df_catalog.ID = df_catalog.ID.str.lower()
    return df_catalog


def read_clusters_harris_catalog(
    radius_scale: float = 1.0, radius_type: Optional[str] = None
) -> list[Cluster]:
    """
    Función que lee "CATALOG OF PARAMETERS FOR MILKY WAY GLOBULAR CLUSTERS"
    de Harris, W.E. 1996, AJ, 112, 1487 devolviendo una lista de clusters.

    Parameters
    ----------
    radius_scale: float = 1.0
        Escala del radio de búsqueda. Si radius type es None se usa el campo de visión.
    radius_type: Optional[str] = None
        Indica que tipo de radio se quiere usar, radio_core "r_c" o halfĺight readio "r_l". Si
        es None se usa el camp de visión ocupado por el cluster.

    Returns
    -------
    clusters_list: list[Cluster]
        Lista de objetos cluster.
    """
    df_catalog = read_harris_catalog()
    clusters_list = []
    if radius_type:
        if radius_type not in ("r_c", "r_l"):
            raise ValueError("No está implementado el radio %s, usa 'r_c' o 'r_l'" % radius_type)
    for index in range(df_catalog.shape[0]):
        row_cluster = df_catalog.loc[index]
        name = row_cluster.ID
        radius = None
        if radius_type:
            # El dato debe de estar en grados.
            radius = row_cluster[radius_type] * radius_scale / 60
        cluster = Cluster(name=name, radius=radius, radio_scale=radius_scale)
        clusters_list.append(cluster)
    return clusters_list


def read_baumgardt_catalog() -> pd.DataFrame:
    """
    Función que lee "Galactic GC mean proper motions & velocities (Baumgardt+, 2019)"
    de Baumgardt, H., Hilker, M., Sollima, A., & Bellini, A. (2019). Mean proper motions,
    space orbits, and velocity dispersion profiles of Galactic globular clusters derived
    from Gaia DR2 data. Monthly Notices of the Royal Astronomical Society, 482(4),
    5138-5155.

    Returns
    -------
    clusters_dr2: pd.DataFrame
        Datos del catálogo con datos de GaiaDR2.
    """
    clusters_dr2 = get_vizier_catalog(
        catalog="J/MNRAS/482/5138",
        columns=[
            "Name",
            "Rsun",
            "e_Rsun",
            "pmRA*",
            "e_pmRA*",
            "pmDE",
            "e_pmDE",
            "RV",
            "e_RV",
            "rho",
        ],
    )[0]

    clusters_dr2.Name = clusters_dr2.Name.str.lower()
    clusters_dr2["parallax"] = 1 / clusters_dr2.Rsun
    clusters_dr2["e_parallax"] = (
        1 / (clusters_dr2.Rsun - clusters_dr2.e_Rsun)
        - 1 / (clusters_dr2.Rsun + clusters_dr2.e_Rsun)
    ) / 2
    clusters_dr2 = clusters_dr2[
        [
            "Name",
            "parallax",
            "e_parallax",
            "pmRA_",
            "e_pmRA_",
            "pmDE",
            "e_pmDE",
            "RV",
            "e_RV",
            "rho",
            "Rsun",
            "e_Rsun",
        ]
    ]
    clusters_dr2["MAIN_ID"] = clusters_dr2["Name"].apply(get_main_id)
    return clusters_dr2


def read_catalog_file(filepath: str) -> list[Cluster]:
    """
    Función que lee los nombres de los cluster d eun acatálogo dado.

    Parameters
    ----------
    filepath: str
        Ruta del archivo

    Returns
    -------
    clusters_list: list[Cluster]
        Lista de objetos cluster.
    """
    # Leer el archivo
    with open(filepath, "r") as f:
        lines = f.readlines()
    # Encontrar el inicio de la tabla buscando la cabecera
    start_idx = 0
    for i, line in enumerate(lines):
        if "ID" in line and "Name" in line:  # Línea con nombres de columnas
            start_idx = i + 1  # Los datos comienzan en la siguiente línea
            break

    # Extraer los datos desde la tabla hasta la siguiente sección
    rows = []
    for line in lines[start_idx:]:
        if "_" in line:  # Línea de separación de secciones
            break

        # Leer las columnas según el formato de la tabla
        data = (
            line[:12].strip(),
            line[12:26].strip(),
            line[26:38].strip(),
            line[38:50].strip(),
        )
        data += (
            line[50:58].strip(),
            line[58:66].strip(),
            line[66:74].strip(),
            line[74:82].strip(),
        )
        data += line[82:90].strip(), line[90:98].strip(), line[98:106].strip()

        rows.append(data)

    # Crear un DataFrame de Pandas
    columns = [
        "ID",
        "Name",
        "RA (J2000)",
        "DEC (J2000)",
        "L (deg)",
        "B (deg)",
        "R_Sun (kpc)",
        "R_GC (kpc)",
        "X (kpc)",
        "Y (kpc)",
        "Z (kpc)",
    ]
    df = pd.DataFrame(rows, columns=columns)
    names = df.ID.str.lower().to_list()
    clusters_list = []
    for name in names:
        if name != "":
            cluster = Cluster(name, 6, 0, 1)
            clusters_list.append(cluster)
    return clusters_list


def get_argument_parser() -> argparse.ArgumentParser:
    """
    Función que genera los argumentos de download_data.

    Returns
    -------
    parser: argparse.ArgumentParser
        Parser con los argumentos del job.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data/globular_clusters/")
    parser.add_argument("--raw_folder", default="raw_data/")
    parser.add_argument("--pm_kms", default=50)
    return parser


def get_params(argv: list[str]) -> argparse.Namespace:
    """
    Función de preprocesado de los argumentos del job de MMM para devolver un
    objeto MarketingMixProject con los parámetros del proceso.

    Parameters
    ----------
    argv: list[str]
        Lista de argumentos del job.

    Returns
    -------
    args: argparse.Namespace
        Argumentos.
    """
    args = get_argument_parser().parse_args(argv)
    return args
