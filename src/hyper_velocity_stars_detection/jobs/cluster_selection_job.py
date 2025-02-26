import logging
import os

from hyper_velocity_stars_detection.astrobjects import AstroObjectProject
from hyper_velocity_stars_detection.jobs.project_vars import PATH, SELECTED_CLUSTERS


class ProjectDontExist(Exception):
    pass


def download_data(cluster_name: str, path: str) -> AstroObjectProject:
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
    project: AstroObjectProject
        Proyecto donde se guardan los resultados.
    """
    path_project = os.path.join(path, cluster_name)
    if os.path.exists(path_project):
        zip_f = [file for file in os.listdir(path_project) if ".zip" == file[-4:]]
        if len(zip_f) > 0:
            project = AstroObjectProject.load_project(cluster_name, path)
            logging.info("Cargando primer catálogo")
            logging.info(str(project.data_list[0]))
            logging.info("Cargando segundo catálogo")
            logging.info(str(project.data_list[1]))
            return project
    raise ProjectDontExist("No hay datos descargados del proyecto")


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [%(asctime)s] %(message)s")
    for cluster in SELECTED_CLUSTERS:
        project = download_data(cluster_name=cluster.name, path=PATH)
