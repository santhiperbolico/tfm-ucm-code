import argparse
import logging
import os
import sys
from tempfile import TemporaryDirectory

from google.cloud import storage
from tqdm import tqdm

from hyper_velocity_stars_detection.astrobjects import AstroObjectProject
from hyper_velocity_stars_detection.jobs.google_jobs.utils import download_from_gcs, load_project
from hyper_velocity_stars_detection.jobs.utils import ProjectDontExist, read_catalog_file


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


def fix_project_parallax(cluster_name: str, path: str) -> AstroObjectProject | None:
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

    for data in project.data_list:
        data.fix_parallax(warnings=False)
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
    dont_exist_projects = []

    for cluster_name in tqdm(
        selected_clusters, desc="Procesando clusters", unit="item", total=len(selected_clusters)
    ):
        with TemporaryDirectory() as temp_path:
            project = fix_project_parallax(cluster_name, temp_path)
            if project is None:
                logging.info("El proyecto %s no existe." % cluster_name)
                dont_exist_projects.append(cluster_name)
            else:
                file_name = f"{cluster_name}.zip"
                logging.info("-- Guardando los datos en  %s." % file_name)
                project.save_project(to_zip=True)

                logging.info("-- Subiendo datos a Storage %s." % file_name)
                file_path = os.path.join(temp_path, file_name)
                client = storage.Client(project=project_id)
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(file_name)
                blob.upload_from_filename(file_path)
                logging.info("-- Archivo %s subido a Storage ." % file_name)

    if dont_exist_projects:
        n_fails = len(dont_exist_projects)
        n_projects = len(selected_clusters)
        logging.info("No se han podido actualizar %i proyectos de %i" % (n_fails, n_projects))
        logging.info("Los proyectos que han fallado son: %s" % str(dont_exist_projects))

    logging.info("Proceso finalizado")
