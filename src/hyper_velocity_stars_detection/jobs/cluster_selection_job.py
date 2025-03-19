import logging
import os
import sys

from hyper_velocity_stars_detection.jobs.utils import get_params, load_project, read_catalog_file

if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [%(asctime)s] %(message)s")
    args = get_params(sys.argv[1:])
    selected_clusters = read_catalog_file(os.path.join(args.path, "mwgc.dat.txt"))
    for cluster in selected_clusters:
        project = load_project(cluster_name=cluster.name, path=args.path)
        logging.info(str(project))
        logging.info("-- Descargando curvas de luz")
        project.xsource.download_light_curves(project.path_project)
        project.xsource.save()
