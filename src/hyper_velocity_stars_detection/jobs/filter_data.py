import logging
import os
import sys

from tqdm import tqdm

from hyper_velocity_stars_detection.jobs.utils import get_params
from hyper_velocity_stars_detection.sources.source import AstroMetricData, AstroObject

if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [%(asctime)s] %(message)s")
    args = get_params(sys.argv[1:])

    files_list = os.listdir(args.path)
    for file_name in tqdm(files_list, desc="Procesando elementos", unit="item"):
        logging.info(f"Procesando elemento {file_name}")
        if file_name[-4:] == ".vot":
            name = file_name.split("_")[1]
            radio_scale = int(file_name.split("_")[2].replace("r", "").replace(".vot", ""))

            astro_object = AstroObject.get_object(name)
            _ = astro_object.read_object(args.path, file_name)
            pm_kms = None
            if radio_scale > 1:
                pm_kms = args.pm_kms
            astro_data = AstroMetricData.load_data_from_object(
                astro_object, radio_scale, pmra_kms_min=pm_kms, pmdec_kms_min=pm_kms
            )
            path_temp = os.path.join(args.path, name)
            astro_data.save(path_temp)

        logging.info(f"{file_name} procesado.\n")
