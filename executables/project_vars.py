import os
from typing import Optional

from attr import attrs

from hyper_velocity_stars_detection.utils import read_catalog_file


@attrs(auto_attribs=True)
class Cluster:
    name: str
    radio_scale: float
    filter_parallax_min: Optional[float] = None
    filter_parallax_max: Optional[float] = None


PATH = "/data/"

PM_KMS_MIN = 50


SELECTED_CLUSTERS = read_catalog_file(os.path.join(PATH, "mwgc.dat.txt"))

PROJECT_ID = "hvs-detection-imbh"
BUCKET = "globular_cluster_data"
