from typing import Optional

from attr import attrs


@attrs(auto_attribs=True)
class Cluster:
    name: str
    radio_scale: float
    filter_parallax_min: Optional[float] = None
    filter_parallax_max: Optional[float] = None


PATH = "../data"

PM_KMS_MIN = 50

SELECTED_CLUSTERS = [
    Cluster("ngc 104", 6, 0, 1),
    Cluster("ngc 5139", 6, 0, 1),
    Cluster("ngc 6121", 6, 0, 1),
    Cluster("ngc 4472", 6, None, 1),
]
