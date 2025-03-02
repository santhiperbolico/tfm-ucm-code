from attr import attrs


@attrs(auto_attribs=True)
class Cluster:
    name: str
    radio_scale: float
    filter_parallax_max: float = None


PATH = "../data"

PM_KMS_MIN = 50

SELECTED_CLUSTERS = [
    Cluster("ngc 104", 6, 1),
    Cluster("ngc 5139", 6, 1),
    Cluster("ngc 6121", 6, 1),
    Cluster("m 49", 6, 1),
]
