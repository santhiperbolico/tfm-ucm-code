import argparse
from typing import Optional

from attr import attrs


@attrs(auto_attribs=True)
class Cluster:
    name: str
    radio_scale: float
    filter_parallax_min: Optional[float] = None
    filter_parallax_max: Optional[float] = None


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
