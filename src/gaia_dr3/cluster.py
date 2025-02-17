import logging
import os
from typing import Optional

import astropy.units as u
import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table.table import Table

from attr import attrs, attrib

from gaia_dr3.etls.download_data import get_object_from_heasarc, get_skycoords
from gaia_dr3.etls.stadistics import get_uwe_from_gaia
from gaia_dr3.etls.utils import convert_mas_yr_in_km_s, get_l_b_velocities
from ruwe_calculation.ruwetools import U0Interpolator
from catalogs import Catalog, CatalogsType


@attrs
class Cluster:

    name = attrib(init=True, type=str)
    info = attrs(init=True, type=Table)
    coord = attrib(init=True, type=SkyCoord)

    data = attrib(init=False, type=pd.DataFrame)

    @classmethod
    def get_cluster(cls, name: str) -> "Cluster":
        """
        Método de clase que dado el nombre del cluster se obtiene las coordenadas
        del objeto.

        Parameters
        ----------
        name: str
            Nombre del cluster

        Returns
        -------
        object: Cluster
            Elemento Cluster instanciado.
        """
        result_cluster = get_object_from_heasarc(name)
        coord = get_skycoords(result_cluster, u.deg, u.deg)
        return cls(name, result_cluster, coord)

    def download_cluster(
            self,
            catalog_name: CatalogsType,
            radius_scale: float = 1,
            radius_type: Optional[str] = None,
            filter_parallax_min: Optional[float] = None,
            filter_parallax_max: Optional[float] = None,
            filter_parallax_error: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Método que descr
        Parameters
        ----------
        catalog_name: CatalogsType
            Nombre del tipo del catálogo.
        radius_scale: float, default 1
            Escala del radio de búsqueda
        radius_type: Optional[str], default None
            Tipo del radio de búsqueda "core_radius", "half_light_radius",
            "vision_fold_radius". Por defecto es "vision_fold_radius".
        filter_parallax_min: Optional[float], default None
            Paralaje mínimo que buscar.
        filter_parallax_max: Optional[float], default None
            Paralaje máximo que buscar.
        filter_parallax_error: Optional[float], default None
            Máximo error en el parale a buscar.

       Returns
        -------
        results: pd.DataFrame
            Tabla con los datos del catálogo descargados.
        """
        ra = self.coord.ra.value  # Ascensión recta (RA) en grados
        dec = self.coord.dec.value  # Declinación (Dec) en grados

        radius = None
        if radius_type is None:
            radius_type = "vision_fold_radius"

        if radius_type == "vision_fold_radius":
            radius = np.power(
                10,
                self.info["CENTRAL_CONCENTRATION"].value[0]
            ) * self.info["CORE_RADIUS"].value[0] / 60
        if radius_type == "core_radius":
            radius = radius_scale * self.info["CORE_RADIUS"][0] * u.arcmin
        if radius_type == "half_light_radius":
            radius = radius_scale * self.info["HALF_LIGHT_RADIUS"][0] * u.arcmin

        if radius is None:
            raise ValueError(
                'El tipo de radio no es correcto, pruebe con "core_radius", '
                '"half_light_radius" o "vision_fold_radius"')

        catalog = Catalog.get_catalog(catalog_name)
        r_scale = f"r{radius_scale:1.f}".replace(".", "_")
        output_file = f"{catalog_name}-{self.name}-{r_scale}.vot"
        results = catalog.download_results(
            ra=ra,
            dec=dec,
            radius=radius,
            output_file=output_file,
            filter_parallax_min=filter_parallax_min,
            filter_parallax_max=filter_parallax_max,
            filter_parallax_error=filter_parallax_error
        )
        logging.info(f"Se encontraron {len(results)} fuentes en el radio de búsqueda:")
        self.data = results
        return self.data

    def read_cluster(
            self,
            path: str,
            file_to_read: Optional[str]=None,
            catalog_name: Optional[CatalogsType]=None,
            radius_scale: Optional[float]=None
    ) -> pd.DataFrame:
        """
        Método que lee los datos del cluster de un archivo del catálogo.

        Parameters
        ----------
        path: str
            Ruta donde se encuentra el archivo.
        file_to_read: Optional[str], default None.
            Nombre del archivo. Si es None se tiene que indicar el resto de parámetros-
        catalog_name: Optional[CatalogsType], default None
            Nombre del catálogo. Se utiliza para construir el nombre dle archivo
        radius_scale: Optional[float], default None
            Escala del radio de búsqueda. Se utiliza para construir el nombre dle archivo.

        Returns
        -------
        results: pd.DataFrame
            Tabla con los datos del catálogo descargados.

        """

        if file_to_read is None:
            r_scale = f"r{radius_scale:1.f}".replace(".", "_")
            file_to_read = f"{catalog_name}-{self.name}-{r_scale}.vot"
        results = Catalog.read_catalog(os.path.join(path, file_to_read))
        self.data = results
        return self.data

    def copy_set_extra_metrics(self) -> pd.DataFrame:
        """
        Método que copia los datos originales y calcula métricas extra como los movimientos
        propios en km por segundo o el movimiento propio en coordenadas galacticas.

        Returns
        -------
        df_r: pd.DataFrame
            Datos copiados con las métricas extra.
        """
        rwi = U0Interpolator()

        df_r = self.data.copy()
        df_r["pmra_kms"] = convert_mas_yr_in_km_s(df_r["parallax"].values, df_r["pmra"].values)
        df_r["pmdec_kms"] = convert_mas_yr_in_km_s(df_r["parallax"].values, df_r["pmdec"].values)
        df_r["pm_kms"] = np.sqrt(df_r.pmra_kms ** 2 + df_r.pmdec_kms ** 2)
        df_r["pm"] = np.sqrt(df_r.pmra ** 2 + df_r.pmdec ** 2)
        df_r["pm_l"], df_r["pm_b"] = get_l_b_velocities(df_r.ra.values, df_r.dec.values,
                                                        df_r.pmra.values, df_r.pmdec.values)

        # Métricas de cualificación de la muestra
        df_r["uwe"] = get_uwe_from_gaia(df_r, 5)
        df_r["ruwe"] = df_r['uwe'] / rwi.get_u0(df_r['phot_g_mean_mag'], df_r['bp_rp'])
        return df_r


    def qualify_data(
            self,
            max_ruwe: float = 1.4,
            pmra_kms_min: Optional[float] =  None,
            pmdec_kms_min: Optional[float] = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Método que devuelve cuatro muestras de datos siguiendo los siguientes criterios. En todos
        ello se ha aplicado un filtro de calidad donde el RUWE es menor que max_ruwe
        - Todas las estrellas seleccionadas.
        - Las estrellas con errores de paralaje y pm menores al 10%.
        - Las estrellas con un error de paralaje menor del 30% y de pm menores al 10%.
        - Las estrellas con un error de paralaje menor del 10% y de pm menores al 20%.

        Parameters
        ----------
        max_ruwe: float
            Máximo RUWE aceptado en la muestra.
        pmra_kms_min: Optional[float], default None
            Filtro opcional. Mímimo valor para pmra en km/s.
        pmdec_kms_min: Optional[float], default None
            Filtro opcional. Mímimo valor para pmra en km/s.

        Returns
        -------
        dfr_c1: pd.DataFrame
            Todas las estrellas seleccionadas.
        dfr_c2: pd.DataFrame
            Las estrellas con errores de paralaje y pm menores al 10%.
        dfr_c3: pd.DataFrame
            Las estrellas con un error de paralaje menor del 30% y de pm menores al 10%.
        dfr_c4: pd.DataFrame
            Las estrellas con un error de paralaje menor del 10% y de pm menores al 20%.
        """
        dfr_c1 = self.copy_set_extra_metrics()
        mask_filter = dfr_c1.ruwe < max_ruwe
        if pmra_kms_min:
            mask_filter = mask_filter & (dfr_c1.pmra_kms > pmra_kms_min)
        if pmdec_kms_min:
            mask_filter = mask_filter & (dfr_c1.pmdec_kms > pmdec_kms_min)
        dfr_c1 = dfr_c1[mask_filter]

        dfr_c2 = dfr_c1[
            (dfr_c1.pmra_error < 0.10) & (dfr_c1.pmdec_error < 0.10) & 
            (dfr_c1.parallax_error < 0.10)]
        dfr_c3 = dfr_c1[
            (dfr_c1.pmra_error < 0.10) & (dfr_c1.pmdec_error < 0.10) & 
            (dfr_c1.parallax_error < 0.30)]
        dfr_c4 = dfr_c1[
            (dfr_c1.pmra_error < 0.30) & (dfr_c1.pmdec_error < 0.30) & 
            (dfr_c1.parallax_error < 0.10)]
        return dfr_c1, dfr_c2, dfr_c3, dfr_c4
