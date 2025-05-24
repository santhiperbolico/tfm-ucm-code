from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from attr import attrib, attrs
from scipy.stats import chi2
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import LocalOutlierFactor


@attrs
class NoiseMethod(ABC):
    noise_method: str = "noise_method"

    @abstractmethod
    def fit(self, x: np.ndarray, **kwargs) -> None:
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def fit_predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        pass


@attrs
class IsolationForestNoise(NoiseMethod):
    noise_method: str = "isolation_forest_method"

    model = attrib(type=IsolationForest, init=False, default=None)

    def fit(self, x: np.ndarray, **kwargs) -> None:
        self.model = IsolationForest(**kwargs)
        self.model.fit(x)
        return None

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)

    def fit_predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        self.model = IsolationForest(**kwargs)
        return self.model.fit_predict(x, **kwargs)


@attrs
class LocalOutlierNoise(NoiseMethod):
    noise_method: str = "local_outlier_method"

    model = attrib(type=LocalOutlierFactor, init=False, default=None)

    def fit(self, x: np.ndarray, **kwargs) -> None:
        self.model = LocalOutlierFactor(**kwargs)
        self.model.fit(x)
        return None

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)

    def fit_predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        self.model = LocalOutlierFactor(**kwargs)
        return self.model.fit_predict(x, **kwargs)


@attrs
class MinCovDetNoise(NoiseMethod):
    noise_method: str = "mcd_method"

    model = attrib(type=MinCovDet, init=False, default=None)
    _mu = attrib(type=np.ndarray, init=False, default=None)
    _s = attrib(type=np.ndarray, init=False, default=None)

    def fit(self, x: np.ndarray, **kwargs) -> None:
        self.model = MinCovDet(**kwargs)
        self.model.fit(x)
        self._mu = self.model.location_.reshape(1, -1)
        self._s = self.model.covariance_
        return None

    def predict(self, x: np.ndarray) -> np.ndarray:
        sinv = np.linalg.inv(self._s)
        dm = pairwise_distances(x, self._mu, metric="mahalanobis", VI=sinv)
        cutoff = np.sqrt(chi2.ppf(0.975, df=x.shape[1]))
        prediction = np.ones(x.shape[0])
        prediction[dm > cutoff] = -1
        return prediction

    def fit_predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        self.fit(x, **kwargs)
        return self.predict(x)


def get_noise_method(method: str | None, params: Optional[dict] = None) -> NoiseMethod | None:
    """
    Función que nos devuelve el método de eliminación de outliers.

    Parameters
    ----------
    method: str | None
        Nombre del método a utilizar. Si es None devuelve None.
    params: Optional[dict] = None
        Parámetros del método de eliminación de outliers.

    Returns
    -------
    noise: ScalersMethods
        Modelo de eliminación de outliers
    """
    if method is None:
        return None

    if params is None:
        params = {}

    methods = {
        IsolationForestNoise.noise_method: IsolationForestNoise,
        LocalOutlierNoise.noise_method: LocalOutlierNoise,
        MinCovDetNoise.noise_method: MinCovDetNoise,
    }
    try:
        return methods[method](**params)
    except KeyError:
        raise ValueError(f"No existe el método {method}, prueba con : {list(methods.keys())}")
