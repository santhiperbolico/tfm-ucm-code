import os
import pickle
import shutil
from abc import ABC, abstractmethod
from tempfile import TemporaryDirectory
from typing import Any, Dict, Mapping, Optional
from zipfile import ZipFile

import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table
from attr import attrs


class InvalidFileFormat(RuntimeError):
    """
    Error que se lanza cuando un archivo no tiene el formato esperado.
    """

    pass


class StorageObject(ABC):
    @staticmethod
    @abstractmethod
    def save(path: str, value: Any) -> None:
        pass

    @staticmethod
    @abstractmethod
    def load(path: str) -> Optional[Any]:
        pass


@attrs(auto_attribs=True)
class StorageObjectPickle(StorageObject):
    """
    Serializador de valores que usa `pickle` como backend.
    """

    @staticmethod
    def save(path: str, value: Any) -> None:
        """
        Método de guardado del elemento.

        Parameters
        ----------
        path: str
            Ruta completa donde se quiere almacenar el archivo.
        value: Any
        """
        with open(path, "wb") as path_handle:
            pickle.dump(value, path_handle)

    @staticmethod
    def load(path: str) -> Optional[Any]:
        """
        Método que carga el elemento usando pickle.

        Parameters
        ----------
        path: str
            Ruta del archivo.

        Returns
        -------
        object: Any
            Objeto cargado
        """
        with open(path, "rb") as path_handle:
            return pickle.load(path_handle)


@attrs(auto_attribs=True)
class StorageObjectPandasCSV(StorageObject):
    """
    Serializador de dataframes que usa csv como backend.
    """

    @staticmethod
    def save(path: str, value: pd.DataFrame) -> None:
        """
        Método de guardado del elemento.

        Parameters
        ----------
        path: str
            Ruta completa donde se quiere almacenar el archivo.
        value: Any
        """
        value.to_csv(path + ".csv", index=False)

    @staticmethod
    def load(path: str) -> pd.DataFrame:
        """
        Método que carga el elemento.

        Parameters
        ----------
        path: str
            Ruta del archivo.

        Returns
        -------
        object: pd.DataFrame
            Objeto cargado
        """
        if not path[-4:] in (".csv", ".tsv"):
            return pd.read_csv(path + ".csv")
        return pd.read_csv(path)


@attrs(auto_attribs=True)
class StorageObjectTableVotable(StorageObject):
    """
    Serializador de Table de astropy que usa vot como backend.
    """

    @staticmethod
    def save(path: str, value: Table) -> None:
        """
        Método de guardado del elemento.

        Parameters
        ----------
        path: str
            Ruta completa donde se quiere almacenar el archivo.
        value: Any
        """
        value.write(path + ".vot", format="votable")

    @staticmethod
    def load(path: str) -> Table:
        """
        Método que carga el elemento.

        Parameters
        ----------
        path: str
            Ruta del archivo.

        Returns
        -------
        object: Table
            Objeto cargado
        """
        return Table.read(path, format="votable")


@attrs(auto_attribs=True)
class StorageObjectFigures(StorageObject):
    """
    Serializador de Table de astropy que usa vot como backend.
    """

    @staticmethod
    def save(path: str, value: plt.Figure) -> None:
        """
        Método de guardado del elemento en png y pkl.

        Parameters
        ----------
        path: str
            Ruta completa donde se quiere almacenar el archivo.
        value: plt.Figure
        """
        value.savefig(path + ".png")
        with open(path, "wb") as path_handle:
            pickle.dump(value, path_handle)

    @staticmethod
    def load(path: str) -> Table:
        """
        Método que carga el elemento.

        Parameters
        ----------
        path: str
            Ruta del archivo.

        Returns
        -------
        object: Table
            Objeto cargado
        """
        with open(path, "rb") as path_handle:
            return pickle.load(path_handle)


@attrs(auto_attribs=True, frozen=True)
class ContainerSerializerZip:
    """
    Serializador de contenedor que guarda todos los valores serializados en un archivo zip.

    Parameters
    ----------
    serializers : Mapping[str, ValueSerializer]
        La relación "campo <-> serializador" de todos los valores que se serializarán del
        contenedor.
    """

    _serializers: Mapping[str, StorageObject]

    def save(self, path: str, container: Mapping[str, Any]) -> None:
        with TemporaryDirectory() as temp_path:
            for name, serializer in self._serializers.items():
                temp_file_path = os.path.join(temp_path, name)
                serializer.save(temp_file_path, container[name])
            shutil.make_archive(path, "zip", temp_path)

    def load(self, path: str) -> Dict[str, Any]:
        with ZipFile(path, "r") as zip_instance:
            if zip_instance.testzip() is not None:
                raise InvalidFileFormat(f"El archivo '{path}' no es un archivo zip válido")
            with TemporaryDirectory() as temp_path:
                zip_instance.extractall(temp_path)
                container = {}
                for name, serializer in self._serializers.items():
                    temp_file_path = os.path.join(temp_path, name)
                    container[name] = serializer.load(temp_file_path)
        return container

    @staticmethod
    def load_files(path: str, serializer: StorageObject) -> Dict[str, Any]:
        with ZipFile(path, "r") as zip_instance:
            if zip_instance.testzip() is not None:
                raise InvalidFileFormat(f"El archivo '{path}' no es un archivo zip válido")
            with TemporaryDirectory() as temp_path:
                zip_instance.extractall(temp_path)
                data_files = os.listdir(temp_path)
                data_files.sort()
                container = {}
                for file in data_files:
                    file_path = os.path.join(temp_path, file)
                    name = file.split("/")[-1].split(".")[0]
                    container[name] = serializer.load(file_path)
        return container
