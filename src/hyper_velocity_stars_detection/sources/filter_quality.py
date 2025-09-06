from abc import ABC
from enum import Enum
from typing import Any, Optional, Sequence

from attr import attrs


class FieldType(Enum):
    POSITION = 0
    FILTERS = 1
    OTHERS = 2


class SEARCH_TYPES:
    CIRCLE = "circle"


OPERATIONS = {"eq": "=", "ls": "<", "lseq": "<=", "gs": ">", "gseq": ">="}


@attrs(auto_attribs=True)
class QueryField(ABC):
    param_name: str
    field_name: str
    type: FieldType
    operation: str
    default_value: Optional[Any] = None

    def get_expression_field(self, value: Any) -> str:
        """
        Obtiene la forma de la expresión que se debe usar en el filtro de la query

        Parameters
        ----------
        value: Any
            Valor del campo a utilizar.

        Return
        ------
        str
            Expresión del filtro.
        """
        try:
            symbol = OPERATIONS[self.operation]
        except KeyError:
            raise ValueError("La operación %s no está registrada" % self.operation)
        return f"{self.field_name}{symbol}{value}"


RA_FIELD = QueryField("ra", "ra", FieldType.POSITION, "eq")
DEC_FIELD = QueryField("dec", "dec", FieldType.POSITION, "eq")
RADIUS_FIELD = QueryField("radius", "radius", FieldType.POSITION, "eq")


class QueryProcessor:
    """
    Clase de objetos cuya tarea es la validación de parámetros de los
    modelos de clasificación de t2o.

    Parameter
    ---------
    fields_to_process: Sequence[Param]
        Lista de Param a validar.
    param: Dict[str, Any]
            Diccionario con los paraámetror
    """

    def __init__(self, fields_to_process: Sequence[QueryField], params: dict[str, Any]):
        self._fields_to_process = fields_to_process
        complete_params = self.get_default_values()
        complete_params.update(params)
        self._fields = {
            f.param_name: complete_params[f.param_name] for f in self._fields_to_process
        }

    def get_query(
        self, catalog_table: str, search_type: SEARCH_TYPES = SEARCH_TYPES.CIRCLE
    ) -> str:
        """
        Función que devuleve la query a ejecutar según los filtros indicados.

        Parameters
        ----------
        catalog_table: str
            Nombre de la tabla a consultar
        search_type: SEARCH_TYPES = SEARCH_TYPES.CIRCLE
            Tipo de búsqueda utilizada

        Returns
        -------
        query: str
            Query usada en la consulta
        """
        position = self._get_position_filter(search_type)
        filter = f"WHERE {position}"

        for field in self._fields_to_process:
            if field.type == FieldType.FILTERS:
                value_field = self._fields.get(field.param_name)
                if value_field is not None:
                    expression = field.get_expression_field(value_field)
                    filter = filter + f" AND {expression}"

        query = f"SELECT {catalog_table}.* FROM {catalog_table} {filter}"
        return " ".join(query.split())

    def _get_position_filter(self, search_type: SEARCH_TYPES) -> str:
        """
        Devuelve la parte del filtro correspondiente con los parámetros de posición.

        Returns
        -------
        : str
            Filtro con los datos de posición
        """
        if search_type == SEARCH_TYPES.CIRCLE:
            ra = self._fields[RA_FIELD.field_name]
            dec = self._fields[DEC_FIELD.field_name]
            radius = self._fields[RADIUS_FIELD.field_name]
            position = f"""
                   CONTAINS(
                          POINT('ICRS', ra, dec),
                          CIRCLE('ICRS', {ra}, {dec}, {radius})
                          )=1
                   """
            return position

        raise ValueError("El tipo de búsqueda %s no está implementado" % search_type)

    def get_default_values(self) -> dict[str, Any]:
        """
        Devuelve los campos que tengan valores por defecto definidos.

        Returns
        -------
        dict[str, Any]
            Campos con valores por defecto definido.
        """
        return {field.param_name: field.default_value for field in self._fields_to_process}

    def get_field_value(self, field_name) -> Any:
        """
        Devuelve el valor del campo seleccionado.

        Returns
        -------
        Any
            Valor asociado al campo
        """
        return self._fields.get(field_name)

    def get_field(self, field_name) -> QueryField:
        """
        Devuelve el valor del campo seleccionado.

        Returns
        -------
        QueryField
            Field asociado
        """
        for field in self._fields_to_process:
            if field.param_name == field_name:
                return field
        raise ValueError("No se ha encontrado el parámetro %s ." % field_name)


GAIA_DR3_FIELDS = (
    RA_FIELD,
    DEC_FIELD,
    RADIUS_FIELD,
    QueryField("ast_params_solved", "astrometric_params_solved", FieldType.FILTERS, "gs", 3),
    QueryField("ruwe", "ruwe", FieldType.FILTERS, "ls", 1.4),
    QueryField("v_periods_used", "visibility_periods_used", FieldType.FILTERS, "gs", 10),
    QueryField("ipd_gof_har_amp", "ipd_gof_harmonic_amplitude", FieldType.FILTERS, "ls"),
    QueryField("ipd_frac_multi_peak", "ipd_frac_multi_peak", FieldType.FILTERS, "ls"),
    QueryField("min_parallax", "parallax", FieldType.FILTERS, "gs"),
    QueryField("max_parallax", "parallax", FieldType.FILTERS, "ls"),
    QueryField("parallax_error", "parallax_error", FieldType.FILTERS, "ls"),
    QueryField("radial_velocity", "radial_velocity", FieldType.FILTERS, "gs"),
)

GAIA_DR2_FIELDS = (
    RA_FIELD,
    DEC_FIELD,
    RADIUS_FIELD,
    QueryField("ast_params_solved", "astrometric_params_solved", FieldType.FILTERS, "gs", 3),
    QueryField("ruwe", "ruwe", FieldType.OTHERS, "gs", 1.4),
    QueryField("v_periods_used", "visibility_periods_used", FieldType.FILTERS, "gs", 5),
    QueryField("min_parallax", "parallax", FieldType.FILTERS, "gs"),
    QueryField("max_parallax", "parallax", FieldType.FILTERS, "ls"),
    QueryField("parallax_error", "parallax_error", FieldType.FILTERS, "ls"),
)

GAIA_FPR_FIELDS = (
    RA_FIELD,
    DEC_FIELD,
    RADIUS_FIELD,
    QueryField("ast_params_solved", "astrometric_params_solved", FieldType.FILTERS, "gs", 3),
    QueryField("v_periods_used", "visibility_periods_used", FieldType.FILTERS, "gs", 10),
    QueryField("ipd_gof_har_amp", "ipd_gof_harmonic_amplitude", FieldType.FILTERS, "gs"),
    QueryField("min_parallax", "parallax", FieldType.FILTERS, "gs"),
    QueryField("max_parallax", "parallax", FieldType.FILTERS, "ls"),
    QueryField("parallax_error", "parallax_error", FieldType.FILTERS, "ls"),
)
