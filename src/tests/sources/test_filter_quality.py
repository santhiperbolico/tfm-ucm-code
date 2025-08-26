import pytest

from hyper_velocity_stars_detection.sources.filter_quality import (
    GAIA_DR3_FIELDS,
    FieldType,
    QueryField,
    QueryProcessor,
)


@pytest.mark.parametrize(
    "field, expression",
    [
        (QueryField("name", "name", FieldType.FILTERS, "eq"), "name=0"),
        (QueryField("name", "name", FieldType.FILTERS, "gs"), "name>0"),
        (QueryField("name", "name", FieldType.FILTERS, "gseq"), "name>=0"),
        (QueryField("name", "name", FieldType.FILTERS, "ls"), "name<0"),
        (QueryField("name", "name", FieldType.FILTERS, "lseq"), "name<=0"),
    ],
)
def test_query_filter_expression(field, expression):
    result = field.get_expression_field(0)
    assert result == expression


def test_query_processor():
    field_filters = {"ra": 0, "dec": 0, "radius": 1, "ast_params_solved": 3, "v_periods_used": 11}
    query_processor = QueryProcessor(GAIA_DR3_FIELDS, field_filters)
    expected_fields = {
        "ra": 0,
        "dec": 0,
        "radius": 1,
        "ast_params_solved": 3,
        "ruwe": 1.4,
        "v_periods_used": 11,
        "ipd_gof_har_amp": None,
        "ipd_frac_multi_peak": None,
        "min_parallax": None,
        "max_parallax": None,
    }
    expected_query = """
                   SELECT tab.* FROM tab WHERE
                   CONTAINS(
                          POINT('ICRS', ra, dec),
                          CIRCLE('ICRS', 0, 0, 1)
                          )=1 AND astrometric_params_solved>3
                          AND ruwe<1.4 AND visibility_periods_used>11
                   """
    assert query_processor._fields == expected_fields
    assert query_processor.get_query("tab") == " ".join(expected_query.split())
