import pytest


@pytest.mark.parametrize("dig1, dig2, expected", [(1, 2, 3)])
def test_dummy(dig1, dig2, expected):
    result = dig1 + dig2
    assert result == expected
