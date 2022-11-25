import pytest

from vampires_dpp.pipeline import check_version


@pytest.mark.parametrize(
    ("cver", "vpver", "exp"),
    (
        ("0.1.0", "0.1.0", True),
        ("0.1.0", "0.2.0", False),
        ("0.2.2", "0.2.0", False),
        ("0.2.2", "0.2.4", True),
        ("0.3.0", "0.2.0", False),
        ("1.0.0", "1.0.0", True),
        ("1.2.0", "1.0.0", False),
        ("1.0.0", "1.2.0", True),
        ("1.2.3", "1.2.0", False),
        ("1.2.3", "1.2.5", True),
        ("1.2.3", "2.0.0", False),
    ),
)
def test_check_version(cver, vpver, exp):
    assert check_version(cver, vpver) == exp
