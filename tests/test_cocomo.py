from cocomo import COCOMO


def test_describe():
    m = COCOMO()
    assert "COCOMO" in m.describe()
