from cocomo import COCOMO, PDBReader, DomainSelector

def test_describe():
    m=COCOMO()
    assert "COCOMO" in m.describe()

