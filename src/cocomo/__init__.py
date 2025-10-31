from .__version__ import __version__
from .cocomocg import COCOMO
from .structure import (
    Atom,
    Chain,
    DomainSelector,
    Model,
    PDBReader,
    Residue,
    SelectionError,
    Structure,
)

__all__ = [
    "__version__",
    "Atom",
    "Chain",
    "COCOMO",
    "DomainSelector",
    "Model",
    "PDBReader",
    "Residue",
    "SelectionError",
    "Structure",
]
