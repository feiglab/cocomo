from __future__ import annotations

import gzip
import io
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import mdtraj as md
import numpy as np
from openmm import Vec3, unit
from openmm.app import Topology, element

FileLike = Union[str, Path, io.BytesIO, io.StringIO]

# --- Data containers ---------------------------------------------------------


@dataclass(frozen=True)
class Atom:
    serial: int
    name: str  # e.g. "CA"
    element: str  # 'H', 'C', 'O', 'N', 'S', 'P' 'CL', 'SOD', 'MG', 'CA'
    resname: str  # e.g. "ALA"
    chain: str  # original PDB chain ID
    resnum: int  # residue sequence number
    x: float
    y: float
    z: float
    seg: str  # segment ID (may be "")

    def __repr__(self) -> str:
        return f"<atom {self.name} {self.resname} {self.resnum} {self.chain} {self.seg}>"


@dataclass
class Residue:
    resname: str
    chain: str  # original PDB chain ID
    resnum: int
    seg: str  # segment ID
    atoms: list[Atom] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"<residue {self.resname} {self.resnum} {self.chain} {self.seg}>"


@dataclass
class Chain:
    key_id: str  # key used in Structure.models[...].chains
    residues: list[Residue] = field(default_factory=list)
    seg_id: Optional[str] = None  # segment ID if grouping by seg
    chain_id: Optional[str] = None  # chain ID from PDB

    def __repr__(self) -> str:
        return f"<chain {self.key_id} : segment {self.seg_id} chain {self.chain_id}>"


@dataclass
class Model:
    model_id: int
    chain: dict[str, Chain] = field(default_factory=dict)  # key_id -> Chain
    residues: list[Residue] = field(default_factory=list)
    atoms: list[Atom] = field(default_factory=list)

    def chains(self) -> Iterator[Chain]:
        return iter(self.chain.values())

    def iter_residues(self) -> Iterator[Residue]:
        for c in self.chain.values():
            for r in c.residues:
                yield from c.residues

    def iter_atoms(self) -> Iterator[Atom]:
        for c in self.chain.values():
            for r in c.residues:
                for a in r.atoms:
                    yield from r.atoms

    def __repr__(self) -> str:
        n_chains = len(self.chain)
        n_res = sum(len(c.residues) for c in self.chain.values())
        n_atoms = len(self.atoms)
        return f"<{n_chains} chains, {n_res} residues, {n_atoms} atoms>"

    __str__ = __repr__

    def positions(self):
        """
        Return positions as an OpenMM Quantity[list[Vec3]] in nanometers.
        Assumes internal coordinates are in Angstroms (PDB convention).
        """
        # Build Vec3 list from parsed coordinates (Å)
        vecs = [Vec3(a.x, a.y, a.z) for a in self.atoms]
        # Attach Å units, then convert to nm to match PDBFile default
        return unit.Quantity(vecs, unit.angstrom).in_units_of(unit.nanometer)

    def topology(self):
        top = Topology()
        for c in self.chains():
            chain = top.addChain(c.key_id)
            for r in c.residues:
                rname = r.resname
                res = top.addResidue(rname, chain)
                for a in r.atoms:
                    sym = (getattr(a, "element", "") or "").upper()
                    el = element.Element.getBySymbol(sym) or element.carbon
                    top.addAtom(a.name, element=el, residue=res)
        return top

    # ---- selections on a single Model ----

    def _select_by_index_set(self, keep: set[int]) -> Model:
        """Internal: build a new Model with only atoms whose model-local indices are in `keep`."""
        m2 = Model(model_id=self.model_id)
        if not keep or not self.atoms:
            return m2

        running_idx = -1
        for key, ch in self.chain.items():
            new_chain = Chain(
                key_id=ch.key_id,
                seg_id=getattr(ch, "seg_id", None),
                chain_id=getattr(ch, "chain_id", None),
            )
            for r in ch.residues:
                kept_atoms: list[Atom] = []
                for a in r.atoms:
                    running_idx += 1
                    if running_idx in keep:
                        kept_atoms.append(a)
                        m2.atoms.append(a)
                if kept_atoms:
                    new_res = Residue(
                        resname=r.resname,
                        chain=r.chain,
                        resnum=r.resnum,
                        seg=r.seg,
                        atoms=kept_atoms,
                    )
                    new_chain.residues.append(new_res)
                    m2.residues.append(new_res)
            if new_chain.residues:
                m2.chain[key] = new_chain
        return m2

    @staticmethod
    def _flatten_indices(indices: Union[list[int], list[list[int]]]) -> list[int]:
        if not indices:
            return []
        if isinstance(indices[0], (list, tuple)):
            out: list[int] = []
            for sub in indices:  # type: ignore[assignment]
                out.extend(int(i) for i in sub)
            return out
        return [int(i) for i in indices]  # type: ignore[return-value]

    def select_byindex(self, indices: Union[list[int], list[list[int]]]) -> Model:
        """
        Return a new Model containing only atoms at the given 0-based indices
        (per this model's atom order). Accepts a flat list or list-of-lists.
        Duplicates removed, negatives ignored, indices sorted before applying.
        """
        flat = self._flatten_indices(indices)
        keep = {i for i in flat if isinstance(i, int) and i >= 0}
        return self._select_by_index_set(keep)

    def select_CA(self) -> Model:
        """Return a new Model containing only CA atoms."""
        keep = {i for i, a in enumerate(self.atoms) if a.name == "CA"}
        return self._select_by_index_set(keep)

    def select_bystring(self, spec: str) -> Model:
        """
        Return a new Model using a textual selection `spec` via DomainSelector.
        This method builds a temporary single-model Structure to reuse the selector.
        """
        if not isinstance(spec, str) or not spec.strip():
            raise ValueError("select_bystring requires a non-empty selection string")

        # normalize "H271:2-91" -> "H271.2-91" (first ':' as chain/res separator)
        raw = spec.strip()
        if "." not in raw and ":" in raw:
            head, tail = raw.split(":", 1)
            if tail and tail.lstrip() and tail.lstrip()[0].isdigit():
                raw = f"{head}.{tail}"

        # Build a temporary Structure with this model only
        temp_struct = Structure(models=[self])

        sel = DomainSelector(raw)
        # list-of-lists (per selector semantics)
        idx_lists = sel.atom_lists(temp_struct, model_index=0)
        return self.select_byindex(idx_lists)

    def mdtraj_trajectory(self):
        top = md.Topology.from_openmm(self.topology())
        coords_nm = [(a.x / 10.0, a.y / 10.0, a.z / 10.0) for a in self.atoms]  # Å -> nm

        if not coords_nm:
            traj = md.Trajectory(xyz=np.zeros((1, 0, 3), dtype=float), topology=top)
            return traj

        xyz = np.array([coords_nm], dtype=np.float32)  # (1, natoms, 3) nm
        traj = md.Trajectory(xyz=xyz, topology=top)
        return traj

    def sasa_by_residue(
        self,
        *,
        probe_radius: float = 0.14,
        n_sphere_points: int = 960,
        radii: str = "bondi",
    ) -> list[float]:
        """
        Fast SASA (nm^2) per residue using MDTraj Shrake–Rupley with Bondi radii.
        Parameters
        ----------
        probe_radius : float  (nm)
        n_sphere_points : int
        radii : str   (currently 'bondi' only; MDTraj uses element radii table)
        """

        if radii.lower() != "bondi":
            raise ValueError("Only 'bondi' radii are supported with the MDTraj backend.")

        traj = self.mdtraj_trajectory()
        if traj.n_atoms == 0:
            return {}

        # MDTraj expects nm for radii; returns nm^2
        sasa_nm2 = md.shrake_rupley(
            traj,
            n_sphere_points=int(n_sphere_points),
            mode="residue",
        )  # shape (1, n_residues)
        per_res_nm2 = sasa_nm2[0]

        return per_res_nm2


@dataclass
class Structure:
    models: list[Model] = field(default_factory=list)

    def __getitem__(self, idx: Union[int, slice]) -> Union[Model, list[Model]]:
        return self.models[idx]

    def __len__(self) -> int:
        return len(self.models)

    def __iter__(self) -> Iterator[Model]:
        return iter(self.models)

    @property
    def model(self) -> Model:
        """Return the first model (common for single-model files)."""
        if not self.models:
            raise ValueError("Structure has no models")
        return self.models[0]

    def positions(self, model_index: int = 0):
        """Positions for the selected model as Quantity[list[Vec3]] in nm."""
        return self.models[model_index].positions()

    def topology(self):
        return self.models[0].topology()

    def select_CA(self) -> Structure:
        """Apply CA selection to each model; return a new Structure."""
        out = Structure()
        for m in self.models:
            out.models.append(m.select_CA())
        if not out.models:
            out.models.append(Model(model_id=1))
        return out

    def select_byindex(self, indices: Union[list[int], list[list[int]]]) -> Structure:
        """
        Apply the same index selection to each model; return a new Structure.
        Indices are interpreted per-model (0-based within each model).
        """
        out = Structure()
        for m in self.models:
            out.models.append(m.select_byindex(indices))
        if not out.models:
            out.models.append(Model(model_id=1))
        return out

    def select_bystring(self, spec: str) -> Structure:
        """
        Apply textual selection to each model independently (chains/residues resolved per model);
        return a new Structure.
        """
        out = Structure()
        for m in self.models:
            out.models.append(m.select_bystring(spec))
        if not out.models:
            out.models.append(Model(model_id=1))
        return out

    def sasa_by_residue(
        self,
        *,
        model_index: int = 0,
        probe_radius: float = 0.14,
        n_sphere_points: int = 960,
        radii: str = "bondi",
    ) -> list[float]:
        """
        Compute SASA (nm^2) by residue for a chosen model (default 0) via MDTraj.
        """
        if not self.models:
            return {}
        if model_index < 0 or model_index >= len(self.models):
            raise IndexError(f"model_index {model_index} out of range (0..{len(self.models)-1})")
        return self.models[model_index].sasa_by_residue(
            probe_radius=probe_radius,
            n_sphere_points=n_sphere_points,
            radii=radii,
        )


# --- Parser ------------------------------------------------------------------


class PDBReader:
    """
    Minimal, fast PDB reader
    - Supports MODEL/ENDMDL (multiple models).
    - Parses ATOM.
    - Groups atoms into chains keyed by SEGID when available; else by PDB chain ID with
      automatic suffixing (A, A1, A2, ...) when non-contiguous repeats occur.
    """

    def __new__(cls, file: Optional[FileLike] = None):
        self = super().__new__(cls)
        if file is None:
            return self
        return cls._read_direct(file)

    def read(self, file: FileLike) -> Structure:
        text_iter = self._open_text(file)
        return self._parse(text_iter)

    def from_string(self, pdb_text: str) -> Structure:
        return self._parse(pdb_text.splitlines())

    # -- internals --

    @staticmethod
    def _open_text(file: FileLike) -> Iterable[str]:
        if isinstance(file, (io.StringIO, io.BytesIO)):
            if isinstance(file, io.BytesIO):
                return io.TextIOWrapper(file, encoding="utf-8", newline="").read().splitlines()
            return file.getvalue().splitlines()
        p = Path(file)
        if p.suffix == ".gz":
            with gzip.open(p, "rt", encoding="utf-8", newline="") as fh:
                return fh.read().splitlines()
        with open(p, encoding="utf-8", newline="") as fh:
            return fh.read().splitlines()

    @classmethod
    def _read_direct(cls, file: FileLike) -> Structure:
        return cls._parse(cls._open_text(file))

    @staticmethod
    def _parse(lines: Iterable[str]) -> Structure:
        s = Structure()
        current_model: Optional[Model] = None

        # State for allocating fallback chain keys when SEGID is absent
        # counts['A'] -> how many extra chains created beyond the first contiguous block
        fallback_counts: dict[str, int] = {}
        last_chain_id_seen: Optional[str] = None  # last PDB chain ID encountered (for contiguity)

        def alloc_chain_key(atom: Atom) -> str:
            """Return chain key for this atom per rules."""
            nonlocal last_chain_id_seen
            if atom.seg.strip():
                # Primary rule: group by segment ID, exactly as key
                key = atom.seg.strip()
                # still track for correctness across TER, but not used for seg
                last_chain_id_seen = atom.chain
                return key

            # Fallback: group by PDB chain ID, splitting non-contiguous repeats
            cid = (atom.chain or "").strip() or " "
            if cid not in s.models[-1].chain:
                # first ever chain with this cid in this model -> plain key
                key = cid
                last_chain_id_seen = cid
                return key

            # If we are still in the same contiguous block (cid has not changed since last atom),
            # keep using it
            if last_chain_id_seen == cid:
                return cid if cid in s.models[-1].chain else cid

            # Non-contiguous repeat: allocate suffixed key
            n = fallback_counts.get(cid, 0) + 1
            fallback_counts[cid] = n
            key = f"{cid}{n}"
            last_chain_id_seen = cid
            return key

        def start_chain_if_needed(m: Model, key: str, atom: Atom):
            ch = m.chain.get(key)
            if ch is None:
                ch = Chain(key_id=key, seg_id=(atom.seg.strip() or None))
                m.chain[key] = ch
            # record original PDB chain id
            ch.chain_id = atom.chain or " "
            return ch

        def add_atom_to_model(m: Model, atom: Atom):
            m.atoms.append(atom)
            key = alloc_chain_key(atom)
            chain = start_chain_if_needed(m, key, atom)

            # residue identity is solely by original PDB chain id and resnum/resname
            rid = (atom.resname, atom.chain, atom.resnum, atom.seg)
            if not chain.residues or _res_id(chain.residues[-1]) != rid:
                chain.residues.append(Residue(*rid))
            chain.residues[-1].atoms.append(atom)

        for raw in lines:
            if not raw:
                continue
            rec = raw[0:6].strip().upper()

            if rec == "MODEL":
                model_id = _safe_int(raw[10:14], default=len(s.models) + 1)
                current_model = Model(model_id=model_id)
                s.models.append(current_model)
                # reset fallback state for a new model
                fallback_counts = {}
                last_chain_id_seen = None
                continue

            if rec == "ENDMDL":
                current_model = None
                continue

            if rec == "ATOM":
                if current_model is None:
                    current_model = Model(model_id=1)
                    s.models.append(current_model)
                    # reset fallback state for implicit first model
                    fallback_counts = {}
                    last_chain_id_seen = None
                atom = _parse_atom_line(raw)
                add_atom_to_model(current_model, atom)
                continue

            if rec == "TER":
                # TER terminates a chain segment
                # subsequent atom with the same PDB chain ID will start a new chain
                # (e.g., A -> TER -> A becomes A1).
                last_chain_id_seen = None
                continue

        if not s.models:
            s.models.append(Model(model_id=1))
        return s


# --- parsing utilities -------------------------------------------------------


def _deduce_element(atomname: str, resname: str, element_hint: str = "") -> str:
    """
    Deduce an element symbol following user rules.
    Priority:
      1) Use PDB element column if present (uppercased, non-letters removed).
      2) Special cases from atom/residue names.
      3) First-letter rules C/N/H/S/P/O (after stripping leading digits in atom name).
      4) Fallback: atom name with digits removed (uppercased).
    """

    def clean(token: str) -> str:
        # keep only letters, upcase
        return re.sub(r"[^A-Za-z]", "", token or "").upper()

    # 1) PDB element column (columns 77-78)
    if element_hint and clean(element_hint):
        return clean(element_hint)

    an = clean(atomname)
    rn = clean(resname)

    # 2) Explicit mappings
    # Chloride / sodium / potassium aliases
    if an in {"CLA", "CL"} or atomname.upper() in {"CL-", "CLA"}:
        return "CL"
    if rn in {"CLA", "CL"}:
        return "CL"

    if an in {"NA", "SOD"} or atomname.upper() == "NA+":
        return "NA"
    if rn in {"NA", "SOD"}:
        return "NA"

    if an == "POT" or rn == "POT":
        return "K"

    # direct “use-name” set
    direct = {"MG", "CAL", "K", "LI", "FE", "CO", "MB"}
    if an in direct:
        return an
    if rn in direct:
        return rn

    # 3) First-letter rules after stripping leading digits from atom name
    atom_wo_lead_digits = re.sub(r"^\d+", "", atomname or "")
    atom_wo_digits = re.sub(r"\d", "", atom_wo_lead_digits).strip()
    if atom_wo_digits:
        ch0 = atom_wo_digits[0].upper()
        if ch0 in {"C", "N", "H", "S", "P", "O"}:
            return ch0

    # 4) Fallback: atom name without any digits, uppercased (e.g., "Cl1" -> "CL")
    fb = clean(atomname)
    return fb if fb else "X"


def _parse_atom_line(line: str) -> Atom:
    # PDB v3.3 column mapping, simplified
    serial = _safe_int(line[4:11], required=True)  # left as in your version
    name = line[12:16].strip()
    resname = line[17:21].strip()
    chain = (line[21] if len(line) >= 22 else " ").strip()
    resnum = _safe_int(line[22:26], required=True)
    x = _safe_float(line[30:38], required=True)
    y = _safe_float(line[38:46], required=True)
    z = _safe_float(line[46:54], required=True)
    seg = (line[72:76] if len(line) >= 76 else " ").strip()
    element_hint = (line[76:78] if len(line) >= 78 else "").strip()
    element = _deduce_element(name, resname, element_hint)
    return Atom(
        serial=serial,
        name=name,
        element=element,
        resname=resname,
        chain=chain,
        resnum=resnum,
        x=x,
        y=y,
        z=z,
        seg=seg,
    )


def _safe_int(s: str, default: Optional[int] = None, required: bool = False) -> Optional[int]:
    try:
        return int(s.strip())
    except Exception:
        if required:
            raise ValueError(f"Expected integer in field '{s}'")
        return default


def _safe_float(s: str, default: Optional[float] = None, required: bool = False) -> Optional[float]:
    try:
        return float(s.strip())
    except Exception:
        if required:
            raise ValueError(f"Expected float in field '{s}'")
        return default


def _res_id(r: Residue) -> tuple[str, str, int, str]:
    return (r.resname, r.chain, r.resnum, r.seg)


# ---- DomainSelector ----------------------------------------------------------


class SelectionError(ValueError):
    """Raised when a selection term cannot be parsed or resolved."""


# ----------------------------- parsing primitives ----------------------------


@dataclass(frozen=True)
class ResidueSelector:
    """Represents residue selection for a chain (or all chains)."""

    all_residues: bool
    ranges: tuple[tuple[int, int], ...] = ()  # inclusive ranges; singletons are (n, n)

    @staticmethod
    def parse(spec: str) -> ResidueSelector:
        s = spec.strip().lower()
        if s == "all":
            return ResidueSelector(all_residues=True)

        toks = [t for t in re.split(r"[.:]", spec) if t.strip()]
        ranges: list[tuple[int, int]] = []
        for t in toks:
            t = t.strip()
            if "-" in t:
                a, b = t.split("-", 1)
                try:
                    lo = int(a)
                    hi = int(b)
                except ValueError as e:
                    raise SelectionError(f"Invalid residue range '{t}' in '{spec}'") from e
                if lo > hi:
                    lo, hi = hi, lo
                ranges.append((lo, hi))
            else:
                try:
                    n = int(t)
                except ValueError as e:
                    raise SelectionError(f"Invalid residue token '{t}' in '{spec}'") from e
                ranges.append((n, n))

        if not ranges:
            raise SelectionError(f"Empty residue spec '{spec}'")

        return ResidueSelector(all_residues=False, ranges=tuple(ranges))

    def contains(self, resnum: int) -> bool:
        if self.all_residues:
            return True
        return any(lo <= resnum <= hi for (lo, hi) in self.ranges)


@dataclass(frozen=True)
class Term:
    """One selection term: chains (or None for all chains) + residue selector (or all)."""

    chains: Optional[tuple[str, ...]]  # None => all chains
    residues: ResidueSelector


def _parse_chain_list(s: str) -> tuple[str, ...]:
    ids = [tok for tok in s.split(":") if tok.strip()]
    if not ids:
        raise SelectionError(f"Empty chain list in '{s}'")
    return tuple(ids)


def _looks_like_residue_spec(s: str) -> bool:
    s = s.strip().lower()
    if s == "all":
        return True
    # digits, dashes, dots, colons => residue expressions (e.g., "2-91.93-94" or "5:7")
    return bool(re.fullmatch(r"[0-9][0-9:.\-]*", s))


# ----------------------------- public selector -------------------------------


class DomainSelector:
    """
    Parse domain spec strings and produce atom lists from your Structure/Model.

    Semantics:
      • If ANY term specifies chains => return ONE atom list pooled over those chains/terms.
      • If NO term specifies chains  => return ONE atom list PER CHAIN (same residue spec applied).

    Grammar (forgiving):
      - Terms separated by whitespace.
      - Chain lists use ':' (e.g., 'H271:H272').
      - Chain vs residues separated by first '.' (e.g., 'H271.2-91.93-94' or 'H271.all').
      - If only residues are given (e.g., '2-91.93-94'), they apply to all chains.
      - 'all' alone => all chains, all residues.
      - After a chain list, residue ranges may be split by '.' or ':'; both accepted.
      - Mixed terms allowed: 'H271.2-91 H273.2-91 H276.all'.

    Chain matching:
      - A chain token matches if it exactly equals any of:
          chain.key_id (primary), chain.seg_id, chain.chain_id (if present).
    """

    def __init__(self, spec: str):
        if not spec or not spec.strip():
            raise SelectionError("Empty selection spec")
        self._raw = spec
        self._terms = self._parse(spec)
        self._has_explicit_chains = any(t.chains is not None for t in self._terms)

    def atom_lists(
        self, structure: Union[Structure, Model], model_index: int = 0
    ) -> list[list[int]]:
        """
        Return one or more atom lists (each sorted, 0-based indices into Model.atoms)
        according to the spec aggregation rule described in the class docstring.
        """
        if isinstance(structure, Structure):
            model = structure.models[model_index]
        else:
            model = structure
        atom_to_idx = {id(a): i for i, a in enumerate(model.atoms)}

        # Map: alias_key -> set(resnums) selected for that alias
        alias_to_resnums: dict[str, set[int]] = self._resolve_residues(model)

        if self._has_explicit_chains:
            # Pool across all chains referenced by the terms.
            pooled: set[int] = set()
            for ch in model.chains():
                resnums = _union_resnums_for_chain(ch, alias_to_resnums)
                if not resnums:
                    continue
                for r in ch.residues:
                    if r.resnum in resnums:
                        for a in r.atoms:
                            idx = atom_to_idx.get(id(a))
                            if idx is not None:
                                pooled.add(idx)
            return [sorted(pooled)]

        # No chains specified in any term: emit one list per chain (if any residues selected)
        lists: list[list[int]] = []
        for ch in model.chains():
            resnums = _union_resnums_for_chain(ch, alias_to_resnums)
            if not resnums:
                continue
            out: set[int] = set()
            for r in ch.residues:
                if r.resnum in resnums:
                    for a in r.atoms:
                        idx = atom_to_idx.get(id(a))
                        if idx is not None:
                            out.add(idx)
            lists.append(sorted(out))
        return lists

    def atom_indices(self, structure: Union[Structure, Model], model_index: int = 0) -> list[int]:
        """Flattened union of all lists returned by atom_lists()."""
        lists = self.atom_lists(structure, model_index=model_index)
        merged: set[int] = set()
        for lst in lists:
            merged.update(lst)
        return sorted(merged)

    def residue_keys(self, structure: Structure, model_index: int = 0) -> list[tuple[str, int]]:
        """
        Return sorted (chain_key_id, residue_number) present in the selection.
        If explicit chains are used: union across selected chains.
        If not: union across all chains where the per-chain list would be emitted.
        """
        model = structure.models[model_index]
        alias_to_resnums = self._resolve_residues(model)

        out: set[tuple[str, int]] = set()
        for ch in model.chains():
            resnums = _union_resnums_for_chain(ch, alias_to_resnums)
            if not resnums:
                continue
            for r in ch.residues:
                if r.resnum in resnums:
                    out.add((ch.key_id, r.resnum))
        return sorted(out)

    # --------------------------- internals ------------------------------------

    def _resolve_residues(self, model: Model) -> dict[str, set[int]]:
        """
        Return a mapping from ANY acceptable chain alias (key_id, seg_id, chain_id)
        to a set of residue numbers selected for that alias.
        """
        # Collect alias universe
        all_aliases: set[str] = set()
        chain_by_alias: dict[str, Chain] = {}
        for ch in model.chains():
            for k in _all_chain_aliases(ch):
                if k is not None:
                    all_aliases.add(k)
                    chain_by_alias[k] = ch

        selected: dict[str, set[int]] = {}

        for term in self._terms:
            # Determine target aliases for this term
            if term.chains is None:
                target_aliases = set(all_aliases)
            else:
                target_aliases = set()
                unknown: list[str] = []
                for tok in term.chains:
                    if tok in all_aliases:
                        target_aliases.add(tok)
                    else:
                        unknown.append(tok)
                if unknown:
                    avail = sorted(all_aliases)
                    raise SelectionError(
                        f"Unknown chain IDs in spec '{self._raw}': {unknown}. Available: {avail}"
                    )

            # Assign residue numbers per targeted alias
            for alias in target_aliases:
                ch = chain_by_alias[alias]
                bucket = selected.setdefault(alias, set())
                if term.residues.all_residues:
                    for r in ch.residues:
                        bucket.add(r.resnum)
                else:
                    for r in ch.residues:
                        if term.residues.contains(r.resnum):
                            bucket.add(r.resnum)

        return selected

    @staticmethod
    def _parse(spec: str) -> tuple[Term, ...]:
        terms: list[Term] = []
        for raw_term in spec.split():
            t = raw_term.strip()
            if not t:
                continue
            if t.lower() == "all":
                terms.append(Term(chains=None, residues=ResidueSelector(all_residues=True)))
                continue

            if "." in t:
                head, tail = t.split(".", 1)
                head = head.strip()
                tail = tail.strip()
                chains = _parse_chain_list(head) if head else None
                residues = ResidueSelector.parse(tail)
                terms.append(Term(chains=chains, residues=residues))
                continue

            if _looks_like_residue_spec(t):
                residues = ResidueSelector.parse(t)
                terms.append(Term(chains=None, residues=residues))
            else:
                chains = _parse_chain_list(t)
                residues = ResidueSelector(all_residues=True)
                terms.append(Term(chains=chains, residues=residues))

        if not terms:
            raise SelectionError(f"Could not parse spec '{spec}'")
        return tuple(terms)


# ----------------------------- helpers ---------------------------------------


def _all_chain_aliases(ch: Chain) -> tuple[str, ...]:
    out: list[str] = []
    if getattr(ch, "key_id", None):
        out.append(str(ch.key_id))
    if getattr(ch, "seg_id", None):
        out.append(str(ch.seg_id))
    if hasattr(ch, "chain_id") and getattr(ch, "chain_id") is not None:
        out.append(str(getattr(ch, "chain_id")))
    # De-duplicate while preserving order
    seen = set()
    uniq = []
    for k in out:
        if k not in seen:
            uniq.append(k)
            seen.add(k)
    return tuple(uniq)


def _union_resnums_for_chain(ch: Chain, alias_to_resnums: dict[str, set[int]]) -> set[int]:
    """Union residue sets across all aliases of a chain."""
    resnums: set[int] = set()
    for alias in _all_chain_aliases(ch):
        resnums |= alias_to_resnums.get(alias, set())
    return resnums
