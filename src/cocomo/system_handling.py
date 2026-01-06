from __future__ import annotations

import shlex
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from math import log
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
from mdsim import (
    Model,
    PDBReader,
    Structure,
    StructureSelector,
)
from openmm.unit import (
    Quantity,
    nanometer,
    norm,
)

FileLike = Union[str, Path]
DirLike = Union[str, Path]

# --- Data containers ---------------------------------------------------------


@dataclass
class ComponentType:
    name: str = "unknown"  # e.g. "hexamer"
    nunit: Optional[int] = None  # number of individual units
    sasa: list[tuple[int, float]] = field(default_factory=list)  # SASA per residue
    domainres: list[list[int]] = field(default_factory=list)  # residue selection for domain
    enmpairs: list[int, int, float] = field(default_factory=list)  # ENM pair list

    model: Optional[Model] = None  # reference model

    pdb: Optional[FileLike] = None  # reference PDB file

    getsasa: Optional[FileLike] = None  # read SASA from file, calculate if 'auto'

    domainsel: Optional[str] = None  # domain selection from string
    getdomains: Optional[FileLike] = None  # read domain lists from file

    getenm: Optional[FileLike] = None  # read ENM pairs from file, calculate if 'auto'
    enmcutoff: Optional[float] = None  # in nm, default is 0.9

    mask_sasa_bydomain: Optional[bool] = True  # mask SASA values outside domain
    default_sasa: Optional[float] = 999.0  # default SASA value

    def __repr__(self) -> str:
        r = f"<component type {self.name!r} with {self.nunit} units"
        r += f", SASA: {len(self.sasa)}, domains: {len(self.domainres)}"
        r += f", ENM pairs: {len(self.enmpairs)}>"
        return r

    def __post_init__(self) -> None:
        if self.enmcutoff is None:
            self.enmcutoff = 0.9

        if self.pdb is not None:
            _ensure_readable(self.pdb)
            s = PDBReader(self.pdb)
            self.model = s[0]

        if self.getdomains is not None and len(self.domainres) == 0:
            _ensure_readable(self.getdomains)
            self.domainres = np.loadtxt(self.getdomains, dtype=int, ndmin=2).tolist()

        if self.domainsel is not None and len(self.domainres) == 0:
            if self.model is None:
                raise ValueError("Reference structure needed for domain selection")
            mca = self.model.select_CA()
            self.domainres = StructureSelector(self.domainsel).atom_lists(mca)

        if self.getsasa is not None and len(self.sasa) == 0:
            if self.getsasa == "auto":
                if self.model is None:
                    raise ValueError("Reference structure needed for SASA calculation")
                mca = self.model.select_CA()
                if mca.natoms() == self.model.natoms():
                    raise ValueError("Cannot calculate SASA from C-alpha coordinates")
                self.sasa = self.model.sasa_by_residue(n_sphere_points=1920)
                if self.mask_sasa_bydomain:
                    if self.domainres and len(self.domainres) > 0:
                        keep: set[int] = {i for lst in self.domainres for i in lst}
                        if keep:
                            for i in range(len(self.sasa)):
                                if i not in keep:
                                    self.sasa[i] = float(self.default_sasa)
            else:
                _ensure_readable(self.getsasa)
                self.sasa = ComponentType._read_sasa_table(self.getsasa)
            self.sasa = self._coerce_sasa_pairs(self.sasa)

        if self.getenm is not None:
            if self.getenm == "auto":
                if len(self.domainres) == 0:
                    raise ValueError("Need domain selection to calculate ENM pairs")
                if self.model is None:
                    raise ValueError("Reference structure needed for ENM pairs")
                mca = self.model.select_CA()
                enmpairs = self._findENMPairs(mca)
                self.enmpairs.extend(enmpairs)
            else:
                _ensure_readable(self.getenm)
                data = np.loadtxt(self.getenm, dtype=float)
                data = np.atleast_2d(data)
                pairs: list[tuple[int, int, Quantity]] = []
                for i, j, d in data:
                    pairs.append((int(i), int(j), float(d) * nanometer))
                self.enmpairs.extend(pairs)

        if self.nunit is None:
            if self.model is not None:
                self.nunit = self.model.nchains()
            else:
                self.nunit = 1

    def _findENMPairs(self, s) -> list[int, int, float]:
        pairs = []
        if s is None:
            return pairs

        if self.domainres is None or len(self.domainres) == 0:
            return pairs

        top = s.topology()
        atm = list(top.atoms())
        res = np.fromiter((a.residue.index for a in atm), dtype=np.int32)
        chain = np.fromiter((id(a.residue.chain) for a in atm), dtype=np.int64)

        cutoff = self.enmcutoff * nanometer
        pos = s.positions()
        for d in self.domainres:
            idx = np.array(sorted(set(d)), dtype=np.int32)
            if idx.size < 2:
                continue
            for i, j in combinations(idx, 2):
                if (abs(res[i] - res[j]) <= 2) and (chain[i] == chain[j]):
                    continue
                distance = norm(pos[i] - pos[j])
                if distance < cutoff:
                    pairs.append([i, j, distance])
        return pairs

    @staticmethod
    def _pairs_from_linear(vals: Iterable[float]) -> list[tuple[int, float]]:
        """Convert a 1D iterable of SASA values into 0-based (index, value) pairs."""
        out: list[tuple[int, float]] = []
        for i, v in enumerate(vals):
            out.append((int(i), float(v)))
        return out

    @staticmethod
    def _coerce_sasa_pairs(s: object) -> list[tuple[int, float]]:
        """
        Normalize arbitrary SASA representations to 0-based (index, value) pairs.

        Accepted:
          - list/array of floats                -> auto-indices starting at 0
          - list/array of (idx, val) pairs      -> idx normalized to 0-based if 1-based detected
          - Nx2 numpy array                     -> same as above
        """
        # None / empty
        if s is None:
            return []

        # numpy array
        if isinstance(s, np.ndarray):
            if s.ndim == 1:
                return ComponentType._pairs_from_linear(s.astype(float))
            if s.ndim == 2 and s.shape[1] == 2:
                idx = s[:, 0].astype(int)
                val = s[:, 1].astype(float)
                # infer 0- vs 1-based
                offset = 0 if idx.min() == 0 else 1
                idx = idx - offset
                return [(int(i), float(v)) for i, v in zip(idx, val)]
            raise ValueError("Unsupported numpy shape for SASA.")

        # sequence of pairs?
        if isinstance(s, Sequence) and len(s) > 0:
            first = s[0]
            # linear floats (or ints)
            if isinstance(first, (int, float, np.floating, np.integer)):
                return ComponentType._pairs_from_linear([float(x) for x in s])  # type: ignore[arg-type]
            # pair-like
            if isinstance(first, (tuple, list, np.ndarray)) and len(first) == 2:
                idx = np.asarray([p[0] for p in s], dtype=int)
                val = np.asarray([p[1] for p in s], dtype=float)
                offset = 0 if idx.min() == 0 else 1
                idx = idx - offset
                return [(int(i), float(v)) for i, v in zip(idx, val)]

        # fallback: treat as empty
        return []

    def writeout(
        self, tag: str = "none", path: Union[str | Path] = "data", *, dir: DirLike = "."
    ) -> None:
        base = Path(dir)
        base.mkdir(parents=True, exist_ok=True)

        if tag == "sasa" and self.sasa:
            idx, val = zip(*self.sasa)
            np.savetxt(base / path, np.column_stack([idx, val]), fmt="%d %.8f")

        if tag == "domains" and self.domainres:
            domain = np.asarray(self.domainres, dtype=int)
            domain = np.atleast_2d(domain)
            np.savetxt(base / path, domain, fmt="%d")

        if tag == "enm" and self.enmpairs:
            pairs = []
            for i, j, w in self.enmpairs or []:
                pairs.append((int(i), int(j), w.value_in_unit(nanometer)))
            if pairs:
                arr = np.asarray(pairs, dtype=float)
                np.savetxt(base / path, arr, fmt=["%d", "%d", "%.6f"], delimiter=" ")

    # Dense view (optional) ----------------------------------------------------
    def sasa_dense(self) -> np.ndarray:
        """
        Materialize SASA into a dense 1D array of length max_index+1.
        Missing indices are filled with 0.0.
        """
        if not self.sasa:
            return np.asarray([], dtype=float)
        max_idx = max(i for i, _ in self.sasa)
        out = np.zeros(max_idx + 1, dtype=float)
        for i, v in self.sasa:
            out[i] = v
        return out

    @staticmethod
    def _read_sasa_table(f: FileLike) -> list[tuple[int, float]]:
        """
        Read SASA from text into 0-based (index, value) pairs:
          - 1 column: contiguous SASA values per residue (in order) -> indices auto [0..N-1]
          - 2 columns: <res_index> <sasa>, where res_index can be 0- or 1-based
        """
        arr = np.loadtxt(f, ndmin=1)

        # 1-column (values only)
        if arr.ndim == 1:
            return ComponentType._pairs_from_linear(arr.astype(float, copy=False))

        # 2-column (index, value)
        if arr.ndim == 2 and arr.shape[1] == 2:
            idx = arr[:, 0].astype(int)
            val = arr[:, 1].astype(float)
            offset = 0 if idx.min() == 0 else 1
            idx = idx - offset
            return [(int(i), float(v)) for i, v in zip(idx, val)]

        raise ValueError("SASA file must have 1 column (values) or 2 columns (index value).")

    @staticmethod
    def read_list(
        path: FileLike = "component_types",
        *,
        dir: DirLike = ".",
    ) -> dict[str, ComponentType]:
        """
        Read component types from `path` (path or file-like) and return {key: ComponentType}.
        """
        base = Path(dir)
        comps: dict[str, ComponentType] = {}

        p = Path(path)
        if not p.is_file() and not p.is_absolute():
            p = base / p
        if not p.is_file():
            raise FileNotFoundError(f"component types file not found: {p}")

        list_dir = p.parent
        fh = p.open()

        try:
            for ln, line in enumerate(fh, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                tokens = _parse_line(line)

                name = tokens.get("name", tokens.get("tag", "unknown"))

                pdb = None
                if "pdb" in tokens:
                    pdb = _resolve_optional_file(tokens["pdb"], base=base, list_dir=list_dir)

                getsasa = None
                if "sasa" in tokens:
                    if tokens["sasa"] == "auto":
                        getsasa = "auto"
                    else:
                        getsasa = _resolve_optional_file(
                            tokens["sasa"], base=base, list_dir=list_dir
                        )

                domainsel = None
                if "domains" in tokens:
                    domainsel = tokens["domains"]

                getdomains = None
                if "domainfile" in tokens:
                    getdomains = _resolve_optional_file(
                        tokens["domainfile"], base=base, list_dir=list_dir
                    )

                getenm = None
                if "enm" in tokens:
                    if tokens["enm"] == "auto":
                        getenm = "auto"
                    else:
                        getenm = _resolve_optional_file(tokens["enm"], base=base, list_dir=list_dir)

                enmcutoff = None
                if "enmcutoff" in tokens:
                    enmcutoff = float(tokens["enmcutoff"])

                ct = ComponentType(
                    name=name,
                    pdb=pdb,
                    getsasa=getsasa,
                    domainsel=domainsel,
                    getdomains=getdomains,
                    getenm=getenm,
                    enmcutoff=enmcutoff,
                )

                comps[name] = ct
        finally:
            fh.close()

        return comps


@dataclass
class Component:
    ctype: ComponentType  # component type
    segment: list[str] = field(default_factory=list)  # list of segment names

    def __repr__(self) -> str:
        return f"<component {self.ctype.name!r} with segments {self.segment}>"


@dataclass
class Interaction:
    pairs: list[tuple[int, int]] = field(default_factory=list)  # global CA indices (i, j)
    strength: float = 1.0  # force constant (computed from probability)
    distance: float = 0.8  # reference distance [nm]
    additive: bool = True  # pairs are additive
    function: str = "switch"  # 'switch', 'Go', 'harmonic'
    parameter: float = 0.0  # extra parameter depending on function

    # Optional angle gating (used for ring-like components)
    angle: bool = False
    angle_def: float = float(np.pi / 2.0)  # radians (90 deg default)
    pgate: float = 2.0

    # Optional centroid groups (global CA indices) defining ring frames
    # A groups: (center, subunit1, subunit2)
    frameA_center: list[int] = field(default_factory=list)
    frameA1: list[int] = field(default_factory=list)
    frameA2: list[int] = field(default_factory=list)

    # B groups: (center, subunit1, subunit2)
    frameB_center: list[int] = field(default_factory=list)
    frameB1: list[int] = field(default_factory=list)
    frameB2: list[int] = field(default_factory=list)


@dataclass
class InteractionSet:

    ctypeA: ComponentType = None
    ctypeB: ComponentType = None

    interactions: Optional[list[Interaction]] = field(default_factory=list)

    mode: Optional[str] = "additive"  # 'additive' or 'exclusive'
    function: Optional[str] = "switch"  # 'Go', 'harmonic', 'switch'
    parameter: Optional[float] = 0.0  # extra parameter depending on function
    value: Optional[str] = "asis"  # 'asis', 'logit', 'neglog'
    scale: Optional[float] = 1.0
    offset: Optional[float] = 0.0

    # Optional orientation gate for ring-like components
    angle: bool = False
    angle_def: float = float(np.pi / 2.0)  # radians
    pgate: float = 2.0

    defdist: Optional[float] = 0.8
    defprob: Optional[float] = 1.0

    contacttable: Optional[str] = None
    options: Optional[str] = None

    _contact_search_dirs: list[Path] = field(default_factory=list, repr=False)

    def __repr__(self) -> str:
        msg = f"<interaction set {self.ctypeA.name!r} - {self.ctypeB.name!r}"
        msg += f" with {len(self.interactions or [])} interactions>"
        return msg

    def __post_init__(self) -> None:
        # Parse options (comma-separated; parentheses may contain commas)
        if self.options:
            for opt in _split_options(self.options):
                if not opt:
                    continue
                name, vals = _parse_option_values(opt)
                lname = name.lower()
                v0 = float(vals[0]) if vals else 0.0

                if lname == "exclusive":
                    self.mode = "exclusive"
                elif lname in {"additive", "all"}:
                    self.mode = "additive"
                elif lname == "switch":
                    self.function = "switch"
                    self.parameter = v0
                elif lname == "go":
                    self.function = "Go"
                    self.parameter = v0
                elif lname == "harmonic":
                    self.function = "harmonic"
                    self.parameter = v0
                elif lname in {"asis", "logit", "neglog"}:
                    self.value = lname
                    self.offset = v0
                elif lname == "distance" and v0 > 0.0:
                    self.defdist = v0
                elif lname in {"prob", "probability"} and v0 > 0.0:
                    self.defprob = v0
                elif lname == "scale" and v0 > 0.0:
                    self.scale = v0
                elif lname == "angle":
                    self.angle = True
                    if vals:
                        self.angle_def = _coerce_angle(vals[0])
                    if len(vals) > 1 and float(vals[1]) > 0.0:
                        self.pgate = float(vals[1])

        if self.contacttable:
            self.interactions = self._read_contact_table(self.contacttable)

    # --- contact table reader -------------------------------------------------
    def _read_contact_table(self, path: FileLike) -> list[Interaction]:
        """
        Read a contact table where each non-empty, non-comment line has one or more
        comma-separated columns. First two fields are residue selections (A, B);
        optional 3rd = probability; optional 4th = distance (nm).
        Columns after the first inherit prob/dist from column 1 unless overridden.
        Columns with identical (prob, dist) on the same line are merged into one
        Interaction (pairs union); different (prob, dist) create separate Interactions.
        """
        if self.ctypeA is None or self.ctypeB is None:
            raise ValueError("InteractionSet requires ctypeA and ctypeB")
        if self.ctypeA.model is None or self.ctypeB.model is None:
            raise ValueError("Both component types must have reference models to build pairs")

        p = Path(path)

        # Search order required by user:
        #  - absolute (handled in resolver)
        #  - relative to CWD
        #  - relative to dir= passed to read_list (base)
        #  - relative to directory containing the interactions file
        search_dirs = list(self._contact_search_dirs) if self._contact_search_dirs else []
        if not search_dirs:
            search_dirs = [Path.cwd()]

        resolved = _resolve_path_candidates(p, search_dirs)
        _ensure_readable(resolved)
        fh = resolved.open()  # type: ignore[assignment]

        interactions: list[Interaction] = []

        try:
            for ln, raw in enumerate(fh, 1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue

                # Split into "columns" (whitespace-separated). Each column then has comma fields.
                cols = [c for c in line.split() if c]
                if not cols:
                    continue

                # Parse first column for defaults
                first = self._parse_column(cols[0], lineno=ln)
                base_prob = self.defprob if first.prob is None else first.prob
                base_dist = self.defdist if first.dist is None else first.dist

                # Build a list of per-column parsed specs with inherited defaults
                parsed = []
                for col in cols:
                    spec = self._parse_column(col, lineno=ln)
                    prob = base_prob if spec.prob is None else spec.prob
                    dist = base_dist if spec.dist is None else spec.dist
                    parsed.append((spec.selA, spec.selB, prob, dist))

                # Group columns by (prob, dist) so alternates with same parameters merge
                groups: dict[tuple[float, float], list[tuple[str, str]]] = {}
                for selA, selB, prob, dist in parsed:
                    key = (float(prob), float(dist))
                    groups.setdefault(key, []).append((selA, selB))

                # For each group, compute union of pairs and create Interaction
                for (prob, dist), selections in groups.items():
                    pair_set: set[tuple[int, int]] = set()
                    for selA, selB in selections:
                        pairs = self._pairs_from_selections(selA, selB)
                        pair_set.update(pairs)

                    if not pair_set:
                        continue

                    strength = self._strength_from_probability(prob)
                    interactions.append(
                        Interaction(
                            pairs=sorted(pair_set),
                            strength=strength,
                            distance=float(dist),
                            additive=(self.mode == "additive"),
                            function=self.function,
                            parameter=self.parameter,
                            angle=bool(self.angle),
                            angle_def=float(self.angle_def),
                            pgate=float(self.pgate),
                        )
                    )
        finally:
            fh.close()

        return interactions

    # Helper: one comma-delimited column
    @dataclass
    class _ColSpec:
        selA: str
        selB: str
        prob: Optional[float]
        dist: Optional[float]

    def _parse_column(self, col: str, *, lineno: int) -> _ColSpec:
        parts = [p for p in col.split(",") if p != ""]
        if len(parts) < 2:
            msg = f"contact table line {lineno}: need at least two fields 'A,B[,prob[,dist]]'"
            raise ValueError(msg)
        selA = parts[0].strip()
        selB = parts[1].strip()
        prob = float(parts[2]) if len(parts) >= 3 else None
        dist = float(parts[3]) if len(parts) >= 4 else None
        return InteractionSet._ColSpec(selA=selA, selB=selB, prob=prob, dist=dist)

    # Helper: selection -> all (i,j) combinations
    def _pairs_from_selections(self, selA: str, selB: str) -> list[tuple[int, int]]:
        # StructureSelector(...).atom_lists(model) returns lists of residue indices per selection.
        listsA = StructureSelector(selA).atom_lists(self.ctypeA.model.select_CA())
        listsB = StructureSelector(selB).atom_lists(self.ctypeB.model.select_CA())

        pairs: list[tuple[int, int]] = []
        for la in listsA or []:
            for lb in listsB or []:
                for i in la:
                    for j in lb:
                        if i == j and self.ctypeA is self.ctypeB:
                            # allow same-index across different components;
                            # skip strict self-self within identical model indexing
                            pass
                        pairs.append((int(i), int(j)))
        return pairs

    # Helper: probability -> strength with transform, offset, scale, clamp>=0
    def _strength_from_probability(self, p: float) -> float:
        # guard against exact 0/1 for log/ratio
        eps = 1e-12
        x = float(p)
        if self.value == "logit":
            x = log(min(max(p, eps), 1.0 - eps) / max(1.0 - p, eps))
        elif self.value == "neglog":
            x = -log(max(p, eps))
        elif self.value == "asis":
            x = p
        else:
            # fallback to 'asis' if unknown
            x = p
        y = (x + float(self.offset)) * float(self.scale)
        return y if y > 0.0 else 0.0

    @staticmethod
    def read_list(
        ctypes: dict[str, ComponentType],
        path: FileLike = "interactions",
        *,
        dir: DirLike = ".",
    ) -> dict[str, InteractionSet]:
        """
        Read interaction sets from `path` (path or file-like) and return {key: InteractionSet}.
        """
        base = Path(dir)
        ints: dict[str, InteractionSet] = {}

        p = Path(path)
        if not p.is_file() and not p.is_absolute():
            p = base / p
        if not p.is_file():
            raise FileNotFoundError(f"interactions file not found: {p}")
        fh = p.open()
        interactions_file_dir = p.parent

        try:
            for ln, line in enumerate(fh, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    raise ValueError(f"{path}:{ln}: expected at least 3 columns, got {len(parts)}")

                tag1, tag2, ftable = parts[0], parts[1], parts[2]

                if tag1 not in ctypes:
                    raise ValueError(f"{tag1} not in component types")
                if tag2 not in ctypes:
                    raise ValueError(f"{tag2} not in component types")

                ptable = Path(ftable)  # keep raw; resolve later with full search logic

                if len(parts) > 3:
                    options = parts[3]
                else:
                    options = None

                search_dirs = [
                    Path.cwd(),  # relative to current directory
                    base,  # relative to dir= passed to read_list
                    interactions_file_dir,  # relative to interactions file location
                ]

                intset = InteractionSet(
                    ctypeA=ctypes[tag1],
                    ctypeB=ctypes[tag2],
                    contacttable=ptable,
                    options=options,
                    _contact_search_dirs=search_dirs,
                )

                key = f"{tag1}.{tag2}"
                ints[key] = intset
        finally:
            fh.close()

        return ints


@dataclass
class Assembly:
    component: list[Component] = field(default_factory=list)
    ctype: dict[str, ComponentType] = field(default_factory=dict)
    interact: dict[str, InteractionSet] = field(default_factory=dict)
    model: Union[Structure, Model, None] = None

    # data structures for precomputing globally mapped values
    sasa: list[float] = field(default_factory=list)
    domains: list[list[int]] = field(default_factory=list)
    interactions: list[Interaction] = field(default_factory=list)

    def __repr__(self) -> str:
        msg = f"<assembly has {len(self.component)} components, {len(self.ctype)} types"
        msg += f" and {len(self.interact)} interaction sets>"
        return msg

    __str__ = __repr__

    def __init__(
        self,
        components: Union[FileLike, None] = None,
        types: Union[FileLike, None] = None,
        *,
        structure: Union[Structure, Model, FileLike, None] = None,
        interactions: Union[FileLike, None] = None,
        dir: DirLike = ".",
    ) -> None:
        """
        If `components` and `types` are provided, load an Assembly from disk,
        if a model is also provided, store it.
        Otherwise, create an empty Assembly.
        """
        self.component = []
        self.ctype = {}  # dict[str, ComponentType]
        self.interact = {}
        self.model = None
        self.sasa = []
        self.domains = []
        self.enmpairs = []
        self.interactions = []

        if (components is None) ^ (types is None):
            raise ValueError("Provide both components and types file names, or neither.")

        if structure is not None:
            if isinstance(structure, Structure):
                if structure.models:
                    self.model = structure.models[0].select_CA()
            elif isinstance(structure, Model):
                self.model = structure.select_CA()
            else:
                spath = Path(structure)
                if not spath.is_absolute():
                    spath = Path(dir) / spath
                _ensure_readable(spath)
                self.model = PDBReader(spath).models[0].select_CA()

        if components is not None and types is not None:
            self.read_components(components, types, dir=dir)
        if interactions is not None:
            self.read_interactions(interactions, dir=dir)

    def components(self) -> Iterator[Component]:
        return iter(self.component)

    def types(self) -> Iterator[ComponentType]:
        return iter(self.ctype.values())

    def add_type(self, ctype: ComponentType) -> None:
        # insert or keep existing by name
        if ctype.name not in self.ctype:
            self.ctype[ctype.name] = ctype

    def add_interactions(self, intset: dict[str, InteractionSet]) -> None:
        for k in intset.keys():
            if k not in self.interact:
                self.interact[k] = intset[k]

    def add_component(self, comp: Component) -> None:
        self.add_type(comp.ctype)
        self.component.append(comp)

    def components_by_type(self) -> dict[str, list[Component]]:
        """Group components by type name."""
        groups: dict[str, list[Component]] = {}
        for comp in self.component:
            groups.setdefault(comp.ctype.name, []).append(comp)
        return groups

    def find_type(self, name: str) -> Optional[ComponentType]:
        return self.ctype.get(name)

    def segments(self) -> list[str]:
        """Flattened list of all segment names in insertion order."""
        out: list[str] = []
        for comp in self.component:
            out.extend(comp.segment)
        return out

    def read_interactions(
        self,
        inter_file: FileLike,
        *,
        dir: DirLike = ".",
    ) -> None:
        base = Path(dir)
        inter_path = Path(inter_file)
        if not inter_path.is_absolute():
            inter_path = base / inter_path
        _ensure_readable(inter_path)
        self.add_interactions(InteractionSet.read_list(self.ctype, inter_path, dir=base))
        return

    def read_components(
        self,
        comp_file: FileLike,
        type_file: FileLike,
        *,
        dir: DirLike = ".",
    ) -> None:
        """
        Build an Assembly from a component list file and a component-type file.
        """
        base = Path(dir)

        comp_path = Path(comp_file)
        if not comp_path.is_absolute():
            comp_path = base / comp_path
        _ensure_readable(comp_path)
        comp_source = comp_path

        type_path = Path(type_file)
        if not type_path.is_absolute():
            type_path = base / type_path
        _ensure_readable(type_path)
        types = ComponentType.read_list(type_path, dir=base)

        fh = Path(comp_source).open()
        try:
            for ln, line in enumerate(fh, 1):
                raw = line.strip()
                if not raw or raw.startswith("#"):
                    continue

                parts = raw.split()
                if len(parts) != 2:
                    msg = f"{comp_source}:{ln}: expected 2 columns "
                    msg += f" '<segA:segB:...> <type_tag>', got {len(parts)}"
                    raise ValueError(msg)

                seg_field, tag = parts
                segments = [s for s in seg_field.split(":") if s]
                if tag not in types:
                    msg = f"{comp_source}:{ln}: unknown component type tag '{tag}' "
                    msg += f" (not found in '{type_file}')"
                    raise KeyError(msg)

                ctype = types[tag]
                comp = Component(ctype=ctype, segment=segments)
                self.add_component(comp)
        finally:
            fh.close()

        return self

    def get_sasa(
        self,
        *,
        mol: Union[Structure, Model, None] = None,
        mask_by_domain: bool = False,
        default_sasa: Optional[float] = None,
    ) -> list[float]:
        """
        Returns global SASA list combined from components
        """
        if not self.sasa or len(self.sasa) == 0:
            self._global_sasa(
                mol=mol,
                mask_by_domain=mask_by_domain,
                default_sasa=999.0 if default_sasa is None else float(default_sasa),
            )
        return self.sasa

    def get_domains(
        self,
        *,
        mol: Union[Structure, Model, None] = None,
    ) -> list[list[int]]:
        """
        Returns global domain list combined from components
        """
        if not self.domains or len(self.domains) == 0:
            self._global_domains(mol=mol)
        return self.domains

    def get_enmpairs(
        self,
        *,
        mol: Union[Structure, Model, None] = None,
    ) -> list[list[int]]:
        """
        Returns global ENM pair list combined from components
        """
        if not self.enmpairs or len(self.enmpairs) == 0:
            self._global_enmpairs(mol=mol)
        return self.enmpairs

    def get_interactions(
        self,
        *,
        mol: Union[Structure, Model, None] = None,
    ) -> list[Interaction]:
        """
        Returns global interaction pair list combined from components
        """
        if not self.interactions or len(self.interactions) == 0:
            self._global_interactions(mol=mol)
        return self.interactions

    # -- helper: build ordered segment info for a component (only those present) --
    @staticmethod
    def _comp_seginfo(
        segments: list[str],
        seg_offsets: dict[str, tuple[int, int]],
    ) -> list[tuple[str, int, int]]:
        """Return list of (seg, global_start, n_res) for segments that exist, in order."""
        out: list[tuple[str, int, int]] = []
        for seg in segments:
            if seg in seg_offsets:
                start, nres = seg_offsets[seg]
                out.append((seg, start, nres))
        return out

    @staticmethod
    def _map_component_index(
        local_idx: int,
        seginfo: list[tuple[str, int, int]],
    ) -> Optional[int]:
        """
        Map a per-component residue index (0-based within the component's
        concatenated segments) to the global CA index.
        Returns None if the index falls outside present segments.
        """
        i = int(local_idx)
        for _seg, start, nres in seginfo:
            if i < nres:
                return start + i
            i -= nres
        return None

    def _ensure_model(self, mol: Union[Structure, Model, None]) -> Model:
        """Normalize/choose CA-only model."""
        if mol is not None:
            self.model = (
                mol.models[0] if isinstance(mol, Structure) and mol.models else mol
            ).select_CA()
        if self.model is None:
            raise ValueError("Need model to combine domains")
        return self.model

    @staticmethod
    def _seg_offsets(model: Model) -> tuple[dict[str, tuple[int, int]], int]:
        """Return per-segment global start and length, plus total residues."""
        seg_offsets: dict[str, tuple[int, int]] = {}
        total = 0
        for ch in model.chains():
            n_res = len(ch.residues)
            seg = getattr(ch, "key_id", None)
            if seg is None:
                continue
            seg_offsets[str(seg)] = (total, n_res)
            total += n_res
        return seg_offsets, total

    def _iter_comp_maps(
        self, seg_offsets: dict[str, tuple[int, int]]
    ) -> Iterator[tuple[ComponentType, callable]]:
        """
        Yield (ctype, map_idx) for each component present in the model, where
        map_idx(local_idx) -> global CA index or None.
        """
        for comp in self.component:
            seginfo = Assembly._comp_seginfo(comp.segment, seg_offsets)
            if not seginfo:
                continue
            yield comp.ctype, (
                lambda i, seginfo=seginfo: Assembly._map_component_index(int(i), seginfo)
            )

    @staticmethod
    def _unique_in_order(xs: Iterable[int]) -> list[int]:
        seen: set[int] = set()
        out: list[int] = []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def _global_domains(self, *, mol: Union[Structure, Model, None] = None) -> None:
        model = self._ensure_model(mol)
        seg_offsets, total = self._seg_offsets(model)

        domain_global: list[list[int]] = []
        if total == 0:
            self.domains = domain_global
            return

        for ctype, map_idx in self._iter_comp_maps(seg_offsets):
            if not ctype.domainres:
                continue
            for sel in ctype.domainres:
                mapped = (map_idx(i) for i in sel)
                kept = [g for g in mapped if g is not None]
                uniq = Assembly._unique_in_order(kept)
                if uniq:
                    domain_global.append(uniq)

        self.domains = domain_global

    def _global_enmpairs(
        self,
        *,
        mol: Union[Structure, Model, None] = None,
    ) -> None:
        model = self._ensure_model(mol)
        seg_offsets, total = self._seg_offsets(model)

        pairs: list[tuple[int, int, Quantity]] = []
        if total == 0:
            self.enmpairs = pairs
            return

        seen: set[tuple[int, int]] = set()

        for ctype, map_idx in self._iter_comp_maps(seg_offsets):
            for ii, jj, dd in ctype.enmpairs or []:
                gi = map_idx(ii)
                gj = map_idx(jj)
                if gi is None or gj is None:
                    continue
                a, b = (gi, gj) if gi <= gj else (gj, gi)
                key = (a, b)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((a, b, dd))

        self.enmpairs = pairs

    def _global_interactions(
        self,
        *,
        mol: Union[Structure, Model, None] = None,
    ) -> None:
        model = self._ensure_model(mol)
        seg_offsets, total = self._seg_offsets(model)

        interactions: list[Interaction] = []
        if total == 0 or not self.interact:
            self.interactions = interactions
            return

        SegInfo = list[tuple[str, int, int]]

        def _make_map_fn(seginfo: SegInfo) -> Callable[[int], Optional[int]]:
            def map_fn(i: int) -> Optional[int]:
                return Assembly._map_component_index(int(i), seginfo)

            return map_fn

        CompRec = tuple[int, Component, SegInfo, Callable[[int], Optional[int]]]
        maps_by_type: dict[str, list[CompRec]] = {}

        comp_key = -1
        for comp in self.component:
            seginfo = Assembly._comp_seginfo(comp.segment, seg_offsets)
            if not seginfo:
                continue
            comp_key += 1
            maps_by_type.setdefault(comp.ctype.name, []).append(
                (comp_key, comp, seginfo, _make_map_fn(seginfo))
            )

        stride = 5
        frame_cache: dict[int, tuple[list[int], list[int], list[int]]] = {}

        def _subsample(vals: list[int]) -> list[int]:
            if stride <= 1:
                return vals
            return vals[::stride]

        def _frame_groups(rec: CompRec) -> tuple[list[int], list[int], list[int]]:
            key, comp, seginfo, map_fn = rec
            cached = frame_cache.get(key)
            if cached is not None:
                return cached

            if len(seginfo) < 2:
                raise ValueError("angle gating requires components with >=2 segments")

            dom: list[int] = []
            if comp.ctype.domainres:
                local = sorted({int(i) for lst in comp.ctype.domainres for i in lst})
                for i in local:
                    g = map_fn(i)
                    if g is not None:
                        dom.append(int(g))
            else:
                for _seg, start, nres in seginfo:
                    dom.extend(range(start, start + nres))

            dom = sorted(set(dom))
            if not dom:
                for _seg, start, nres in seginfo:
                    dom.extend(range(start, start + nres))
                dom = sorted(set(dom))
            if not dom:
                raise ValueError("empty component; cannot build ring frame")

            center = _subsample(dom) or [dom[0]]

            def _subunit(k: int) -> list[int]:
                _seg, start, nres = seginfo[k]
                stop = start + nres
                sub = [x for x in dom if start <= x < stop]
                if not sub:
                    sub = list(range(start, stop))
                sub = _subsample(sub)
                if not sub and nres > 0:
                    sub = [start]
                return sub

            sub1 = _subunit(0)
            sub2 = _subunit(1)
            if not sub1 or not sub2:
                raise ValueError("failed to build subunit frame groups for angle gating")

            frame_cache[key] = (center, sub1, sub2)
            return center, sub1, sub2

        for intset in self.interact.values():
            if not intset or not intset.ctypeA or not intset.ctypeB:
                continue

            nameA = intset.ctypeA.name
            nameB = intset.ctypeB.name

            listA = maps_by_type.get(nameA, [])
            listB = maps_by_type.get(nameB, [])
            if not listA or not listB:
                continue

            templates = intset.interactions or []
            if not templates:
                continue

            want_angle = bool(getattr(intset, "angle", False))
            angle_def = float(getattr(intset, "angle_def", float(np.pi / 2.0)))
            pgate = float(getattr(intset, "pgate", 2.0))

            if nameA == nameB:
                comp_pairs = combinations(listA, 2)
            else:
                comp_pairs = ((a, b) for a in listA for b in listB)

            for recA, recB in comp_pairs:
                if want_angle:
                    a_cen, a1, a2 = _frame_groups(recA)
                    b_cen, b1, b2 = _frame_groups(recB)
                else:
                    a_cen, a1, a2 = [], [], []
                    b_cen, b1, b2 = [], [], []

                _keyA, _compA, _segA, mapA = recA
                _keyB, _compB, _segB, mapB = recB

                for tmpl in templates:
                    pair_set: set[tuple[int, int]] = set()
                    for i_local, j_local in tmpl.pairs or []:
                        gi = mapA(i_local)
                        gj = mapB(j_local)
                        if gi is None or gj is None or gi == gj:
                            continue
                        pair_set.add((int(gi), int(gj)))

                    if not pair_set:
                        continue

                    interactions.append(
                        Interaction(
                            pairs=sorted(pair_set),
                            strength=float(tmpl.strength),
                            distance=float(tmpl.distance),
                            additive=bool(tmpl.additive),
                            function=str(tmpl.function),
                            parameter=float(tmpl.parameter),
                            angle=want_angle,
                            angle_def=angle_def,
                            pgate=pgate,
                            frameA_center=a_cen,
                            frameA1=a1,
                            frameA2=a2,
                            frameB_center=b_cen,
                            frameB1=b1,
                            frameB2=b2,
                        )
                    )

        self.interactions = interactions

    def _global_sasa(
        self,
        *,
        mol: Union[Structure, Model, None] = None,
        mask_by_domain: bool = False,
        default_sasa: float = 999.0,
    ) -> None:
        #        print(f"global_sasa mask: {mask_by_domain}")
        model = self._ensure_model(mol)
        seg_offsets, total = self._seg_offsets(model)

        sasa = [float(default_sasa)] * total
        if total == 0:
            self.sasa = sasa
            return

        for ctype, map_idx in self._iter_comp_maps(seg_offsets):
            for idx, val in ctype.sasa or []:
                g = map_idx(idx)
                if g is not None and 0 <= g < total:
                    sasa[g] = float(val)

        if mask_by_domain:
            if not self.domains:
                self._global_domains(mol=model)
            if self.domains and len(self.domains) > 0:
                keep: set[int] = {i for lst in self.domains for i in lst}
                if keep:
                    for i in range(total):
                        if i not in keep:
                            sasa[i] = float(default_sasa)
                else:
                    sasa = [float(default_sasa)] * total

        self.sasa = sasa


def _resolve_path_candidates(p: Path, search_dirs: Sequence[Path]) -> Path:
    """
    Resolve `p` by trying:
      - absolute p (as-is)
      - then p relative to each directory in search_dirs (in order)
    Returns the first existing file path, otherwise raises FileNotFoundError
    with a helpful message.
    """
    if p.is_absolute():
        if p.is_file():
            return p
        raise FileNotFoundError(p)

    tried: list[Path] = []

    # Try as provided relative to CWD (first search dir may be Path.cwd()).
    for d in search_dirs:
        cand = d / p
        tried.append(cand)
        if cand.is_file():
            return cand

    msg = "Contact table not found. Tried:\n" + "\n".join(f"  - {t}" for t in tried)
    raise FileNotFoundError(msg)


def _resolve_optional_file(
    raw: str,
    *,
    base: Path,
    list_dir: Path,
) -> str:
    """
    Resolve a possibly-relative file path appearing inside a list file.

    Search order:
      1) as given (relative to CWD)
      2) relative to base (dir=)
      3) relative to list file directory
    """
    p = Path(raw)

    if p.is_absolute():
        return str(p)

    # 1) CWD-relative (as written)
    if p.is_file():
        return str(p)

    # 2) dir= base
    cand = base / p
    if cand.is_file():
        return str(cand)

    # 3) list file location
    cand = list_dir / p
    if cand.is_file():
        return str(cand)

    # Fallback (keeps previous behavior; will error later if unreadable)
    return str(base / p)


def _ensure_readable(f: FileLike) -> None:
    p = Path(f)
    if not p.is_file():
        raise FileNotFoundError(p)


def _parse_line(line: str) -> dict[str, str]:
    """
    Parse a line of space-separated tokens into a dict.
    - First token may be bare (no '='); it will be stored as key 'tag'.
    - All remaining tokens must be key=value; values may be quoted.

    Examples
    --------
    >>> parse_line("mytag a=1 b=two c='three words'")
    {'tag': 'mytag', 'a': '1', 'b': 'two', 'c': 'three words'}
    >>> parse_line("a=1 b=2")
    {'a': '1', 'b': '2'}
    >>> parse_line("onlytag")
    {'tag': 'onlytag'}
    """
    tokens = shlex.split(line, posix=True)
    out: dict[str, str] = {}
    if not tokens:
        return out

    idx = 0
    if "=" not in tokens[0]:
        out["tag"] = tokens[0]
        idx = 1

    for tok in tokens[idx:]:
        if "=" not in tok:
            raise ValueError(f"Invalid token without '=': {tok!r}")
        k, v = tok.split("=", 1)
        out[k] = v
    return out


def _split_options(s: str) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    depth = 0

    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)

        if ch == "," and depth == 0:
            tok = "".join(buf).strip()
            if tok:
                out.append(tok)
            buf = []
            continue

        buf.append(ch)

    tok = "".join(buf).strip()
    if tok:
        out.append(tok)

    return out


def _parse_option_values(s: str) -> tuple[str, list[float]]:
    s = s.strip()
    if "(" not in s:
        return s, []

    if not s.endswith(")"):
        raise ValueError(f"Invalid option(value): {s!r}")

    name, inner = s.split("(", 1)
    name = name.strip()
    if not name:
        raise ValueError("Empty option name")

    inner = inner[:-1].strip()  # drop trailing ')'
    if not inner:
        return name, []

    vals: list[float] = []
    for part in inner.split(","):
        part = part.strip()
        if part:
            vals.append(float(part))

    return name, vals


def _coerce_angle(v: float) -> float:
    x = float(v)
    if x > 2.0 * float(np.pi):
        return float(np.deg2rad(x))
    return x


def _parse_option_value(s: str) -> tuple[str, float]:
    name, vals = _parse_option_values(s)
    v0 = float(vals[0]) if vals else 0.0
    return name, v0
