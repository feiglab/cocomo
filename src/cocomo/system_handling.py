from __future__ import annotations

import io
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

from cocomo import (
    DomainSelector,
    Model,
    PDBReader,
    Structure,
)

FileLike = Union[str, Path, io.BytesIO, io.StringIO]

# --- Data containers ---------------------------------------------------------


@dataclass
class ComponentType:
    name: str = "unknown"  # e.g. "hexamer"
    nunit: Optional[int] = None  # number of individual units
    domainres: list[list[int]] = field(default_factory=list)  # residue selection for domain
    sasa: list[tuple[int, float]] = field(default_factory=list)  # SASA per residue

    sasapdb: Optional[FileLike] = None  # read PDB and calculate SASA
    sasafile: Optional[FileLike] = None  # read SASA from file

    domainsel: list[str] = field(default_factory=list)  # domain selection from strings
    domainpdb: Optional[FileLike] = None  # PDB for parsing selection
    domainmol: Optional[Union[Structure | Model]] = None  # molecule for parsing selection
    domainfile: Optional[FileLike] = None  # read domain lists from file

    def __repr__(self) -> str:
        r = f"<component type {self.name!r} with {self.nunit} units"
        r += f", SASA: {len(self.sasa)}, domain entries: {len(self.domainres)}>"
        return r

    def __post_init__(self) -> None:
        spdb = None
        sca = None

        if self.sasapdb is not None and len(self.sasa) == 0:
            _ensure_readable(self.sasapdb)
            spdb = PDBReader(self.sasapdb)
            self.sasa = spdb.sasa_by_residue(n_sphere_points=1920)

        if self.sasafile is not None and len(self.sasa) == 0:
            _ensure_readable(self.sasafile)
            self.sasa = ComponentType._read_sasa_table(self.sasafile)

        if self.domainmol is not None:
            sca = self.domainmol.select_CA()
        elif self.domainpdb is not None:
            _ensure_readable(self.domainpdb)
            sca = PDBReader(self.domainpdb).select_CA()
        elif spdb is not None:
            sca = spdb.select_CA()

        if len(self.domainsel) > 0 and len(self.domainres) == 0:
            if sca is not None:
                self.domainres = DomainSelector(self.domainsel).atom_lists(sca)
            else:
                raise ValueError("No structure available for selection.")

        if self.domainfile is not None and len(self.domainres) == 0:
            _ensure_readable(self.domainfile)
            self.domainres = np.loadtxt(self.domainfile, dtype=int, ndmin=2).tolist()

        self.sasa = self._coerce_sasa_pairs(self.sasa)

        if self.nunit is None:
            if sca is not None:
                self.nunit = sca.nchains()
            else:
                self.nunit = 1

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
        path: Union[str | Path] = "component_types",
        *,
        writeout: bool = False,
        dir: Union[str | Path] = ".",
    ) -> dict[str, ComponentType]:
        """
        Read component types from `path` and return {key: ComponentType}.
        Supports two formats:

        1) PDB + selection
           key <tag>.pdb <selection>
           e.g.  hexamer h1.pdb A:B:C:D:E:F.2-91

        2) Precomputed files
           key <tag>.surface <tag>.domainsel
           e.g.  hexamer h1.surface h1.domainsel

        """
        path = Path(path)
        base = Path(dir)
        comps: dict[str, ComponentType] = {}

        if not path.is_file():
            raise FileNotFoundError(f"component types file not found: {path}")

        if writeout:
            base.mkdir(parents=True, exist_ok=True)

        with path.open() as fh:
            for ln, line in enumerate(fh, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) < 3:
                    raise ValueError(f"{path}:{ln}: expected 3 columns, got {len(parts)}")

                key, col2, col3 = parts[0], parts[1], parts[2]

                # --- Format 1: "<tag>.pdb  <selection>"
                if col2.endswith(".pdb"):
                    tag = Path(col2).stem
                    selection = col3
                    ct = ComponentType(
                        name=key,
                        sasapdb=str(base / f"{tag}.pdb"),
                        domainsel=selection,
                    )
                    comps[key] = ct

                    if writeout:
                        if ct.sasa:
                            idx, val = zip(*ct.sasa)
                            np.savetxt(
                                base / f"{tag}.surface",
                                np.column_stack([idx, val]),
                                fmt="%d %.8f",
                            )
                        else:
                            # empty
                            open(base / f"{tag}.surface", "w").close()

                        # ct.domainsel: handle 1D/2D ints
                        domain = np.asarray(ct.domainres, dtype=int)
                        domain = np.atleast_2d(domain)
                        np.savetxt(base / f"{tag}.domainsel", domain, fmt="%d")

                # --- Format 2: "<tag>.surface  <tag>.domainsel"
                else:
                    ct = ComponentType(
                        name=key,
                        sasafile=str(base / f"{col2}"),
                        domainfile=str(base / f"{col3}"),
                    )
                    comps[key] = ct
        return comps


@dataclass
class Component:
    ctype: ComponentType  # component type
    segment: list[str] = field(default_factory=list)  # list of segment names

    def __repr__(self) -> str:
        return f"<component {self.ctype.name!r} with segments {self.segment}>"


@dataclass
class Assembly:
    component: list[Component] = field(default_factory=list)
    ctype: list[ComponentType] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"<assembly has {len(self.component)} components and {len(self.ctype)} types>"

    __str__ = __repr__

    def __init__(
        self,
        components: Union[FileLike, None] = None,
        types: Union[FileLike, None] = None,
        *,
        dir: Union[str, Path] = ".",
    ) -> None:
        """
        If `components` and `types` are provided, load an Assembly from disk.
        Otherwise, create an empty Assembly.
        """

        self.component = []
        self.ctype = []

        if (components is None) ^ (types is None):
            raise ValueError("Provide both components and types file names, or neither.")
        if components is not None and types is not None:
            self.read_components(components, types, dir=dir)

    def components(self) -> Iterator[Component]:
        return iter(self.component)

    def types(self) -> Iterator[ComponentType]:
        return iter(self.ctype)

    def add_type(self, ctype: ComponentType) -> None:
        if any(t.name == ctype.name for t in self.ctype):
            return
        self.ctype.append(ctype)

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
        for t in self.ctype:
            if t.name == name:
                return t
        return None

    def segments(self) -> list[str]:
        """Flattened list of all segment names in insertion order."""
        out: list[str] = []
        for comp in self.component:
            out.extend(comp.segment)
        return out

    def read_components(
        self,
        comp_file: FileLike,
        type_file: FileLike,
        *,
        dir: Union[str, Path] = ".",
    ) -> Assembly:
        """
        Build an Assembly from a component list file and a component-type file.
        """

        base = Path(dir)

        if _is_fileobj(comp_file):
            comp_source = comp_file
        else:
            comp_path = Path(comp_file)
            if not comp_path.is_absolute():
                comp_path = base / comp_path
            _ensure_readable(comp_path)
            comp_source = comp_path

        if _is_fileobj(type_file):
            type_source = type_file
            types = ComponentType.read_list(type_source, dir=base)
        else:
            type_path = Path(type_file)
            if not type_path.is_absolute():
                type_path = base / type_path
            _ensure_readable(type_path)
            types = ComponentType.read_list(type_path, dir=base)

        if _is_fileobj(comp_source):
            fh = comp_source
            _need_close = False
        else:
            fh = Path(comp_source).open()
            _need_close = True
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
            if _need_close:
                fh.close()

    def global_sasa_domains(
        self,
        mol: Union[Structure, Model],
        *,
        mask_by_domain: bool = False,
        default_sasa: float = 999.0,
    ) -> tuple[list[float], list[list[int]]]:
        """
        Build composite SASA (1D list) and domain selections (list of index lists) for `mol`.

        Returns
        -------
        sasa : list[float]
            Length equals total residues across all segments/chains of the model.
        domains : list[list[int]]
            One list per component *per domain selection list* in its ComponentType.
        """
        # -- normalize to a single Model --
        if isinstance(mol, Structure):
            if not mol.models:
                return ([], [])
            model = mol.models[0]
        else:
            model = mol

        # -- per-segment global offsets --
        chain_list = list(model.chains())
        seg_offsets: dict[str, tuple[int, int]] = {}  # seg -> (global_start, n_res)
        total = 0
        for ch in chain_list:
            n_res = len(ch.residues)
            seg = getattr(ch, "key_id", None)
            if seg is None:
                continue
            seg_offsets[str(seg)] = (total, n_res)
            total += n_res

        sasa = [float(default_sasa)] * total
        domain_global: list[list[int]] = []

        if total == 0:
            return (sasa, domain_global)

        # -- helper: build ordered segment info for a component (only those present) --
        def _comp_seginfo(segments: list[str]) -> list[tuple[str, int, int]]:
            # returns list of (seg, global_start, n_res) in component order
            out = []
            for seg in segments:
                if seg in seg_offsets:
                    start, nres = seg_offsets[seg]
                    out.append((seg, start, nres))
            return out

        # -- helper: map one component-local residue index -> global index --
        def _map_component_index(idx: int, seginfo: list[tuple[str, int, int]]) -> int | None:
            if idx < 0:
                return None
            rem = idx
            for _, gstart, nres in seginfo:
                if rem < nres:
                    return gstart + rem
                rem -= nres
            return None  # out of range for this component

        # -- main fill --
        for comp in self.component:
            ctype = comp.ctype
            seginfo = _comp_seginfo(comp.segment)
            if not seginfo:
                continue

            # (1) SASA: indices are component-local -> map across segs in order
            # pairs are 0-based (idx, value) already normalized in ComponentType
            for idx, val in ctype.sasa or []:
                g = _map_component_index(idx, seginfo)
                if g is not None and 0 <= g < total:
                    sasa[g] = float(val)

            # (2) Domains: combine mapped indices from all segments INTO ONE list per domain entry
            # domainres is a list of component-local index lists
            if ctype.domainres:
                combined_for_component: list[list[int]] = [[] for _ in ctype.domainres]
                for k, sel in enumerate(ctype.domainres):
                    for idx in sel:
                        g = _map_component_index(int(idx), seginfo)
                        if g is not None:
                            combined_for_component[k].append(g)
                # de-dup while preserving order; append non-empty lists
                for lst in combined_for_component:
                    seen = set()
                    uniq = [x for x in lst if (x not in seen and not seen.add(x))]
                    if uniq:
                        domain_global.append(uniq)

        # Optional masking by domain union
        if mask_by_domain:
            keep: set[int] = set()
            for lst in domain_global:
                keep.update(lst)
            if keep:
                for i in range(total):
                    if i not in keep:
                        sasa[i] = float(default_sasa)
            else:
                sasa = [float(default_sasa)] * total

        return (sasa, domain_global)


def _is_fileobj(f) -> bool:
    return isinstance(f, (io.TextIOBase, io.BufferedIOBase, io.RawIOBase))


def _ensure_readable(f: FileLike) -> None:
    if _is_fileobj(f):
        if hasattr(f, "readable") and not f.readable():
            raise OSError("File object is not readable")
        return
    p = Path(f)
    if not p.is_file():
        raise FileNotFoundError(p)


#        if len(self.domainres)>0 and len(self.sasa)>0 and self.masksasabydomain:
#            mask=np.ones_like(self.sasa,dtype=bool)
#            mask[np.concatenate(self.domainres)]=False
#            self.sasa[mask]=999.0
