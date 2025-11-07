from __future__ import annotations

import io
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Optional, Union

import numpy as np
from openmm.unit import (
    Quantity,
    nanometer,
    norm,
)

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

    enmpairs: list[int, int, float] = field(default_factory=list)  # ENM pair list
    enmfile: Optional[FileLike] = None  # read ENM pairs from file
    enmcutoff: float = 0.9  # nm

    def __repr__(self) -> str:
        r = f"<component type {self.name!r} with {self.nunit} units"
        r += f", SASA: {len(self.sasa)}, domains: {len(self.domainres)}"
        r += f", ENM pairs: {len(self.enmpairs)}>"
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

        if self.domainfile is not None and len(self.domainres) == 0:
            _ensure_readable(self.domainfile)
            self.domainres = np.loadtxt(self.domainfile, dtype=int, ndmin=2).tolist()

        if len(self.domainsel) > 0 and len(self.domainres) == 0:
            if sca is not None:
                self.domainres = DomainSelector(self.domainsel).atom_lists(sca)
            else:
                raise ValueError("No structure available for selection.")

        if len(self.domainres) > 0 and sca is not None:
            enmpairs = self._findENMPairs(sca)
            self.enmpairs.extend(enmpairs)

        if self.enmfile is not None:
            _ensure_readable(self.enmfile)
            data = np.loadtxt(self.enmfile, dtype=float)
            data = np.atleast_2d(data)
            pairs: list[tuple[int, int, Quantity]] = []
            for i, j, d in data:
                pairs.append((int(i), int(j), float(d) * nanometer))
            self.enmpairs.extend(pairs)

        self.sasa = self._coerce_sasa_pairs(self.sasa)

        if self.nunit is None:
            if sca is not None:
                self.nunit = sca.nchains()
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
        path: FileLike | Path | str = "component_types",
        *,
        writeout: bool = False,
        dir: Union[str | Path] = ".",
        enmcutoff: Optional[float] = None,
    ) -> dict[str, ComponentType]:
        """
        Read component types from `path` (path or file-like) and return {key: ComponentType}.
        Two formats are supported (see original docstring).
        """
        base = Path(dir)
        comps: dict[str, ComponentType] = {}

        # open handle
        need_close = False
        if _is_fileobj(path):
            fh = path  # already-open file-like
        else:
            p = Path(path)
            if not p.is_file() and not p.is_absolute():
                p = base / p
            if not p.is_file():
                raise FileNotFoundError(f"component types file not found: {p}")
            fh = p.open()
            need_close = True

        if writeout:
            base.mkdir(parents=True, exist_ok=True)

        try:
            for ln, line in enumerate(fh, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    raise ValueError(f"{path}:{ln}: expected 3 columns, got {len(parts)}")

                key, col2, col3 = parts[0], parts[1], parts[2]

                if col2.endswith(".pdb"):  # Format 1
                    tag = Path(col2).stem
                    selection = col3
                    if enmcutoff is not None:
                        ct = ComponentType(
                            name=key,
                            sasapdb=str(base / f"{tag}.pdb"),
                            domainsel=selection,
                            enmcutoff=enmcutoff,
                        )
                    else:
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
                                base / f"{tag}.surface", np.column_stack([idx, val]), fmt="%d %.8f"
                            )
                        else:
                            open(base / f"{tag}.surface", "w").close()

                        domain = np.asarray(ct.domainres, dtype=int)
                        domain = np.atleast_2d(domain)
                        np.savetxt(base / f"{tag}.domainsel", domain, fmt="%d")

                        pairs = []
                        for i, j, w in ct.enmpairs or []:
                            pairs.append((int(i), int(j), w.value_in_unit(nanometer)))

                        path = base / f"{tag}.enmpairs"
                        if not pairs:
                            path.write_text("")
                        else:
                            arr = np.asarray(pairs, dtype=float)
                            np.savetxt(path, arr, fmt=["%d", "%d", "%.6f"], delimiter=" ")

                else:  # Format 2
                    if len(parts) > 3:
                        col4 = parts[3]
                        ct = ComponentType(
                            name=key,
                            sasafile=str(base / f"{col2}"),
                            domainfile=str(base / f"{col3}"),
                            enmfile=str(base / f"{col4}"),
                        )
                    else:
                        ct = ComponentType(
                            name=key,
                            sasafile=str(base / f"{col2}"),
                            domainfile=str(base / f"{col3}"),
                        )
                    comps[key] = ct
        finally:
            if need_close:
                fh.close()

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
    ctype: dict[str, ComponentType] = field(default_factory=dict)  # name -> type
    model: Union[Structure, Model, None] = None

    sasa: list[float] = field(default_factory=list)
    domains: list[list[int]] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"<assembly has {len(self.component)} components and {len(self.ctype)} types>"

    __str__ = __repr__

    def __init__(
        self,
        components: Union[FileLike, None] = None,
        types: Union[FileLike, None] = None,
        mol: Union[Structure, Model, None] = None,
        *,
        dir: Union[str, Path] = ".",
        enmcutoff: Optional[float] = None,
    ) -> None:
        """
        If `components` and `types` are provided, load an Assembly from disk,
        if a model is also provided, store it.
        Otherwise, create an empty Assembly.
        """
        self.component = []
        self.ctype = {}  # dict[str, ComponentType]
        self.model = None
        self.sasa = []
        self.domains = []
        self.enmpairs = []

        if (components is None) ^ (types is None):
            raise ValueError("Provide both components and types file names, or neither.")
        if components is not None and types is not None:
            self.read_components(components, types, dir=dir, enmcutoff=enmcutoff)
        if mol is not None:
            if isinstance(mol, Structure):
                if mol.models:
                    self.model = mol.models[0].select_CA()
            else:
                self.model = mol.select_CA()

    def components(self) -> Iterator[Component]:
        return iter(self.component)

    def types(self) -> Iterator[ComponentType]:
        return iter(self.ctype.values())

    def add_type(self, ctype: ComponentType) -> None:
        # insert or keep existing by name
        if ctype.name not in self.ctype:
            self.ctype[ctype.name] = ctype

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

    def read_components(
        self,
        comp_file: FileLike,
        type_file: FileLike,
        *,
        dir: Union[str, Path] = ".",
        enmcutoff: Optional[float] = None,
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
            types = ComponentType.read_list(type_source, dir=base, enmcutoff=enmcutoff)
        else:
            type_path = Path(type_file)
            if not type_path.is_absolute():
                type_path = base / type_path
            _ensure_readable(type_path)
            types = ComponentType.read_list(type_path, dir=base, enmcutoff=enmcutoff)

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

    def _global_sasa(
        self,
        *,
        mol: Union[Structure, Model, None] = None,
        mask_by_domain: bool = False,
        default_sasa: float = 999.0,
    ) -> None:
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
            keep: set[int] = {i for lst in self.domains for i in lst}
            if keep:
                for i in range(total):
                    if i not in keep:
                        sasa[i] = float(default_sasa)
            else:
                sasa = [float(default_sasa)] * total

        self.sasa = sasa


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
