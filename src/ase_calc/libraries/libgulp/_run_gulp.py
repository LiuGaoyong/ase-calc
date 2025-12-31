"""The GULP Binary Executable Files."""

from functools import reduce
from pathlib import Path
from re import match
from sys import platform
from tempfile import TemporaryDirectory

import numpy as np
from ase import Atoms
from ase.calculators.calculator import CalculationFailed
from ase.data import chemical_symbols
from ase.units import Ang, eV
from typing_extensions import override

from .._executor import CommandExecutorABC

LIB_GULP = Path(__file__).parent
SEP = LIB_GULP._flavour.sep  # type: ignore
AN2ELEM = np.char.rjust(chemical_symbols, 2)


class GulpExecutor(CommandExecutorABC):
    @override
    def _exe_name(self) -> str:
        return "gulp"

    @override
    def _exe_is_available(self) -> bool:
        if platform.startswith("win"):
            return False
        else:
            version, is_success, _ = self.run_content("--version")
            return bool(is_success and "Julian Gale" in version)

    def __format_atoms(self, atoms: Atoms) -> str:
        fmt = "%15.8f"
        result: list[str] = []
        if atoms.cell.rank != 0:
            result.append("vectors")
            cell = atoms.cell.complete()
            x = np.char.mod(fmt, cell[:, 0])  # type: ignore
            y = np.char.mod(fmt, cell[:, 1])  # type: ignore
            z = np.char.mod(fmt, cell[:, 2])  # type: ignore
            result.extend(reduce(np.char.add, [x, y, z]))
        result.append("cartesian")
        elem = AN2ELEM[atoms.numbers]
        x = np.char.mod(fmt, atoms.positions[:, 0])
        y = np.char.mod(fmt, atoms.positions[:, 1])
        z = np.char.mod(fmt, atoms.positions[:, 2])
        result.extend(reduce(np.char.add, [elem, " core ", x, y, z]))
        return "\n".join(result) + "\n"

    def run_spe(
        self,
        atoms: Atoms,
        n_threads: int = 1,
        method: Path | str = "gfnff",
    ) -> tuple[float, np.ndarray, np.ndarray | None]:
        content = "conp gradients\n"
        if method == "gfnff":
            content = f"{method} {content}"
            library = ""
        else:
            LIBRARY_FIND = False
            path = Path(method) if isinstance(method, str) else method
            library = path.name
            assert isinstance(library, str) and isinstance(path, Path)
            for p in (
                LIB_GULP.joinpath(path.name),
                LIB_GULP.joinpath(f"{path.name}.lib"),
            ):
                if p.exists() and p.is_file():
                    library = f"library nodump {library}\n"
                    LIBRARY_FIND = True
                    break
            if not LIBRARY_FIND and path.exists() and path.is_file():
                library = f"reaxff_library {path}\n"
                LIBRARY_FIND = True
            if not LIBRARY_FIND:
                library = "library nodump reaxff_general\n"
        inp = f"{content}{self.__format_atoms(atoms)}{library}"
        with TemporaryDirectory(prefix="gulp_") as dir:
            result, success, _ = self.run_content(
                content=inp,
                n_threads=n_threads,
                envdct={"GULP_LIB": LIB_GULP.__fspath__()},
                workdir=dir,
            )
        if success:
            return self.__parse_results(content=result)
        else:
            raise CalculationFailed(f"GULP run error:\n{result}\nInput:\n{inp}")

    def __parse_results(
        self,
        content: str,
    ) -> tuple[float, np.ndarray, np.ndarray | None]:
        lines, results = str(content).splitlines(), {}
        for i, line in enumerate(lines):
            m = match(r"\s*Total lattice energy\s*=\s*(\S+)\s*eV", line)
            if m:
                energy = float(m.group(1))
                results["energy"] = energy
            elif line.find("Final Cartesian derivatives") != -1:
                s = i + 5
                forces = []
                while True:
                    s = s + 1
                    if lines[s].find("------------") != -1:
                        break
                    if lines[s].find(" s ") != -1:
                        continue
                    g = lines[s].split()[3:6]
                    G = [-float(x) * eV / Ang for x in g]
                    forces.append(G)
                results["forces"] = np.array(forces)
            elif line.find("Final internal derivatives") != -1:
                s = i + 5
                forces = []
                while True:
                    s = s + 1
                    if lines[s].find("------------") != -1:
                        break
                    g = lines[s].split()[3:6]

                    # Uncomment the section below to separate the numbers when
                    # there is no space between them, in the case of long
                    # numbers. This prevents the code to break if numbers are
                    # too big.

                    """for t in range(3-len(g)):
                        g.append(' ')
                    for j in range(2):
                        min_index=[i+1 for i,e in enumerate(g[j][1:])
                                   if e == '-']
                        if j==0 and len(min_index) != 0:
                            if len(min_index)==1:
                                g[2]=g[1]
                                g[1]=g[0][min_index[0]:]
                                g[0]=g[0][:min_index[0]]
                            else:
                                g[2]=g[0][min_index[1]:]
                                g[1]=g[0][min_index[0]:min_index[1]]
                                g[0]=g[0][:min_index[0]]
                                break
                        if j==1 and len(min_index) != 0:
                            g[2]=g[1][min_index[0]:]
                            g[1]=g[1][:min_index[0]]"""

                    G = [-float(x) * eV / Ang for x in g]
                    forces.append(G)
                forces = np.array(forces)
                results["forces"] = forces
            elif line.find("Final stress tensor components") != -1:
                res = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                for j in range(3):
                    var = lines[i + j + 3].split()[1]
                    res[j] = float(var)
                    var = lines[i + j + 3].split()[3]
                    res[j + 3] = float(var)
                results["stress"] = np.array(res)
        assert set(["energy", "forces"]) <= set(results.keys()), content
        return results["energy"], results["forces"], results.get("stress", None)

    @classmethod
    def run_gulp(
        cls,
        atoms: Atoms,
        n_threads: int = 1,
        method: Path | str = "gfnff",
        gulp_path: Path | str | None = None,
    ) -> tuple[float, np.ndarray, np.ndarray | None]:
        return cls(gulp_path).run_spe(atoms, n_threads, method)
