"""The XTB Binary Executable Files."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from ase import Atoms
from ase.units import Angstrom, Bohr, Hartree, eV
from numpy import loadtxt, ndarray
from typing_extensions import override

from .._executor import CommandExecutorABC

LIB_XTB_PARAM = Path(__file__).parent


class XtbExecutor(CommandExecutorABC):
    @override
    def _exe_name(self) -> str:
        return "xtb"

    @override
    def _exe_is_available(self) -> bool:
        version, is_success, _ = self.run_commond("--version")
        return bool(is_success and "S. Grimme" in version)

    def run_spe(
        self,
        atoms: Atoms,
        n_threads: int = 1,
        method: Literal["gfnff", "gfn0", "gfn1", "gfn2"] = "gfnff",
    ) -> tuple[float, ndarray]:
        if method == "gfnff":
            k = " --gfnff "
        elif method == "gfn0":
            k = " --gfn 0 "
        elif method == "gfn1":
            k = " --gfn 1 "
        elif method == "gfn2":
            k = " --gfn 2 "
        else:
            raise ValueError(f"Not supported level={method}.")

        with TemporaryDirectory(prefix="xtb_", suffix=method) as dir:
            if atoms.cell.rank == 0:
                fname, format = "atoms.xyz", "xyz"
            else:
                fname, format = "POSCAR", "vasp"
                # if "gfnff" in k:
                #     k = " --mcgfnff --norestart "
            # fname = str(Path(dir).joinpath(fname))
            atoms.write(str(Path(dir).joinpath(fname)), format=format)
            out, is_success, outputfiles_exist = self.run_commond(
                f" {fname} {k} --grad --norestart ",
                outputfiles=["gradient"],
                envdct={"XTBPATH": LIB_XTB_PARAM.__fspath__()},
                n_threads=n_threads,
                workdir=dir,
            )
            assert is_success and all(outputfiles_exist), out
            # A. Parse total energy
            energy = None
            for line in out.splitlines():
                if "TOTAL ENERGY" in line:
                    for item in line.strip().split():
                        try:
                            energy = float(item)
                            break
                        except ValueError:
                            ...
                if energy is not None:
                    break
            assert energy is not None
            energy *= Hartree / eV
            # B. Parse forces from `gradient` file
            with Path(dir).joinpath("gradient").open() as f:
                data = f.readlines()
                f = data[2 + len(atoms) : 2 + 2 * len(atoms)]
                forces = loadtxt(f) * (Hartree / Bohr) / (eV / Angstrom)
        return energy, forces

    @classmethod
    def run_xtb(
        cls,
        atoms: Atoms,
        n_threads: int = 1,
        method: Literal["gfnff", "gfn0", "gfn1", "gfn2"] = "gfnff",
        xtb_path: Path | str | None = None,
    ) -> tuple[float, ndarray]:
        return cls(xtb_path).run_spe(atoms, n_threads, method)
