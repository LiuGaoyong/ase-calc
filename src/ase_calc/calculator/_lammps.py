"""Calculator Based on LAMMPS."""

from pathlib import Path

import pytest
from ase import Atoms
from ase.calculators.calculator import all_changes
from ase.calculators.lammpslib import LAMMPSlib as _LAMMPSlib
from ase.calculators.lammpsrun import LAMMPS as _LAMMPS
from typing_extensions import override

from ase_calc.calculator._abc import CalculatorABC
from ase_calc.libraries import LIB_REAXFF as LIBREAXFF_PATH

try:
    import lammps
except ImportError:
    lammps = None
LAMMPS_AVAILABLE = bool(lammps is not None)
if LAMMPS_AVAILABLE:
    PATH_LAMMPS_ROOT = Path(lammps.__file__).parent  # type: ignore
    PATH_POTENTIALS = PATH_LAMMPS_ROOT / "share" / "lammps" / "potentials"
    if not PATH_POTENTIALS.exists() or not PATH_POTENTIALS.is_dir():
        PATH_POTENTIALS = None
else:
    PATH_POTENTIALS = None


class LAMMPS(CalculatorABC):
    implemented_properties = [
        "energy",
        "free_energy",
        "forces",
        "stress",
        "energies",
    ]
    ignored_changes = {"initial_charges", "initial_magmoms"}

    def __init__(
        self,
        model: str,
        pair_style: str,
        pair_coeff: str,
        use_cli: bool = True,
        **kwargs,
    ) -> None:
        self.__pot = Path(model)
        if not self.__pot.exists() or not self.__pot.is_file():
            if PATH_POTENTIALS is not None:
                self.__pot = PATH_POTENTIALS / model
            if not self.__pot.exists():
                self.__pot = LIBREAXFF_PATH / model
        assert self.__pot.exists() and self.__pot.is_file(), (
            f"The file of '{self.__pot}' is not found."
            f" The search folder are {PATH_POTENTIALS}."
        )
        self.__use_cli = bool(use_cli)
        self.__pair_style = str(pair_style)
        self.__pair_coeff = str(pair_coeff)
        assert self.__pot.name in self.__pair_coeff, (
            f"Invalid pair_coeff='{self.__pair_coeff}'."
            f" The model name '{self.__pot.name}' is not in it."
        )
        self.__pair_coeff = self.__pair_coeff.replace(
            self.__pot.name, self.__pot.absolute().__fspath__()
        )
        print(self.__pair_coeff)
        super().__init__(**kwargs)

    @property
    @override
    def available(self) -> bool:
        return LAMMPS_AVAILABLE

    @override
    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] = ["energy"],
        system_changes=all_changes,
    ) -> None:
        super().calculate(
            atoms=atoms,
            properties=properties,
            system_changes=system_changes,
        )
        _atoms: Atoms = self.atoms.copy()
        if self.__use_cli:
            _atoms.calc = _LAMMPS(
                pair_style=self.__pair_style,
                pair_coeff=[self.__pair_coeff],
                command="lmp ",
            )
        else:
            _atoms.calc = _LAMMPSlib(
                lmpcmds=[
                    f"pair_style {self.__pair_style}",
                    f"pair_coeff {self.__pair_coeff}",
                ],
            )
        _atoms.get_potential_energy()
        self.results = _atoms.calc.results  # type: ignore


class ReaxFF(LAMMPS):
    def __init__(self, model: str, **kwargs) -> None:
        raise NotImplementedError
        pot = Path(model)
        super().__init__(
            model=model,
            pair_style="reaxff NULL",
            pair_coeff=f"* * {pot.name}, ELEM_LST",
            use_cli=False,
            **kwargs,
        )
        pot: Path = getattr(self, "_LAMMPS__pot")
        elem_lst: list[str] = []
        with pot.open("r") as f:
            while line := f.readline():
                if "Nr" in line:
                    break
            while True:
                for _ in range(3):
                    f.readline()
                line = f.readline()
                if "Nr" in line:
                    break
                else:
                    elem_lst.append(line.strip().split()[0])
        pair_coeff = getattr(self, "_LAMMPS__pair_coeff")
        new_pair_coeff = pair_coeff.replace("ELEM_LST", "Cu")  # " ".join(elem_lst))
        setattr(self, "_LAMMPS__pair_coeff", new_pair_coeff)


@pytest.mark.parametrize("use_cli", [True, False])
def test_lammps(use_cli: bool) -> None:
    import platform

    if not LAMMPS_AVAILABLE:
        print("Skip for lammps is not installed.")
        return
    elif platform.platform().startswith("Windows") and use_cli:
        print("Skip for Windows and use_cli=True.")
        return

    from ase import Atom, Atoms
    from ase.build import bulk

    Ni: Atoms = bulk("Ni", cubic=True)
    H = Atom("H", position=Ni.cell.diagonal() / 2)  # type: ignore
    atoms = Ni + H
    atoms.calc = LAMMPS(
        pair_style="eam/alloy",
        model="NiAlH_jea.eam.alloy",
        pair_coeff="* * NiAlH_jea.eam.alloy Ni H",
        use_cli=use_cli,
    )
    print("Energy ", atoms.get_potential_energy())
    print(atoms.calc.results)


# def test_lammps_reaxff() -> None:
#     if not LAMMPS_AVAILABLE:
#         print("Skip for lammps is not installed.")
#         return
#     else:
#         from ase.build import bulk
#     atoms: Atoms = bulk("Zn")
#     atoms.calc = ReaxFF(model="ffield.reax.ZnOH")
#     print("Energy ", atoms.get_potential_energy())
#     print(atoms.calc.results)


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-v", "--maxfail=5"])
