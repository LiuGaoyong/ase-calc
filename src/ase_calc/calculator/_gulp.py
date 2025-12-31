from pathlib import Path

from ase import Atoms
from ase.calculators.calculator import all_changes
from typing_extensions import override

from ase_calc.calculator._abc import CalculatorABC
from ase_calc.libraries import LIB_REAXFF as LIBREAXFF_PATH
from ase_calc.libraries.libgulp import GulpExecutor


class GULP(CalculatorABC):
    implemented_properties = [
        "energy",
        "free_energy",
        "forces",
        # "stress",
    ]

    def __init__(
        self,
        method: Path | str = "gfnff",
        gulp_path: Path | str | None = None,
        n_threads: int = 1,
        **kwargs,
    ) -> None:
        self.__gulp_executor = GulpExecutor(gulp_path)
        self.__n_threads = int(n_threads)
        self.__method = str(method)
        super().__init__(**kwargs)

    @property
    @override
    def available(self) -> bool:
        return self.__gulp_executor.available

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
        e, f, stress = self.__gulp_executor.run_spe(
            method=self.__method,  # type: ignore
            n_threads=self.__n_threads,
            atoms=self.atoms,
        )
        self.results["free_energy"] = self.results["energy"] = float(e)
        self.results["forces"] = f
        if stress is not None:
            self.results["stress"] = stress


class ReaxFF(GULP):
    def __init__(
        self,
        library: Path | str | None = None,
        gulp_path: Path | str | None = None,
        n_threads: int = 1,
        **kwargs,
    ) -> None:
        if library is not None:
            path = Path(library)
            for path in [path, LIBREAXFF_PATH / path.name]:
                if path.exists() and path.is_file():
                    library = path.__fspath__()
                    break
        else:
            library = "reaxff_general"
        super().__init__(
            method=library,
            n_threads=n_threads,
            gulp_path=gulp_path,
            **kwargs,
        )


def test_gulp_reaxff() -> None:
    from ase.cluster import Octahedron

    atoms = Atoms(Octahedron("Cu", 3, 1), cell=[30, 30, 30], pbc=True)
    atoms.calc = ReaxFF(library="CuCHO.ff")
    if not atoms.calc.available:
        print("Skip for GULP is not installed.")
        return
    else:
        from time import perf_counter
    for _ in range(10):
        t = perf_counter()
        atoms.calc.results = {}
        print("Energy: ", atoms.get_potential_energy(), sep="\t")
        print("Cost Time: ", f"{perf_counter() - t}", sep="\t")
    print(atoms.calc.results)  #    -36.13248008 eV / 16 ms


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s", "-v", "--maxfail=5"])
