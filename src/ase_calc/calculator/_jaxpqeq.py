"""PQEq Calculator Based on Jax-MD."""

import warnings

from ase import Atoms
from ase.calculators.calculator import all_changes
from typing_extensions import override

from ase_calc.calculator._abc import CalculatorABC
from ase_calc.calculator._jaxmd import JAX_MD_AVAILABLE, KCAL_MOL_2_EV
from ase_calc.libraries.libpqeq import nonbonded


class JaxPQEq(CalculatorABC):
    implemented_properties = [
        "energy",
        "free_energy",
        "forces",
        "charges",
    ]

    def __init__(
        self,
        r_cutoff: float = 12.5,
        pqeq_iterations: int = 2,
        capacity_multiplier: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.__rcutoff = float(r_cutoff)
        self.__pqeq_iterations = pqeq_iterations
        self.__capacity_multiplier = capacity_multiplier

    @property
    @override
    def available(self) -> bool:
        return JAX_MD_AVAILABLE

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
        assert isinstance(self.atoms, Atoms)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            qs, e, f = nonbonded(
                self.atoms,
                iforce="energy" in properties,
                pqeq_iterations=self.__pqeq_iterations,
                capacity_multiplier=self.__capacity_multiplier,
                r_cutoff=self.__rcutoff,
            )
        self.results["charges"] = qs
        if e is not None:
            self.results["energy"] = float(e) * KCAL_MOL_2_EV
            self.results["free_energy"] = self.results["energy"]
        if f is not None:
            self.results["forces"] = f * KCAL_MOL_2_EV


def test_jax_pqeq() -> None:
    if not JAX_MD_AVAILABLE:
        print("Skip for jax-md is not installed.")
        return
    else:
        from time import perf_counter
    from ase.cluster import Octahedron

    atoms = Atoms(Octahedron("Cu", 3, 1))  # , cell=[30, 30, 30], pbc=True)
    atoms.calc = JaxPQEq()
    for _ in range(10):
        t = perf_counter()
        atoms.calc.results = {}
        print("Charge: ", atoms.get_charges().sum(), sep="\t")
        print("Cost Time: ", f"{perf_counter() - t}", sep="\t")
    print(atoms.calc.results)  # -36.1322066711041 eV / 900 ms
    for _ in range(10):
        t = perf_counter()
        atoms.calc.results = {}
        print("Energy: ", atoms.get_potential_energy(), sep="\t")
        print("Cost Time: ", f"{perf_counter() - t}", sep="\t")
    print(atoms.calc.results)  # -36.1322066711041 eV / 900 ms


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s", "-v", "--maxfail=5"])
