"""Calculator Based on LAMMPS."""

import warnings
from pathlib import Path
from typing import Literal

import jax
import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import all_changes
from ase.geometry.cell import cell_to_cellpar
from jax import numpy as jnp
from typing_extensions import override

from ase_calc.calculator._abc import CalculatorABC
from ase_calc.calculator._lammps import PATH_POTENTIALS
from ase_calc.libraries import LIB_REAXFF as LIBREAXFF_PATH

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        import jax_md  # type: ignore
        from jax_md import space  # type: ignore
        from jax_md.reaxff.reaxff_forcefield import ForceField  # type: ignore
        from jax_md.reaxff.reaxff_helper import read_force_field  # type: ignore
        from jax_md.reaxff.reaxff_interactions import (  # type: ignore
            reaxff_inter_list,
        )
except ImportError:
    jax_md = None
JAX_MD_AVAILABLE = bool(jax_md is not None)
KCAL_MOL_2_EV = (units.kcal / units.mol) / units.eV
jax.config.update("jax_traceback_filtering", "off")
jax.config.update("jax_logging_level", "ERROR")
jax.config.update("jax_enable_x64", True)


class JaxReaxFF(CalculatorABC):
    implemented_properties = [
        "energy",
        "free_energy",
        "forces",
        # "stress",
        # "energies",
    ]

    def __init__(
        self,
        model: str,
        cutoff2: float = 0.001,
        tors_2013: bool = False,
        backprop_solve: bool = False,
        use_single_float: bool = False,
        long_inters_capacity_multiplier: float = 1.2,
        short_inters_capacity_multiplier: float = 1.2,
        solver_model: str | Literal["EEM", "ACKS"] = "EEM",
        solver_iter_count: int = 500,
        solver_tol: float = 1e-6,
        **kwargs,
    ) -> None:
        """The ReaxFF based on Jax.

        Args:
            model (str): The reaxff file path.
            cutoff2 (float, optional): _description_. Defaults to 0.001.
            tors_2013 (bool, optional): Control variable to decide whether to use
                more stable version of the torsion interactions. Defaults to False.
            backprop_solve (bool, optional): Control variable to decide whether to
                do a solve to calculate the gradients of the charges wrt positions.
                By definition, the gradients should be 0 but if the solver tolerance
                is high, the gradients might be non-ignorable. Defaults to False.
            use_single_float (bool, optional): _description_. Defaults to False.
            long_inters_capacity_multiplier (float, optional): capacity
                multiplier for all long range interactions. Defaults to 1.2.
            short_inters_capacity_multiplier (float, optional): capacity
                multiplier for all short range interactions. Defaults to 1.2.
            solver_model (str | Literal['EEM', 'ACKS'], optional): Control
                variable for the solver model ("EEM" or "ACKS"). Defaults to "EEM".
            solver_iter_count (int, optional): Maximum number of solver iterations. Defaults to 500.
            solver_tol (float, optional): Tolarence for the charge solver. Defaults to 1e-6.
        """  # noqa: E501
        self.__solver_tol = solver_tol
        self.__solver_model = solver_model
        self.__solver_iter_count = solver_iter_count
        self.__short_capacity_multiplier = short_inters_capacity_multiplier
        self.__long_capacity_multiplier = long_inters_capacity_multiplier
        self.__backprop_solve = backprop_solve
        self.__tors_2013 = tors_2013
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
        self.__type = jnp.float32 if use_single_float else jnp.float64
        force_field = read_force_field(  # type: ignore
            self.__pot,
            cutoff2=float(cutoff2),
            dtype=self.__type,
        )
        force_field = ForceField.fill_off_diag(force_field)  # type: ignore
        self.__reaxff = ForceField.fill_symm(force_field)  # type: ignore
        super().__init__(**kwargs)

    @property
    @override
    def available(self) -> bool:
        return JAX_MD_AVAILABLE

    def _initialize(self, atoms: Atoms) -> None:
        Z = jnp.asarray(atoms.numbers, dtype=jnp.int64)
        if atoms.cell.volume > 1e-3 or atoms.pbc.any():
            cell = atoms.get_cell().complete().array
        else:
            min = atoms.positions.min(axis=0)
            max = atoms.positions.max(axis=0)
            cell = np.diag((max - min) + 15)
        cellpar = cell_to_cellpar(cell)
        if not np.allclose(cellpar[3:], 90):
            raise KeyError("Only orth ... supported.")
            BOX = jnp.asarray(cell, dtype=self.__type)
            displacement, _ = space.periodic_general(BOX)  # type: ignore
        else:
            BOX = jnp.asarray(cellpar[:3], dtype=self.__type)
            displacement, _ = space.periodic(BOX)  # type: ignore

        S = jnp.array(
            [self.__reaxff.name_to_index[t] for t in atoms.get_chemical_symbols()]
        )
        self.__reaxff_inter_fn, self.__energy_fn = reaxff_inter_list(  # type: ignore
            displacement,
            BOX,  # cell length, 3 float
            S,  # Species in force filed, 1xN array
            Z,  # Atomic numbers, 1xN array
            self.__reaxff,
            tors_2013=self.__tors_2013,
            solver_model=self.__solver_model,
            backprop_solve=self.__backprop_solve,
            total_charge=atoms.get_initial_charges().sum(),
            short_inters_capacity_multiplier=self.__short_capacity_multiplier,  # type: ignore
            long_inters_capacity_multiplier=self.__long_capacity_multiplier,
            max_solver_iter=self.__solver_iter_count,
            tol=self.__solver_tol,
        )
        R = jnp.asarray(atoms.get_positions(), dtype=self.__type)
        self.__nbrs = self.__reaxff_inter_fn.allocate(R)
        self.__energy_fn = jax.jit(jax.value_and_grad(self.__energy_fn))

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
            if "numbers" in system_changes:
                self._initialize(self.atoms)

            R = jnp.asarray(self.atoms.get_positions(), dtype=self.__type)
            self.__nbrs = self.__reaxff_inter_fn.update(R, self.__nbrs)
            if self.__nbrs.did_buffer_overflow:
                self.__nbrs = self.__reaxff_inter_fn.allocate(R)
            e, grad = self.__energy_fn(R, self.__nbrs)
        self.results["energy"] = float(e) * KCAL_MOL_2_EV
        self.results["free_energy"] = self.results["energy"]
        self.results["forces"] = -np.asarray(grad * KCAL_MOL_2_EV)


def test_jax_reaxff() -> None:
    if not JAX_MD_AVAILABLE:
        print("Skip for jax-md is not installed.")
        return
    else:
        from time import perf_counter
    from ase.cluster import Octahedron

    atoms = Atoms(Octahedron("Cu", 3, 1))  # , cell=[30, 30, 30], pbc=True)
    atoms.calc = JaxReaxFF(model="CuCHO.ff")
    # atoms.calc._initialize(atoms)
    for _ in range(10):
        t = perf_counter()
        atoms.calc.results = {}
        print("Energy: ", atoms.get_potential_energy(), sep="\t")
        print("Cost Time: ", f"{perf_counter() - t}", sep="\t")
    print(atoms.calc.results)  # -36.1322066711041 eV / 900 ms


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s", "-v", "--maxfail=5"])
