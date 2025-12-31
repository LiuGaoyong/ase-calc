"""Calculator by OpenKim Based on `kimpy`."""

from functools import cached_property
from os import getcwd
from pathlib import Path
from re import search
from tempfile import TemporaryDirectory

from ase import Atoms
from ase.calculators.calculator import all_changes
from ase.calculators.kim import KIM as _AseKIM
from ase.calculators.kim.kimpy_wrappers import PortableModel as _KimPM
from ase.calculators.kim.kimpy_wrappers import Wrappers as _Wrappers
from ase.calculators.kim.kimpy_wrappers import kimpy as _kimpy
from typing_extensions import override

from ase_calc.calculator._abc import CalculatorABC

__all__ = ["KIM", "KIMPY_AVAILABLE"]


KIMPY_AVAILABLE = True
try:
    import kimpy  # type: ignore
except ImportError:
    KIMPY_AVAILABLE = False
    kimpy = _kimpy


class _KimpyWrappers(_Wrappers):
    @cached_property
    def all_portable_models(self) -> dict[str, int]:
        PM = self.collection_item_type_portableModel
        collections = self.collections_create()
        result: dict[str, int] = {}
        for collection in [
            kimpy.collection.system,
            kimpy.collection.user,
            kimpy.collection.environmentVariable,
            kimpy.collection.currentWorkingDirectory,
        ]:
            f1 = collections.cache_list_of_item_names_by_collection_and_type
            for i in range(f1(collection, PM)):
                name = collections.get_item_name_by_collection_and_type(i)
                matched = search(r"[^0-9][1|2][0-9]{3}[^0-9]", name)
                if matched is not None:
                    result[name] = int(matched.group()[1:5])
        return result


KIMPY_WRAPPERS = _KimpyWrappers()


class KIM(CalculatorABC):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]
    ignored_changes = {"initial_charges", "initial_magmoms"}

    def __init__(self, model: str = "", **kwargs):
        self.__model = str(model)
        super().__init__(**kwargs)

    @property
    @override
    def available(self) -> bool:
        if not KIMPY_AVAILABLE:
            return False
        else:
            all_models = KIMPY_WRAPPERS.all_portable_models
            return self.model in all_models

    @property
    def model(self) -> str:
        if self.__model == "":
            return str(self.__model)
        else:
            return self.model_auto

    @property
    def model_auto(self) -> str:
        atoms: Atoms = self.atoms
        sp, _year, result = set(atoms.symbols.species()), 0, ""
        all_models = KIMPY_WRAPPERS.all_portable_models
        if len(all_models) == 0:
            return ""
        for model, year in all_models.items():
            if year > _year:
                pm = _KimPM(model, debug=False)
                if sp <= set(pm.get_model_supported_species_and_codes()[0]):
                    _year, result = year, model
        return result

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
        with TemporaryDirectory(prefix="kim_"):
            _atoms: Atoms = self.atoms.copy()
            _atoms.calc = _AseKIM(self.model)
            _atoms.get_potential_energy()
            self.results = _atoms.calc.results  # type: ignore
        self.delete_kim_log_file()

    @staticmethod
    def delete_kim_log_file() -> None:
        f = Path(getcwd()).joinpath("kim.log")
        f.unlink(missing_ok=True)


def test_kimauto() -> None:
    if KIMPY_AVAILABLE:
        from ase.build import molecule
    else:
        print("Skip for kimpy is not installed.")
        return
    atoms = molecule("C6H6")
    atoms.calc = KIM()
    print(atoms.calc.model)
    print(atoms.get_potential_energy())


if __name__ == "__main__":
    import pytest
    from ase.build import bulk

    print("dfasd")
    atoms: Atoms = bulk("Zn") * (3, 3, 3)
    atoms.calc = _AseKIM(
        "Sim_LAMMPS_ReaxFF_RaymandVanDuinBaudin_2008_ZnOH__SM_449472104549_001"
    )
    print(atoms.get_potential_energy())
    print(atoms.get_forces())
    pytest.main([__file__, "-s", "-v"])
