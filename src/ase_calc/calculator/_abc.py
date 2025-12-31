from abc import abstractmethod

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from typing_extensions import override


class CalculatorABC(Calculator):
    @override
    @abstractmethod
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
        self.atoms: Atoms = self.atoms

    @property
    def available(self) -> bool:
        return True
