from typing import Any

from ase.calculators.calculator import Calculator
from sklearn import naive_bayes as sknb
from dscribe.descriptors import SOAP


class OnlineLearningCalculator(Calculator):
    def __init__(
        self,
        *,
        backend_cls: type[Calculator],
        backend_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.__backend = backend_cls(**backend_kwargs)
        self.__model = sknb._BaseNB
