"""The ASE-compatible calculator for this package."""

import jax
import pytest
from ase.calculators.calculator import Calculator
from ase.calculators.emt import EMT

from ase_calc.calculator._abc import CalculatorABC
from ase_calc.calculator._gfn import XTB
from ase_calc.calculator._gulp import GULP, ReaxFF
from ase_calc.calculator._kim import KIM

jax.config.update("jax_logging_level", "ERROR")
jax.numpy.asarray([])


def get_calculator(method: str) -> Calculator:
    args, kwargs = [], {}
    if method.count(",") != 0:
        split = method.split(",")
        for arg in split[1:]:
            match arg.count("="):
                case 0:
                    args.append(arg)
                case 1:
                    k, v = arg.split("=")
                    kwargs[k] = v
                case _:
                    raise KeyError(f"Invalid SPE method: {method}")
        method = split[0]

    method = method.lower()
    try:
        if method == "emt":
            result = EMT(*args, **kwargs)
        elif method.startswith("kim"):
            result = KIM(*args, **kwargs)
        elif method.startswith("xtb"):
            result = XTB(*args, **kwargs)
        elif method.startswith("gulp"):
            result = GULP(*args, **kwargs)
        elif method.startswith("reaxff"):
            result = ReaxFF(*args, **kwargs)
        elif method.startswith("gfn"):
            match method[3]:
                case "0":
                    result = XTB(method="GFN0-xTB", *args, **kwargs)
                case "1":
                    result = XTB(method="GFN1-xTB", *args, **kwargs)
                case "2":
                    result = XTB(method="GFN2-xTB", *args, **kwargs)
                case _:
                    result = XTB(method="GFNFF", *args, **kwargs)
        else:
            raise KeyError(f"Invalid SPE method: {method}")
    except Exception as e:
        raise KeyError(f"Invalid SPE method: {method}; {e}")

    if isinstance(result, CalculatorABC):
        assert result.available, f"The SPE method: {method} not available!"
    return result


@pytest.mark.parametrize(
    "spe",
    [
        "emt",
        "gfnff",
        "xtb,gfn1xtb",
        "xtb,method=gfn2xtb",
        "reaxff,CuCHO.ff",
    ],
)
def test_get_calculator(spe: str) -> None:
    print(get_calculator(spe))


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-v", "--maxfail=5"])
