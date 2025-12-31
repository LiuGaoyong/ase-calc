import contextlib
import os
from collections import defaultdict
from functools import partial
from importlib import import_module
from importlib.util import find_spec
from io import StringIO
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
from ase import Atoms
from ase.calculators import calculator as ase_calc
from typing_extensions import override

from ase_calc.calculator._abc import CalculatorABC
from ase_calc.libraries.libxtb import LIB_XTB_PARAM, XtbExecutor

os.environ["XTBPATH"] = LIB_XTB_PARAM.__fspath__()
CALC_DCT: dict[str, dict[str, type[ase_calc.Calculator]]] = defaultdict(dict)


###########################################################################
# TBLite: Only availible on MacOS & Linux
try:
    from tblite.ase import TBLite as tblite_cls  # type: ignore

    param = tblite_cls.default_parameters
    param["verbosity"] = 0

    class TBLite(tblite_cls):
        """The ASE frontend for tblite.

        It supports the following methods:
            GFN1-xTB
            GFN2-xTB
            IPEA1-xTB
        """

        default_parameters = param

        def calculate(
            self,
            atoms: Atoms | None = None,
            properties: list[str] = None,  # type: ignore
            system_changes=ase_calc.all_changes,
        ) -> None:
            # Original code output some infomation to stdout(main) and stderr,
            # which is not suitable for logging. Redirect them to StringIO.
            with contextlib.redirect_stdout(StringIO()):
                with contextlib.redirect_stderr(StringIO()):
                    return super().calculate(
                        atoms=atoms,
                        properties=properties,
                        system_changes=system_changes,
                    )

    CALC_DCT["tblite"]["tblite"] = TBLite
    CALC_DCT["tblite"]["gfn1"] = partial(TBLite, method="GFN1-xTB")  # type: ignore
    CALC_DCT["tblite"]["gfn2"] = partial(TBLite, method="GFN2-xTB")  # type: ignore
    CALC_DCT["tblite"]["ipea"] = partial(TBLite, method="IPEA1-xTB")  # type: ignore
except ImportError:
    pass
###########################################################################
# xtb: Availible on Windows, MacOS & Linux by Anaconda
try:
    from xtb.ase.calculator import XTB as xtb_cls  # type: ignore

    @contextlib.contextmanager
    def capture_cpp_stdout():
        """Context manager to capture stdout from C++ code.

        See details at: https://www.pythontutorials.net/blog/in-python-how-to-capture-the-stdout-from-a-c-shared-library-to-a-variable/#1-understanding-the-problem-why-standard-redirection-fails
        """
        original_stdout_fd = os.dup(1)  # Duplicate FD 1 (stdout)
        try:
            read_fd, write_fd = os.pipe()
            os.dup2(write_fd, 1)
            yield
            os.dup2(original_stdout_fd, 1)
            os.close(write_fd)  # Close write end to signal EOF
            with os.fdopen(read_fd, "r") as pipe_reader:
                captured_output = pipe_reader.read()
        finally:
            os.close(original_stdout_fd)
        return captured_output

    class PythonXTB(xtb_cls):
        def calculate(
            self,
            atoms: Atoms | None = None,
            properties: list[str] = None,  # type: ignore
            system_changes=ase_calc.all_changes,
        ) -> None:
            # Original code output some infomation to stdout(main) and stderr,
            # which is not suitable for logging. Redirect them to StringIO.
            with contextlib.redirect_stdout(StringIO()):
                with contextlib.redirect_stderr(StringIO()):
                    with capture_cpp_stdout():
                        super().calculate(
                            atoms=atoms,
                            properties=properties,
                            system_changes=system_changes,
                        )
            Path(".").joinpath("gfnff_topo").unlink(missing_ok=True)

    CALC_DCT["xtb"]["xtb"] = PythonXTB
    CALC_DCT["xtb"]["gfnff"] = partial(PythonXTB, method="GFNFF")  # type: ignore
    CALC_DCT["xtb"]["gfn0"] = partial(PythonXTB, method="GFN0-xTB")  # type: ignore
    CALC_DCT["xtb"]["gfn1"] = partial(PythonXTB, method="GFN1-xTB")  # type: ignore
    CALC_DCT["xtb"]["gfn2"] = partial(PythonXTB, method="GFN2-xTB")  # type: ignore
    CALC_DCT["xtb"]["ipea"] = partial(PythonXTB, method="IPEA-xTB")  # type: ignore
except ImportError:
    pass
###########################################################################
# pygfnxtb: Availible on Windows, MacOS & Linux by Anaconda
try:
    from pygfnxtb.ase import XTB as PygfnXTB  # type: ignore

    CALC_DCT["pygfnxtb"]["xtb"] = PygfnXTB
    CALC_DCT["pygfnxtb"]["gfnff"] = partial(PygfnXTB, method="GFNFF")  # type: ignore
    CALC_DCT["pygfnxtb"]["gfn0"] = partial(PygfnXTB, method="GFN0-xTB")  # type: ignore
    CALC_DCT["pygfnxtb"]["gfn1"] = partial(PygfnXTB, method="GFN1-xTB")  # type: ignore
    CALC_DCT["pygfnxtb"]["gfn2"] = partial(PygfnXTB, method="GFN2-xTB")  # type: ignore
except ImportError:
    pass
###########################################################################
# GFNFF & GFN0
lst: list[tuple[str, str]] = [("pygfnff", "GFNFF"), ("pygfn0", "GFN0")]
for module_name, class_name in lst:
    if find_spec(module_name) is not None:
        module = import_module(module_name)
        cls: type[ase_calc.Calculator] = getattr(module, class_name)
        CALC_DCT[module_name][class_name.lower()] = cls
###########################################################################


class XTB(CalculatorABC):
    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(
        self,
        restart=None,
        ignore_bad_restart_file=ase_calc.BaseCalculator._deprecated,
        label=None,
        atoms: Atoms | None = None,
        directory: str = mkdtemp(),  # type: ignore
        *,
        method: str = "GFN1-xTB",
        xtb_path: Path | str | None = None,
        n_threads: int = 1,
    ) -> None:
        super().__init__(
            atoms=atoms,
            label=label,
            restart=restart,
            ignore_bad_restart_file=ignore_bad_restart_file,
            directory=Path(directory).absolute().__fspath__(),
        )
        try:
            assert method.upper()[:3] == "GFN", f"Invalid: {method}."
            if method.upper() != "GFNFF":
                assert method[3] in "012", f"Invalid: {method}."
                self.__method = method.upper()[:4]
            else:
                self.__method = method.upper()
            assert self.__method in ["GFNFF", "GFN0", "GFN1", "GFN2"], (
                f"Invalid: {method.upper()}."
            )
        except Exception as e:
            raise ase_calc.InputError(e)
        self.__xtb_executor = XtbExecutor(xtb_path)
        self.__n_threads = int(n_threads)

    @property
    @override
    def available(self) -> bool:
        return self.__xtb_executor.available or any(
            self.__method.lower() in dct.keys()  #
            for dct in CALC_DCT.values()
        )

    def __run_spe(self, calc: ase_calc.Calculator) -> None:
        assert isinstance(calc, ase_calc.Calculator)
        calc.calculate(self.atoms)
        self.results = calc.results

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] = ["energy"],
        system_changes: list[str] = ase_calc.all_changes,
    ) -> None:
        super().calculate(
            atoms=atoms,
            properties=properties,
            system_changes=system_changes,
        )
        if self.__method.lower() in CALC_DCT["tblite"]:
            self.__run_spe(CALC_DCT["tblite"][self.__method.lower()]())
        elif self.__method.lower() in CALC_DCT["xtb"]:
            self.__run_spe(CALC_DCT["xtb"][self.__method.lower()]())
        elif self.__method.lower() in CALC_DCT["pygfnxtb"]:
            self.__run_spe(CALC_DCT["pygfnxtb"][self.__method.lower()]())
        elif self.__method.lower() in CALC_DCT["pygfnff"]:
            self.__run_spe(CALC_DCT["pygfnff"][self.__method.lower()]())
        elif self.__method.lower() in CALC_DCT["pygfn0"]:
            self.__run_spe(CALC_DCT["pygfn0"][self.__method.lower()]())
        elif self.available:
            e, f = self.__xtb_executor.run_spe(
                method=self.__method.lower(),  # type: ignore
                n_threads=self.__n_threads,
                atoms=self.atoms,
            )
            self.results["free_energy"] = self.results["energy"] = float(e)
            self.results["forces"] = f
        else:
            raise RuntimeError()


def test_calc_dict() -> None:
    from ase.build import molecule

    atoms = molecule("H2O")
    print()
    for k0, v in CALC_DCT.items():
        for k1, cls in v.items():
            atoms.calc = cls()
            try:
                e = atoms.get_potential_energy()
            except Exception:
                e = np.nan
            print(k0, k1, e, sep="\t")


def test_xtb() -> None:
    from ase.build import molecule

    atoms = molecule("H2O")
    atoms.calc = XTB(method="GFN1-xTB")
    print(atoms.get_potential_energy())


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
