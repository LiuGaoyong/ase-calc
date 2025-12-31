"""The CLI Executor."""

# ruff: noqa: D101, D102, D107
import subprocess
from abc import ABC, abstractmethod
from os import _Environ, environ
from pathlib import Path

if subprocess._mswindows:  # type: ignore
    DEFAULT_EXE = "cmd"
else:
    DEFAULT_EXE = "bash"


def run_script(
    content: str,
    exe: str = DEFAULT_EXE,
    outputfiles: list[str | Path] = [],
    workdir: str | Path = Path("."),
    env: _Environ = environ,
    timeout=None,
) -> tuple[str, str, str, bool, list[bool]]:
    """Run script by CLI based on `subprocess.Popen`.

        https://docs.python.org/3/library/subprocess.html

    Args:
        content (str): The content of script which to be run.
        exe: (str): The executor which to be run.
            If platform is Windows, it defaults to CMD.
            If platform is Mac or Linux, it defaults to Bash.
        outputfiles (list[str | Path], optional): The output files that
            will be check exist of not after script run. Defaults to [].
        workdir (str | Path, optional): The workdir. Defaults to Path(".").
        env (os._Environ, optional): The envrionment which to be run.
        timeout (float, optional): ...

    Returns:
        tuple[str, bool, list[bool]]:
            The first item is input content before script run.
            The second item is output content after script run.
            The third item is error content after script run.
            The 4th item is whether script run successfully.
            The 5th item is whether exist or not for each output file.
    """
    if not isinstance(workdir, Path):
        workdir = Path(workdir)
    assert isinstance(workdir, Path)
    if workdir.exists():
        assert workdir.is_dir()
    else:
        workdir.mkdir(parents=True)
    kwargs = dict(shell=True, cwd=workdir, env=env)
    kwargs["stdin"] = subprocess.PIPE  # type: ignore
    kwargs["stdout"] = subprocess.PIPE  # type: ignore
    kwargs["stderr"] = subprocess.PIPE  # type: ignore
    with subprocess.Popen(exe, **kwargs) as p:  # type: ignore
        content_splitlines = str(content).splitlines()
        for line in content_splitlines:
            bline = bytes(f"{line}\n", encoding="utf-8")
            p.stdin.write(bline)  # type: ignore
        try:
            out, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            p.kill()
            if subprocess._mswindows:  # type: ignore
                # Windows accumulates the output in a single blocking
                # read() call run on child threads, with the timeout
                # being done in a join() on those threads.  communicate()
                # _after_ kill() is required to collect that and add it
                # to the exception.
                exc.stdout, exc.stderr = p.communicate()
            else:
                # POSIX _communicate already populated the output so
                # far into the TimeoutExpired exception.
                p.wait()
            raise
        except:  # Including KeyboardInterrupt, communicate handled that.
            p.kill()
            # We don't call process.wait() as .__exit__ does that for us.
            raise
        is_success = bool(p.poll() == 0)
    filesexist = [workdir.joinpath(f).exists() for f in outputfiles]

    # decode for out & err
    result = dict(out=out, err=err)
    for k, v in result.items():
        assert isinstance(v, str | bytes), f"TYPE={type(v)}"
        if isinstance(v, bytes):
            has_decode = False
            encoding_lst = ["UTF-8", "GB2312", "GBK", "ISO-8859-1", "UTF-16"]
            for encoding in encoding_lst:
                try:
                    result[k] = v.decode(encoding=encoding)
                    has_decode = True
                    break
                except Exception:
                    has_decode = False
            if not has_decode:
                raise
    out, err = result["out"], result["err"]
    if is_success and exe == "cmd":
        out, nPS = out.strip().splitlines(), []
        for i, line in enumerate(out):
            if ">" in line:
                try:
                    p_startswith = Path(line.split(">")[0])
                    if p_startswith.resolve() == workdir.resolve():
                        nPS.append(i)
                except Exception:
                    pass
        if len(nPS) == 1:
            nPS.append(len(out))
        assert len(nPS) == 2, f"nPS: {nPS}\nout: {out}\ncontent: {content}"
        out = "\n".join(out[nPS[0] + 1 : nPS[1]])

    return (content, out, err, is_success, filesexist)


class CommandExecutorABC(ABC):
    __EXEC_PATH_LST = [
        "/bin",
        "/usr/bin",
        "/usr/local/bin",
        "/opt/homebrew/bin",
    ]

    def __init__(self, exe_path: Path | str | None = None) -> None:
        exe: str = self._exe_name()
        path_list: list[Path] = []
        if exe_path is not None:
            if not isinstance(exe_path, Path):
                exe_path = Path(exe_path)
            path_list.append(exe_path.joinpath(exe))
            path_list = [exe_path] + path_list
        for path in self.__EXEC_PATH_LST:
            path_list.append(Path(path).joinpath(exe))
        which_path, is_success = self._run_where(exe)
        if is_success:
            path_list.insert(0, Path(which_path.strip()))
        self.__available = False
        for path in path_list:
            self.exe = Path(path).resolve()
            if not self.exe.exists():
                continue
            if not self.exe.is_file():
                continue
            if self._exe_is_available():
                self.__available = True
                break

    @staticmethod
    def _run_where(command) -> tuple[str, bool]:
        if subprocess._mswindows:  # type: ignore
            which = f"where.exe {command}"
        else:
            which = f"which {command}"
        _, p, _, success, _ = run_script(which)
        if success:
            p = Path(p.strip())
            success = p.exists() and p.is_file()
        return str(p), success

    @abstractmethod
    def _exe_name(self) -> str: ...

    @abstractmethod
    def _exe_is_available(self) -> bool: ...

    @property
    def available(self) -> bool:
        return self.__available

    def run_content(
        self,
        content: str,
        n_threads: int = 1,
        envdct: dict[str, str] = {},
        workdir: str | Path = Path("."),
        outputfiles: list[str | Path] = [],
    ) -> tuple[str, bool, list[bool]]:
        """Run commond without arguments.

        Usage: exe < content
        """
        return self._run_content(
            content=str(content),
            n_threads=int(n_threads),
            outputfiles=outputfiles,
            exe=str(self.exe),
            workdir=workdir,
            envdct=envdct,
        )

    def run_commond(
        self,
        *args,
        n_threads: int = 1,
        envdct: dict[str, str] = {},
        workdir: str | Path = Path("."),
        outputfiles: list[str | Path] = [],
    ) -> tuple[str, bool, list[bool]]:
        """Run commond with arguments.

        Usage: exe [*args]
        """
        arguments = " ".join([str(arg) for arg in args])
        if subprocess._mswindows:  # type: ignore
            content = f'"{self.exe}" {arguments}'
        else:
            content = f"{self.exe} {arguments}"
        return self._run_content(
            content=content,
            n_threads=int(n_threads),
            outputfiles=outputfiles,
            workdir=workdir,
            envdct=envdct,
        )

    @staticmethod
    def _run_content(
        content: str,
        n_threads: int = 1,
        envdct: dict[str, str] = {},
        workdir: str | Path = Path("."),
        outputfiles: list[str | Path] = [],
        exe: str = DEFAULT_EXE,
    ) -> tuple[str, bool, list[bool]]:
        for k, v in envdct.items():
            environ[str(k)] = str(v)
        for k in (
            "MKL_NUM_THREADS",
            "OMP_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        ):
            environ[str(k)] = f"{n_threads:d}"
        inp, result, err, is_success, outputfiles_exist = run_script(
            content=content,
            outputfiles=outputfiles,
            workdir=workdir,
            env=environ,
            exe=exe,
        )
        if not is_success:
            result = "\n".join(
                [
                    "\n".join(["STDIN:", inp, ""]),
                    "\n".join(["STDOUT:", result, ""]),
                    "\n".join(["STDERR:", err, ""]),
                ]
            )
        return result, is_success, outputfiles_exist
