import os
import sys
from typing import Any, Union, List


def create_run_dir(dir_name: str, root_dir: str = "results") -> str:
    """Creates a directory for the current execution"""
    if os.path.exists(root_dir):
        dir_content = os.listdir(root_dir)
        dir_content = [os.path.join(root_dir, d) for d in dir_content]
        dir_content = [d for d in dir_content if os.path.isdir(d)]
        next_idx = find_next_idx(dir_content)
    else:
        next_idx = "000"
    run_dir_path = os.path.join(root_dir, f"{next_idx}-{dir_name}")
    os.makedirs(run_dir_path)
    return run_dir_path


def find_next_idx(dir_content: List[str]) -> str:
    """Finds the ID for the next experiment.

    Args:
        dir_content (List[str]): Files inside directory.

    Returns:
        str: Formatted experiment id.
    """
    if len(dir_content) > 0:
        idxs = [d.split("/")[-1].split("-")[0][-3:] for d in dir_content]
        idxs = [int(i) for i in idxs if i.isnumeric()]
        next_idx = max(idxs) + 1 if len(idxs) else 0
    else:
        next_idx = 0
    return f"{next_idx:03d}"


class Logger(object):
    """Redirect stderr to stdout, optionally print stdout to a file, and
    optionally force flushing on both stdout and the file.

    Taken from: https://github.com/NVlabs/stylegan3"""

    def __init__(
        self, file_name: str = None, file_mode: str = "w", should_flush: bool = True
    ):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: Union[str, bytes]) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if isinstance(text, bytes):
            text = text.decode()
        # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
        if len(text) == 0:
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()
            self.file = None
