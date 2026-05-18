"""Utility helpers shared across the FluoroFate pipeline."""

import logging
import os
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def running_in_notebook():
    """Return ``True`` when the calling code is executing inside a Jupyter notebook.

    Detection is best-effort: it checks for the presence of an
    ``ipykernel`` module and an ``IPKernelApp`` entry in the IPython
    config. If IPython is not importable, returns ``False``.
    """
    if "ipykernel" in sys.modules:
        return True
    try:
        from IPython import get_ipython
        ipython_shell = get_ipython()
        return ipython_shell is not None and "IPKernelApp" in getattr(ipython_shell, "config", {})
    except ImportError:
        return False


def configure_java_home():
    """Set ``JAVA_HOME`` (and update ``PATH``) to a working JDK if not already set.

    Searches a list of likely locations (``jdk4py.JAVA_HOME`` and the
    current Conda prefix's ``Library`` directory) for a JVM shared
    library and configures the environment so that PyImageJ / scyjava
    can locate it at runtime.

    Returns
    -------
    str or None
        The resolved ``JAVA_HOME`` path, or ``None`` if no JDK was found.
        If ``JAVA_HOME`` is already set in the environment, that value is
        returned unchanged.
    """
    if os.environ.get("JAVA_HOME"):
        return os.environ["JAVA_HOME"]
    candidate_homes = []
    try:
        import jdk4py
        candidate_homes.append(str(jdk4py.JAVA_HOME))
    except ImportError:
        pass
    candidate_homes.append(str(Path(sys.prefix) / "Library"))
    for candidate_home in candidate_homes:
        if (Path(candidate_home) / "bin" / "server" / "jvm.dll").exists():
            os.environ["JAVA_HOME"] = candidate_home
            os.environ["PATH"] = str(Path(candidate_home) / "bin") + os.pathsep + os.environ.get("PATH", "")
            LOGGER.info("Configured JAVA_HOME = %s", candidate_home)
            return candidate_home
    LOGGER.warning("No JDK found; PyImageJ may fail to initialise. Set JAVA_HOME manually.")
    return None
