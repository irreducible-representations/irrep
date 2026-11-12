try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .logging import configure_logging
configure_logging()

from .parsers.parse_files_bandstructure import parse_files as parse_bandstructure  # noqa: F401 E402
