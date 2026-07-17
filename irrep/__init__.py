try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
from .parsers.parse_files_bandstructure import parse_files as parse_bandstructure
