from abc import ABC, abstractmethod


class ParserCommon(ABC):
    """Abstract class for parsers of different codes."""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def parse_header(self):
        pass

    @abstractmethod
    def parse_kpoint(self, ik):
        pass
