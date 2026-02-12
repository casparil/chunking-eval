from abc import ABC, abstractmethod
from typing import List, Dict


class AbstractSplitter(ABC):
    """
    Abstract base class for chunking.
    """

    @abstractmethod
    def split_text(self, documents: List[Dict[str, str]]) -> List[List[str]]:
        """
        Split a list of texts into chunks.
        """
        pass
