from abc import ABC, abstractmethod
from typing import List, Dict

import torch
from transformers import AutoModel, AutoTokenizer


class AbstractModelWrapper(ABC):
    """
    Abstract base class for models.
    """

    def __init__(self, pretrained_model_name: str, data_type: torch.dtype = None, trust_remote_code: bool = False,
                 max_tokens: int = 512):
        """
        Initializes the model wrapper by loading the given model's tokenizer and model parameters.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, trust_remote_code=trust_remote_code)
        if data_type:
            self.encoder = AutoModel.from_pretrained(pretrained_model_name, dtype=data_type,
                                                     trust_remote_code=trust_remote_code)
        else:
            self.encoder = AutoModel.from_pretrained(pretrained_model_name, trust_remote_code=trust_remote_code)
        self.model_path = pretrained_model_name
        self.encoder.eval()
        self.encoder.cuda()
        self.max_tokens = max_tokens

    @abstractmethod
    def pool_and_normalize(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.tensor:
        """
        Applies pooling to the given model output and normalizes the embeddings.
        """
        pass

    @abstractmethod
    def encode_queries(self, queries: List[str]) -> torch.Tensor:
        """
        Encode a list of queries into embeddings.
        """
        pass

    @abstractmethod
    def encode_corpus(self, corpus: List[str]) -> torch.Tensor:
        """
        Encode a list of documents into embeddings.
        """
        pass

    @abstractmethod
    def encode_corpus_no_pooling(self, corpus: List[str]) -> List[Dict[str, torch.Tensor]]:
        """
        Encode the list of documents and return the regular model outputs without any pooling.
        """
        pass
