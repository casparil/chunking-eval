from typing import List, Tuple, Dict

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BatchEncoding
from transformers.utils import is_flash_attn_2_available

from models.AbstractModelWrapper import AbstractModelWrapper
from models.utils import last_token_pool


class Qwen3Wrapper(AbstractModelWrapper):

    def __init__(self, pretrained_model_name: str = "Qwen/Qwen3-Embedding-0.6B", chunk_size: int = 32):
        super().__init__(pretrained_model_name, torch.float16, True, 8192)
        self.instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        if is_flash_attn_2_available():
            self.encoder = AutoModel.from_pretrained(pretrained_model_name, trust_remote_code=True,
                                                     attn_implementation="flash_attention_2", dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, trust_remote_code=True,
                                                       padding_side='left')
        self.chunk_size = chunk_size

    def _get_detailed_instruct(self, query: str) -> str:
        return f'Instruct: {self.instruction}\nQuery:{query}'

    @torch.no_grad()
    def _encode_input(
            self, sentences: List[str], apply_pooling: bool = True
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=8192)
        iterator = range(0, len(sentences), self.chunk_size)
        embeds, model_outputs = [], []

        for start_idx in tqdm(iterator, desc="Generating embeddings", total=len(iterator)):
            sub_inputs = BatchEncoding()
            sub_inputs['input_ids'] = inputs['input_ids'][start_idx:start_idx + self.chunk_size]
            sub_inputs['attention_mask'] = inputs['attention_mask'][start_idx:start_idx + self.chunk_size]
            sub_inputs.to('cuda')
            outputs = self.encoder(**sub_inputs)

            if apply_pooling:
                embeds.append(self.pool_and_normalize(outputs.last_hidden_state, sub_inputs['attention_mask']))
            else:
                model_outputs.append({
                    "last_hidden_state": outputs.last_hidden_state.detach().cpu(),
                    "attention_mask": sub_inputs["attention_mask"].detach().cpu()
                })

        if embeds:
            embeds = torch.vstack(embeds).detach().cpu()

        return embeds, model_outputs

    def pool_and_normalize(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.tensor:
        embeds = last_token_pool(last_hidden_states, attention_mask)
        return torch.nn.functional.normalize(embeds, p=2, dim=-1)

    def encode_queries(self, queries: List[str]) -> torch.Tensor:
        sentences = [self._get_detailed_instruct(query) for query in queries]
        embeds, _ = self._encode_input(sentences)
        return embeds

    def encode_corpus(self, corpus: List[str]) -> torch.Tensor:
        embeds, _ = self._encode_input(corpus)
        return embeds

    def encode_corpus_no_pooling(self, corpus: List[str]) -> List[Dict[str, torch.Tensor]]:
        _, outputs = self._encode_input(corpus, False)
        return outputs
