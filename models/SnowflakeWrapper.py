from functools import partial
from typing import List, Tuple, Dict

import torch
from datasets import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, BatchEncoding

from models.AbstractModelWrapper import AbstractModelWrapper
from models.utils import move_to_cuda, transform_func, pool


class SnowflakeWrapper(AbstractModelWrapper):

    def __init__(self, pretrained_model_name: str = "Snowflake/snowflake-arctic-embed-l-v2.0", batch_size: int = 64,
        pool_type: str = "avg", doc_as_query: bool = False):
        super().__init__(pretrained_model_name, torch.float16, trust_remote_code=True, max_tokens=8192)
        self.pool_type = pool_type
        self.doc_as_query = doc_as_query
        self.batch_size = batch_size
        self.encoder.cuda()
        self.encoder.eval()
        self.prefix = "query: "

    def pool_and_normalize(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.tensor:
        embeds = pool(last_hidden_states, attention_mask, self.pool_type)
        return torch.nn.functional.normalize(embeds, p=2, dim=-1)

    def encode_queries(self, queries: List[str]) -> torch.Tensor:
        input_texts = [
            "{}{}".format(self.prefix, q)
            for q in queries
        ]
        embeds, _ = self._do_encode(input_texts)
        return embeds

    def encode_corpus(self, corpus: List[str]) -> torch.Tensor:
        if self.doc_as_query:
            return self.encode_queries([d for d in corpus])

        embeds, _ = self._do_encode(corpus)
        return embeds

    def encode_corpus_no_pooling(self, corpus: List[str]) -> List[Dict[str, torch.Tensor]]:
        _, outputs = self._do_encode(corpus, False)
        return outputs

    @torch.no_grad()
    def _do_encode(
            self, input_texts: List[str], apply_pooling: bool = True
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        dataset: Dataset = Dataset.from_dict({"contents": input_texts})
        dataset.set_transform(partial(transform_func, self.tokenizer))

        data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            collate_fn=data_collator,
            pin_memory=True,
        )

        encoded_embeds, model_outputs = [], []
        for batch_dict in tqdm(data_loader, desc="Generating embeddings", total=len(data_loader)):
            batch_dict = move_to_cuda(batch_dict)

            with torch.cuda.amp.autocast():
                outputs: BatchEncoding = self.encoder(**batch_dict)

                if apply_pooling:
                    encoded_embeds.append(
                        self.pool_and_normalize(outputs.last_hidden_state, batch_dict["attention_mask"])
                    )

                model_outputs.append({
                    "last_hidden_state": outputs.last_hidden_state.detach().cpu(),
                    "attention_mask": batch_dict["attention_mask"].detach().cpu()
                })

        if encoded_embeds:
            encoded_embeds = torch.vstack(encoded_embeds).detach().cpu()

        return encoded_embeds, model_outputs
