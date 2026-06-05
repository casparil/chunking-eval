# Evaluating Chunking Strategies

## Install Dependencies

The project's dependencies can be installed in multiple ways.

### Using uv (recommended)

The project was developed using the [uv](https://docs.astral.sh/uv/) package manager for fast dependency management and can be installed using

```
uv sync
```

### Using pip (alternative)

You can also create a venv yourself and use `pip` to install dependencies:

```console
python3 -m venv venv
source venv/bin/activate
pip install .
```

## Run Evaluation Code

The evaluation code runs supported chunking strategies on the CoRE and KILT datasets using one of three supported
embedding models.
Currently, the supported models are [embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)
(requires to accept license agreement), [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) and
[snowflake-arctic-embed-l-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0).
Supported chunking techniques include token- and sentence-based chunking, as well as semantic, context-enriched,
summary-based and contextual chunking.
To start an evaluation run, execute the command
```
uv run main.py qwen token           # Evaluates token-based chunking using Qwen
uv run main.py gemma semantic kilt  # Evaluates semantic chunking on KILT using Gemma
uv run main.py snowflake enriched   # Evaluates context-enriched chunking using Snowflake
```

The code downloads the respective dataset from Hugging Face and uses the chosen model to generate embeddings in batches.
Each batch is added to a FAISS index, which is queried for retrieval.

After running the evaluation code, you will find the results in the `results` folder.
The results are stored in a JSON file in a folder structure organized by model name and chunking method.
Default folders for storing results and embeddings can be changed in `config.py`.

To run LLM-based chunking methods, create a `.env` file, copy the content of the `.env.example` file and replace the
empty strings with the actual URL, model name and API key. Any model compatible with the OpenAI API can be used.

## Extend Code

### Add New Chunking Strategy

Adding new chunking methods requires extending the [AbstractSplitter](splitters/AbstractSplitter.py) class, which
has a method `split_text()` where you need to add your custom implementation for splitting a list of documents into
chunks.

````python
from typing import List, Dict

from splitters.AbstractSplitter import AbstractSplitter


class CharacterSplitter(AbstractSplitter):

    def __init__(self, chars_per_chunk: int = 512, char_overlap: int = 25):
        assert 0 <= char_overlap < chars_per_chunk
        self.char_overlap = char_overlap
        self.chars_per_chunk = chars_per_chunk

    def split_text(self, documents: List[Dict[str, str]]) -> List[List[str]]:
        chunks = []
        for doc in documents:
            text = "{} {}".format(doc.get("title", ""), doc["text"]).strip()
            start_idx = 0
            doc_chunks = []

            while start_idx < len(text):
                start_idx = 0 if start_idx == 0 else start_idx - self.char_overlap
                end_idx = start_idx + self.chars_per_chunk
                end_idx = min(end_idx, len(text))
                doc_chunks.append(text[start_idx:end_idx])
                start_idx = end_idx

            chunks.append(doc_chunks)
        return chunks
````

To include the new class in the evaluation, it needs to be registered in the `load_splitter()` method in
[chunking.py](utils/chunking.py).

### Add New Model

New embedding models can be added by implementing the [AbstractModelWrapper](models/AbstractModelWrapper.py) class,
which requires implementing encoding functions for queries and documents.
Any model available via `transformers` can be added easily.
For reference, consider the example below:

```python
from typing import List, Tuple, Dict

import torch
from transformers import BatchEncoding

from models.AbstractModelWrapper import AbstractModelWrapper
from models.utils import last_token_pool


class CustomModelWrapper(AbstractModelWrapper):

    def __init__(self, pretrained_model_name: str = "myModel/myModel-v-2"):
        super().__init__(pretrained_model_name, torch.float16, True, 8192)
        self.instruction = 'This is a web search query:'

    def _get_detailed_instruct(self, query: str) -> str:
        return f'Instruct: {self.instruction}\nQuery:{query}'

    @torch.no_grad()
    def _encode_input(
            self, sentences: List[str], apply_pooling: bool = True
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=8192)
        embeds, model_outputs = [], []
        sub_inputs = BatchEncoding()
        inputs.to('cuda')
        outputs = self.encoder(**inputs)
        
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
```

The wrapper then needs to be registered in the `_load_model()` method of the [main script](main.py).
