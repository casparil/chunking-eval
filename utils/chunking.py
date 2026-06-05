import logging
import os
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from splitters import AbstractSplitter, TokenSplitter, SentenceTokenSplitter, SemanticSplitter, SummarySplitter, \
    ContextualSplitter, ContextEnrichedSplitter

# Use one splitter per worker
_splitter = None
# Get a list of currently visible GPUs
VISIBLE_GPUS = os.environ.get("CUDA_VISIBLE_DEVICES", "")


def _load_splitter(num_tokens: int, overlap: int, threshold:int, splitter_name: str, model_path: str,
                   tokenizer: AutoTokenizer, use_summary: bool = False) -> AbstractSplitter:
    """
    Loads the text splitter corresponding to the given name or throws an exception if an unsupported name is passed.

    :param num_tokens: The maximum number of tokens per chunk.
    :param overlap: The number of overlapping tokens between consecutive chunks.
    :param threshold: The similarity threshold to use for semantic chunking.
    :param splitter_name: The name of the splitter to load.
    :param model_path: The model path to use to instantiate the embedding model for semantic chunking.
    :param tokenizer: The tokenizer to use for splitters relying on tokens.
    :return: The splitter instance.
    """
    api_key = os.getenv("API_KEY")
    model = os.getenv("MODEL")
    url = os.getenv("URL")

    if splitter_name == "token" or splitter_name == "late":
        return TokenSplitter(tokenizer, num_tokens, overlap)
    elif splitter_name == "sentence":
        return SentenceTokenSplitter(tokenizer, num_tokens, overlap)
    elif splitter_name == "semantic":
        return SemanticSplitter(tokenizer, model_path, threshold)
    elif splitter_name == "summary":
        return SummarySplitter(api_key, url, model, tokenizer, 100)
    elif splitter_name == "contextual":
        return ContextualSplitter(api_key, url, model, tokenizer, num_tokens, overlap)
    elif splitter_name == "enriched":
        summary_splitter = None
        if use_summary:
            summary_splitter = SummarySplitter(api_key, url, model, tokenizer)
        return ContextEnrichedSplitter(summary_splitter, tokenizer, num_tokens, overlap)
    else:
        raise NotImplementedError(f"Cannot load splitter for unsupported splitter {splitter_name}!")


def _get_splitter(num_tokens: int, overlap: int, threshold:int, splitter_name: str, model_path: str,
                  num_gpus: int) -> AbstractSplitter:
    """
    Initializes the worker's global splitter instance and attempts to assign them to different GPUs.

    :param num_tokens: The maximum number of tokens per chunk.
    :param overlap: The number of overlapping tokens between consecutive chunks.
    :param threshold: The similarity threshold to use for semantic chunking.
    :param splitter_name: The name of the splitter to load.
    :param model_path: The model path to use to instantiate the embedding model for semantic chunking.
    :param num_gpus: The number of available GPUs.
    :return: The splitter instance.
    """
    global _splitter
    if _splitter is None:
        pid = os.getpid()
        if VISIBLE_GPUS:
            gpu_id = VISIBLE_GPUS.split(",")[pid % num_gpus]
        else:
            gpu_id = pid % num_gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        _splitter = _load_splitter(num_tokens, overlap, threshold, splitter_name, model_path, tokenizer)
        # Suppress spamming of too long token sequence length
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    return _splitter


def get_chunks(
        num_tokens: int, overlap: int, threshold:int, corpus_ids: List[str], splitter_name: str, model_path: str,
        documents: List[Dict[str, str]], summaries: List[str] | None, num_gpus: int
) -> Dict[str, List[str]]:
    """
    Applies the given text splitter to the passed documents and chunks them, returning the created chunks along with
    their ids. The chunk ids are a combination of the original document id and the chunk number.

    :param num_tokens: The maximum number of tokens per chunk.
    :param overlap: The number of overlapping tokens between consecutive chunks.
    :param threshold: The similarity threshold to use for semantic chunking.
    :param corpus_ids: The ids of the documents.
    :param splitter_name: The name of the splitter to load.
    :param model_path: The model path to use to instantiate the embedding model for semantic chunking.
    :param documents: The documents to chunk.
    :param num_gpus: The number of available GPUs.
    :return: A list of chunk ids and document chunks.
    """
    ids, all_chunks = [], []
    splitter = _get_splitter(num_tokens, overlap, threshold, splitter_name, model_path, num_gpus)

    if isinstance(splitter, ContextEnrichedSplitter) and splitter.summary_splitter is not None and summaries is not None:
        chunks = splitter.split_text_with_summary(documents, summaries)
    else:
        chunks = splitter.split_text(documents)

    for idx, doc_chunks in enumerate(chunks):
        ids.extend([f"{corpus_ids[idx]}_{num + 1}" for num in range(len(doc_chunks))])
        all_chunks.extend(doc_chunks)
    return {"ids": ids, "chunks": all_chunks}


@torch.no_grad()
def apply_late_chunking(
        model_outputs: List[Dict[str, torch.Tensor]], num_tokens: int, chunk_ids: List[str]
) -> Tuple[torch.Tensor, List[str]]:
    """
    Applies late chunking to the given model outputs by splitting the token embeddings into chunks of the equal length
    and pooling them. Chunk ids are updated accordingly.

    :param model_outputs: A list of model outputs.
    :param num_tokens: The maximum number of tokens per chunk.
    :param chunk_ids: The original ids of the chunks.
    :return: The generated embeddings and the new chunk ids.
    """
    outputs, new_chunk_ids = [], []
    total_chunk_num = 0

    for model_output in tqdm(model_outputs, desc="Applying late chunking", total=len(model_outputs)):
        token_embeddings = model_output["last_hidden_state"]
        attention_masks = model_output["attention_mask"]
        current_doc_id = None
        chunk_num = 1

        for embeddings, attention_mask in zip(token_embeddings.cuda(), attention_masks.cuda()):
            chunk_id = chunk_ids[total_chunk_num]
            total_chunk_num += 1
            doc_id = '_'.join(chunk_id.split("_")[:-1])
            iterator = range(0, len(embeddings), num_tokens)
            pooled_embeddings = []

            if current_doc_id != doc_id:
                current_doc_id = doc_id
                chunk_num = 1

            for start_pos in iterator:
                end_pos = min(len(embeddings), start_pos + num_tokens)
                valids = attention_mask[start_pos:end_pos].sum()
                if end_pos - start_pos > 1 and valids > 0:
                    embeds = (embeddings[start_pos:end_pos] * attention_mask[start_pos:end_pos].unsqueeze(-1)).sum(
                        dim=0) / valids
                    pooled_embeddings.append(embeds)

            for _ in pooled_embeddings:
                new_chunk_ids.append(f"{doc_id}_{chunk_num}")
                chunk_num += 1

            pooled_embeddings = torch.vstack(pooled_embeddings)
            pooled_embeddings = torch.nn.functional.normalize(pooled_embeddings, p=2, dim=-1)
            outputs.append(pooled_embeddings.detach().cpu())

    return torch.vstack(outputs).float(), new_chunk_ids
