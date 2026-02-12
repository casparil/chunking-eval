import json
import os
import time
import tracemalloc
from typing import List, Dict, Optional, Tuple

import click
import faiss
import pandas as pd
import torch
import torch.multiprocessing as mp
from dotenv import load_dotenv
from faiss import IndexFlatIP
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

from config import CHUNK_SIZE, RETRIEVE_K, EMBED_PATH, RESULTS_PATH
from utils.chunking import apply_late_chunking, get_chunks
from utils.dataset_loading import load_data
from models import AbstractModelWrapper, GemmaWrapper, Qwen3Wrapper, SnowflakeWrapper
from utils.evaluate import evaluate_results


def _load_model(model_name: str) -> AbstractModelWrapper:
    """
    Loads the model wrapper corresponding to the name or throws an exception if an unsupported model is passed.

    :param model_name: The name of the model to load.
    :return: The model instance.
    """
    if model_name == "qwen":
        return Qwen3Wrapper()
    elif model_name == "snowflake":
        return SnowflakeWrapper()
    elif model_name == "gemma":
        return GemmaWrapper()
    else:
        raise NotImplementedError(f"Cannot load model wrapper for unsupported model {model_name}!")


def _generate_corpus_embeddings(
        num_tokens: int, overlap: int, threshold:int, batch_num: int, corpus_size:int, splitter_name: str,
        corpus_ids: List[str], corpus_batch: List[Dict[str, str]], index: Optional[IndexFlatIP],
        model: AbstractModelWrapper
) -> Tuple[List[str], IndexFlatIP, float]:
    """
    Chunks a given batch of documents, creates embeddings for the chunks and adds them to a FAISS index. If the index is
    not yet instantiated, it is first created.

    :param num_tokens: The maximum number of tokens per chunk.
    :param overlap: The number of overlapping tokens between consecutive chunks.
    :param threshold: The similarity threshold to use for semantic chunking.
    :param batch_num: The current batch number.
    :param corpus_size: The total size of the corpus.
    :param splitter_name: The name of the chunking module to use.
    :param corpus_ids: The ids of the documents.
    :param corpus_batch: The documents to chunk and embed.
    :param index: The FAISS index used to store the embeddings.
    :param model: The embedding model to use.
    :return: The list of chunk ids.
    """
    batch_size = 10
    last_batch_size = len(corpus_batch) % batch_size
    num_gpus = torch.cuda.device_count()
    iterator = range(0, len(corpus_batch) - last_batch_size, batch_size)
    split_tokens = num_tokens

    if splitter_name == "late":
        num_tokens = model.max_tokens
        assert split_tokens < num_tokens

    chunk_start = time.time()
    with tqdm_joblib(desc="Chunking document batch", total=len(iterator)):
        results = Parallel(n_jobs=4, backend="multiprocessing")(
            delayed(get_chunks)(
                num_tokens, overlap, threshold, corpus_ids[start_idx:start_idx+batch_size], splitter_name,
                model.model_path, corpus_batch[start_idx:start_idx+batch_size], num_gpus
            ) for start_idx in iterator
        )

    chunk_ids, doc_chunks = [], []
    for res_dict in results:
        chunk_ids.extend(res_dict["ids"])
        doc_chunks.extend(res_dict["chunks"])

    if last_batch_size != 0:
        result = get_chunks(num_tokens, overlap, threshold, corpus_ids[-last_batch_size:], splitter_name,
                             model.model_path, corpus_batch[-last_batch_size:], num_gpus)
        chunk_ids.extend(result["ids"])
        doc_chunks.extend(result["chunks"])
    chunk_time = time.time() - chunk_start

    if splitter_name != "late":
        embeddings = model.encode_corpus(doc_chunks).detach().cpu()
    else:
        model_outputs = model.encode_corpus_no_pooling(doc_chunks)
        embeddings, chunk_ids = apply_late_chunking(model_outputs, split_tokens, chunk_ids)

    if not index:
        index = IndexFlatIP(embeddings.shape[1])

    index.add(embeddings)
    return chunk_ids, index, chunk_time


def _get_corpus_matches(matches: Dict[str, List[str]]):
    """
    Extracts the information on duplicate chunks stored in the given dictionary into a list and returns it.

    :param matches: Stores information about duplicate chunks.
    :return: The flattened duplicate information.
    """
    corpus_matches = []

    for chunk_id in matches:
        duplicate_ids = matches[chunk_id]
        for duplicate_id in duplicate_ids:
            corpus_matches.append([chunk_id, duplicate_id])

    return corpus_matches


def get_query_embeddings(save_path: str, embed_model: AbstractModelWrapper, queries: List[str]) -> torch.Tensor:
    """
    Loads previously saved query embeddings or generates new ones, if the expected embeddings file could not be found.

    :param save_path: The path where the query embeddings should be stored.
    :param embed_model: The embedding model to use.
    :param queries: The queries to be embedded.
    :return: The query embeddings.
    """
    if save_path and os.path.exists(os.path.join(save_path, "queries.pt")):
        return torch.load(os.path.join(save_path, "queries.pt"))
    else:
        click.echo("Generating query embeddings.")
        query_embeds = embed_model.encode_queries(queries)
        if save_path:
            torch.save(query_embeds, os.path.join(save_path, "queries.pt"))
        return query_embeds


def get_faiss_index(
        corpus_size: int, num_tokens: int, overlap: int, threshold: int, save_path: str, splitter_name: str,
        model: AbstractModelWrapper, corpus: Dict[str, Dict[str, str]]
) -> Tuple[IndexFlatIP, pd.DataFrame, float, int]:
    """
    Loads a previously created FAISS index containing the document embeddings or creates a new index. If a new index is
    created, documents are chunked using the chosen splitter and subsequently embedded before being added to the new
    index. The index is then stored at the given path along with a file mapping chunk ids to index ids.

    :param corpus_size: The size of the corpus used to create embeddings in batches.
    :param num_tokens: The maximum number of tokens per chunk.
    :param overlap: The number of overlapping tokens between consecutive chunks.
    :param threshold: The similarity threshold to use for semantic chunking.
    :param save_path: The path to store the FAISS index.
    :param splitter_name: The name of the chunking module to use.
    :param model: The embedding model to use.
    :param corpus: The documents to be embedded.
    :return: The index along with the id mapping.
    """
    chunk_time, index_size = 0, 0
    index_name, map_name = f"IndexFlatIP_{corpus_size}_{num_tokens}", f"id_map_{corpus_size}_{num_tokens}.csv"

    if save_path and os.path.exists(os.path.join(save_path, index_name)):
        index = faiss.read_index(os.path.join(save_path, index_name))
        df = pd.read_csv(os.path.join(save_path, map_name))
        index_size = os.path.getsize(os.path.join(save_path, index_name))
        click.echo(f"Loaded pre-existing index for corpus size {corpus_size}.")
    else:
        click.echo(f"Could not find any pre-existing index for corpus size {corpus_size}. Generating embeddings and "
                   f"index. This can take while.")
        corpus_ids = sorted(corpus, key=lambda k: len(corpus[k]["text"]), reverse=True)
        corpus = [corpus[cid] for cid in corpus_ids]
        chunk_ids, doc_hashes, chunk_hashes = [], [], []
        index = None

        iterator = range(0, len(corpus), CHUNK_SIZE)
        start = time.time()

        for batch_num, corpus_start_idx in enumerate(iterator):
            click.echo(f"Encoding batch {batch_num + 1}/{len(iterator)}")
            corpus_end_idx = min(corpus_start_idx + CHUNK_SIZE, len(corpus))
            ids, index, batch_time = _generate_corpus_embeddings(
                num_tokens, overlap, threshold, batch_num, corpus_size, splitter_name,
                corpus_ids[corpus_start_idx:corpus_end_idx], corpus[corpus_start_idx:corpus_end_idx], index, model
            )
            chunk_ids.extend(ids)
            chunk_time += batch_time
            torch.cuda.empty_cache()

        click.echo(f"Generated and embedded {len(chunk_ids)} chunks for {len(corpus)} documents.")
        click.echo(f"Time taken: {time.time() - start:.2f} seconds.")
        df = pd.DataFrame({"chunk_ids": chunk_ids, "chunk_num": list(range(len(chunk_ids)))})

        if save_path:
            faiss.write_index(index, os.path.join(save_path, index_name))
            df.to_csv(os.path.join(save_path, map_name), index=False)
            index_size = os.path.getsize(os.path.join(save_path, index_name))

        if not index:
            raise ValueError("Index could not be generated. Exiting.")

    return index, df, chunk_time, index_size


@click.command()
@click.argument("model", type=str, default="qwen")
@click.argument("splitter", type=str, default="token")
@click.argument("tokens", type=int, default=512)
@click.argument("overlap", type=int, default=25)
@click.argument("threshold", type=int, default=95)
def main(model: str, splitter: str, tokens: int, overlap: int, threshold: int):
    """
    Evaluates the performance of different document chunking strategies at increasing corpus size. Performance metrics
    are stored as a JSON file.

    :param model: A string representing the embedding model to use.
    :param splitter: A string representing the splitting strategy to use.
    :param tokens: An integer representing the maximum number of tokens per chunk.
    :param overlap: An integer representing the number of overlapping tokens between consecutive chunks.
    :param threshold: An integer representing the similarity threshold to use for semantic chunking.
    """
    tracemalloc.start()
    load_dotenv()
    mp.set_start_method('spawn')
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    embed_model = _load_model(model)
    corpora, queries, qrels, qrels_relevant_only = load_data()
    query_ids = list(queries.keys())
    queries = [queries[qid] for qid in queries]
    results = {}
    embed_path = None
    save_path = os.path.join(RESULTS_PATH, model, splitter)
    os.makedirs(save_path, exist_ok=True)
    messages, index_sizes = [], []
    chunk_time, retrieval_time, index_time, num_chunks = 0, 0, 0, 0

    if EMBED_PATH:
        embed_path = os.path.join(EMBED_PATH, model, splitter)
        os.makedirs(embed_path, exist_ok=True)

    query_embeddings = get_query_embeddings(os.path.join(EMBED_PATH, model), embed_model, queries).detach().cpu()

    start = time.time()
    for corpus_size in corpora:
        results[corpus_size] = {}
        index_start = time.time()
        index, id_df, corpus_chunk_time, index_size = get_faiss_index(
            corpus_size, tokens, overlap, threshold, embed_path, splitter, embed_model, corpora[corpus_size]
        )
        index_sizes.append(index_size)
        chunk_time += corpus_chunk_time
        index_time += time.time() - index_start
        torch.cuda.empty_cache()
        chunk_ids = id_df["chunk_ids"].values
        num_chunks += len(chunk_ids)
        messages.append(f"Generated and embedded {len(chunk_ids)} chunks for {corpus_size} documents.")
        messages.append(f"Time taken: {time.time() - index_start:.2f} seconds.")
        retrieval_start = time.time()
        distances, neighbors = index.search(query_embeddings, RETRIEVE_K)
        retrieval_time += time.time() - retrieval_start

        for num, (distance, neighbor) in enumerate(zip(distances, neighbors)):
            query_id = query_ids[num]
            results[corpus_size][query_id] = {}
            for score, index_id in zip(distance, neighbor):
                chunk_id = chunk_ids[index_id]
                doc_id = '_'.join(chunk_id.split("_")[:-1])
                if not doc_id in results[corpus_size][query_id]:
                    results[corpus_size][query_id][doc_id] = score.item()

    for message in messages:
        click.echo(message)

    click.echo(f"Finished retrieval. Time taken: {time.time() - start}.")
    corpus_results = None

    for corpus_size in sorted(corpora.keys()):
        click.echo(f"Evaluating corpus of size {corpus_size}.")

        if corpus_results is None:
            corpus_results = results[corpus_size]
        else:
            for qid in results[corpus_size]:
                for cid in results[corpus_size][qid]:
                    corpus_results[qid][cid] = results[corpus_size][qid][cid]

        scores = evaluate_results(corpus_results, qrels, qrels_relevant_only)
        click.echo(f"NDCG@10: {scores['ndcg_at_10']}")
        results_path = os.path.join(save_path, f"{corpus_size}.json")

        with open(results_path, "w") as f:
            json.dump(scores, f, indent=4)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    stats = {
        "retrieval_time": retrieval_time,
        "chunk_time": chunk_time,
        "index_construction_time": index_time,
        "chunk_number": num_chunks,
        "current_memory": current,
        "peak_memory": peak,
        "index_sizes": index_sizes
    }

    if not os.path.isfile(os.path.join(save_path, "stats.json")):
        with open(os.path.join(save_path, "stats.json"), "w") as f:
            json.dump(stats, f, indent=4)


if __name__ == "__main__":
    main()
