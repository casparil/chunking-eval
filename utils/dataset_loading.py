from collections import defaultdict
from typing import Tuple, Dict, List

import click
from datasets import load_dataset, Dataset

CoRE = {
    "doc_core": 10_000,
    "doc_10k": 10_000,
    "doc_100k": 100_000,
    "doc_1M": 1_000_000,
}
KILT = {
    "pass_core": 10_000,
    "pass_10k": 10_000,
    "pass_100k": 100_000,
    "pass_1M": 1_000_000,
    "pass_10M": 10_000_000,
}


def _create_corpus_dict(datasets_corpus: Dict[str, Dataset], corpus_sizes: List[int]) -> defaultdict:
    """
    Transforms the given dictionary of datasets into python dictionary with one entry per document.

    :param datasets_corpus: A dictionary containing the loaded corpora split by corpus size.
    :return: The constructed dictionary.
    """
    corpora = defaultdict(dict)

    for corpus_size, dataset_corpus in zip(corpus_sizes, datasets_corpus.values()):
        for d in dataset_corpus:
            corpora[corpus_size][d["_id"]] = {
                "title": d["title"],
                "text": d["text"],
            }
        click.echo(
            f"Loaded {len(corpora[corpus_size])} documents for corpus of size {corpus_size}"
        )

    return corpora


def _load_core_data(
        qrels_only: bool = False
) -> Tuple[defaultdict | None, Dict[str, str] | None, defaultdict | None, defaultdict]:
    """
    Loads the CoRE document dataset collection from Hugging Face. The function returns two sets of qrels: One that
    contains only relevant documents along with their score for each query and one that contains both relevant and
    distractor documents per query.

    :param qrels_only: Whether to load only the qrels or the entire dataset corpus.
    :return: The loaded document corpora, queries and qrels.
    """
    # Load queries dataset
    dataset_queries = load_dataset("PaDaS-Lab/CoRE", "queries")["test"]

    # Load the qrels dataset
    dataset_qrels = load_dataset("PaDaS-Lab/CoRE", "qrels", split="document")

    # Transform the datasets
    qrels = defaultdict(dict)
    for q in dataset_qrels:
        query_id = q["query-id"]
        corpus_id = q["corpus-id"]
        qrels[query_id][corpus_id] = q["type"]

    # Simplify qrels
    qrels_relevant_only = defaultdict(dict)
    for qid in qrels:
        for docid in qrels[qid]:
            if qrels[qid][docid] == "relevant":
                qrels_relevant_only[qid][docid] = 1

    if qrels_only:
        return None, None, None, qrels_relevant_only

    queries = {q["_id"]: q["text"] for q in dataset_queries if q["_id"] in qrels.keys()}
    click.echo(f"Loaded {len(queries)} queries")

    # Load the corpus datasets
    datasets_corpus = {}
    for split_name in CoRE:
        dataset_corpus = load_dataset("PaDaS-Lab/CoRE", "corpus", split=split_name)
        datasets_corpus[split_name] = dataset_corpus

    # Transform the corpus datasets
    corpora = _create_corpus_dict(datasets_corpus, list(CoRE.values()))

    return corpora, queries, qrels, qrels_relevant_only


def _load_kilt_data(
        qrels_only: bool = False
) -> Tuple[defaultdict | None, Dict[str, str] | None, defaultdict | None, defaultdict]:
    """
    Loads the KILT dataset collection from Hugging Face using NQ qrels.

    :param qrels_only: Whether to load only the qrels or the entire dataset corpus.
    :return: The loaded document corpora, queries and qrels.
    """
    dataset_queries = load_dataset("anonymous202501/KILT", "queries", split="test")
    dataset_qrels = load_dataset("anonymous202501/KILT", "qrels", split="default")

    qrels, qrels_relevant_only = defaultdict(dict), defaultdict(dict)
    for q in dataset_qrels:
        query_id = q["query-id"]
        corpus_id = q["corpus-id"]
        qrels[query_id][corpus_id] = "relevant"
        qrels_relevant_only[query_id][corpus_id] = q["score"]

    if qrels_only:
        return None, None, None, qrels_relevant_only

    queries = {q["_id"]: q["text"] for q in dataset_queries if q["_id"] in qrels.keys()}
    click.echo(f"Loaded {len(queries)} queries")

    datasets_corpus = {}
    for split_name in KILT:
        dataset_corpus = load_dataset("anonymous202501/KILT", "corpus", split=split_name)
        datasets_corpus[split_name] = dataset_corpus

    corpora = _create_corpus_dict(datasets_corpus, list(KILT.values()))
    return corpora, queries, qrels, qrels_relevant_only



def load_data(dataset: str, qrels_only: bool = False) -> Tuple[defaultdict, Dict[str, str], defaultdict, defaultdict]:
    """
    Loads the CoRE document dataset collection from Hugging Face or the scaled KILT dataset.

    :param dataset: The name of the dataset to load.
    :param qrels_only: Whether to load only the qrels or the entire dataset corpus.
    :return: The loaded document corpora, queries and qrels.
    """
    if dataset == "core":
        return _load_core_data(qrels_only)
    elif dataset == "kilt":
        return _load_kilt_data(qrels_only)
    else:
        raise ValueError(f"Cannot load unsupported dataset {dataset}!")
