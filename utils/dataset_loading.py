from collections import defaultdict
from typing import Tuple, Dict

import click
from datasets import load_dataset


CoRE = {
    "doc_core": 10_000,
    "doc_10k": 10_000,
    "doc_100k": 100_000,
    "doc_1M": 1_000_000,
}


def load_data() -> Tuple[defaultdict, Dict[str, str], defaultdict, defaultdict]:
    """
    Loads the CoRE document dataset collection from Hugging Face. The function returns two sets of qrels: One that
    contains only relevant documents along with their score for each query and one that contains both relevant and
    distractor documents per query.

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

    queries = {q["_id"]: q["text"] for q in dataset_queries if q["_id"] in qrels.keys()}
    click.echo(f"Loaded {len(queries)} queries")

    # Load the corpus datasets
    datasets_corpus = {}
    for split_name in CoRE:
        dataset_corpus = load_dataset("PaDaS-Lab/CoRE", "corpus", split=split_name)
        datasets_corpus[split_name] = dataset_corpus

    # Transform the corpus datasets
    corpora = defaultdict(dict)
    for corpus_size, dataset_corpus in datasets_corpus.items():
        for d in dataset_corpus:
            corpora[CoRE[corpus_size]][d["_id"]] = {
                "title": d["title"],
                "text": d["text"],
            }
    for corpus_size in corpora:
        click.echo(
            f"Loaded {len(corpora[corpus_size])} documents for corpus of size {corpus_size}"
        )

    # Simplify qrels
    qrels_relevant_only = defaultdict(dict)
    for qid in qrels:
        for docid in qrels[qid]:
            if qrels[qid][docid] == "relevant":
                qrels_relevant_only[qid][docid] = 1

    return corpora, queries, qrels, qrels_relevant_only
