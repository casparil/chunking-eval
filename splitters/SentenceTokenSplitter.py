from typing import List, Dict

import click
from llama_index.core.text_splitter import SentenceSplitter, TokenTextSplitter
from transformers import AutoTokenizer

from splitters.AbstractSplitter import AbstractSplitter


class SentenceTokenSplitter(AbstractSplitter):
    """
    Chunker that tries to end chunks at sentence boundaries using LlamaIndex
    """

    def __init__(self, tokenizer: AutoTokenizer, tokens_per_chunk: int = 512, chunk_overlap: int = 25):
        """
        Initializes the splitter class with the tokenizer to be used, maximum chunk length and overlap.

        :param tokenizer: The tokenizer to use to convert text to tokens.
        :param tokens_per_chunk: The maximum number of tokens per chunk.
        :param chunk_overlap: The number of overlapping tokens between consecutive chunks.
        """
        assert 0 <= chunk_overlap < tokens_per_chunk
        self.splitter = SentenceSplitter(chunk_size=tokens_per_chunk, chunk_overlap=chunk_overlap,
                                         tokenizer=tokenizer.encode)

    def split_text(self, documents: List[Dict[str, str]]) -> List[List[str]]:
        """
        Splits the texts into chunks with a preference for complete sentences.

        :param documents: The document to be chunked.
        :return: The created text chunks.
        """
        chunks = []
        for doc in documents:
            text = "{} {}".format(doc.get("title", ""), doc["text"]).strip()
            try:
                chunks.append(self.splitter.split_text(text))
            except RecursionError:
                # Split by token count if maximum recursion depth is exceeded
                click.echo(f"Chunking document with title {doc.get("title", "")} threw recursion error. Switching to "
                           f"token-based chunking.")
                token_splitter = TokenTextSplitter(chunk_size=self.splitter.chunk_size,
                                                   chunk_overlap=self.splitter.chunk_overlap,
                                                   tokenizer=self.splitter._tokenizer)
                chunks.append(token_splitter.split_text(text))
        return chunks
