from typing import List, Dict

from transformers import AutoTokenizer

from splitters.AbstractSplitter import AbstractSplitter


class TokenSplitter(AbstractSplitter):
    """
    Chunker implementing token-based text splitting using LlamaIndex.
    """

    def __init__(self, tokenizer: AutoTokenizer, tokens_per_chunk: int = 512, chunk_overlap: int = 25):
        """
        Initializes the splitter class with the tokenizer to be used, maximum chunk length and overlap.

        :param tokenizer: The tokenizer to use to convert text to tokens.
        :param tokens_per_chunk: The maximum number of tokens per chunk.
        :param chunk_overlap: The number of overlapping tokens between consecutive chunks.
        """
        assert 0 <= chunk_overlap < tokens_per_chunk
        self.tokenizer = tokenizer
        self.chunk_overlap = chunk_overlap
        self.tokens_per_chunk = tokens_per_chunk

    def split_text(self, documents: List[Dict[str, str]]) -> List[List[str]]:
        """
        Splits the texts into chunks of equal size by token length.

        :param documents: The documents to be chunked.
        :return: The created text chunks.
        """
        chunks = []
        for doc in documents:
            text = "{} {}".format(doc.get("title", ""), doc["text"]).strip()
            tokens = self.tokenizer(text, return_tensors="pt")["input_ids"][0]
            start_idx = 0
            doc_chunks = []

            while start_idx < len(tokens):
                start_idx = 0 if start_idx == 0 else start_idx - self.chunk_overlap
                end_idx = start_idx + self.tokens_per_chunk
                end_idx = min(end_idx, len(tokens))
                doc_chunks.append(self.tokenizer.decode(tokens[start_idx:end_idx], skip_special_tokens=True))
                start_idx = end_idx

            chunks.append(doc_chunks)
        return chunks
