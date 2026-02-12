from typing import List, Dict, Optional

from transformers import AutoTokenizer

from splitters import AbstractSplitter


class ContextEnrichedSplitter(AbstractSplitter):
    """
    Chunker that enriches each text chunk with the document's title or a summary of its content.
    """

    def __init__(self, summary_splitter: Optional[AbstractSplitter], tokenizer: AutoTokenizer,
                 tokens_per_chunk: int = 512, chunk_overlap: int = 25):
        """
        Initializes the splitter class with the tokenizer to be used, maximum chunk length and overlap.

        :param summary_splitter: The summary splitter to use for generating document summaries.
        :param tokenizer: The tokenizer to use to convert text to tokens.
        :param tokens_per_chunk: The maximum number of tokens per chunk.
        :param chunk_overlap: The number of overlapping tokens between consecutive chunks.
        """
        assert 0 <= chunk_overlap < tokens_per_chunk
        self.tokenizer = tokenizer
        self.tokens_per_chunk = tokens_per_chunk
        self.chunk_overlap = chunk_overlap
        self.summary_splitter = summary_splitter

    def split_text(self, documents: List[Dict[str, str]]) -> List[List[str]]:
        """
        Splits the document content into chunks of equal size, then prepends the document title to each chunk.

        :param documents: The documents to be chunked.
        :return: The created text chunks.
        """
        chunks = []
        summaries = None

        if self.summary_splitter:
            summaries = self.summary_splitter.split_text(documents)
            summaries = [summary[0] for summary in summaries]
            assert len(summaries) == len(documents)

        for idx, doc in enumerate(documents):
            summary = None if summaries is None else summaries[idx]
            title, text = doc.get("title", "").strip(), doc["text"].strip()
            tokens = self.tokenizer(text, return_tensors="pt")["input_ids"][0]
            start_idx = 0
            chunked_text = []

            while start_idx < len(tokens):
                start_idx = 0 if start_idx == 0 else start_idx - self.chunk_overlap
                end_idx = min(start_idx + self.tokens_per_chunk, len(tokens))
                chunked_text.append(self.tokenizer.decode(tokens[start_idx:end_idx], skip_special_tokens=True))
                start_idx = end_idx

            if summary:
                chunks.append([f"{summary} {chunk}" for chunk in chunked_text])
            elif title:
                title = title if title.endswith(".") else f"{title}."
                chunks.append([f"{title} {chunk}" for chunk in chunked_text])
            else:
                chunks.append(chunked_text)

        return chunks
