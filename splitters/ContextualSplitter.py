from typing import List, Dict

import click
from openai import OpenAI
from transformers import AutoTokenizer

from splitters import AbstractSplitter, TokenSplitter


class ContextualSplitter(AbstractSplitter):
    """
    Chunker implementing contextual retrieval where each chunk is combined with additional information that situates it
    in the overall document context.
    """

    def __init__(self, api_key: str, url: str, model: str, tokenizer: AutoTokenizer, tokens_per_chunk: int = 512,
                 chunk_overlap: int = 25):
        """
        Initializes the splitter class with the name of the parameters required to establish a connection to an LLM
        endpoint to generate contextual chunk information and parameters to initialize a simple token splitter that
        generates the chunks.

        :param api_key: The API key used for calling the LLM endpoint.
        :param url: The URL of the LLM endpoint.
        :param model: The name of the model to use for contextualizing chunks.
        :param tokenizer: The tokenizer to use for tokenizing documents.
        :param tokens_per_chunk: The maximum number of tokens per chunk.
        :param chunk_overlap: The number of overlapping tokens between consecutive chunks.
        """
        assert 0 <= chunk_overlap < tokens_per_chunk
        self.instruct = """
        Summarize the content of the document above for the purpose of improving search retrieval.
        Use no more than 1000 words.
        Answer only with the summary and nothing else.
        """
        self.document_context = """
        <document>
        {doc_content}
        </document>
        """
        self.chunk_context = """
        Here is the chunk we want to situate within the whole document given above
        <chunk>
        {chunk_content}
        </chunk>
        
        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
        Answer only with the succinct context and nothing else.
        """
        self.splitter = TokenSplitter(tokenizer, tokens_per_chunk, chunk_overlap)
        self.client = OpenAI(base_url=url, api_key=api_key)
        self.model = model
        self.max_chars = 800_000
        self.char_overlap = 100

    def _create_partial_summaries(self, text: str) -> str:
        """
        Summarizes parts of a long document and returns a concatenated string of these summaries.

        :param text: The long text to summarize.
        :return: The concatenated string of the partial summaries.
        """
        partial_summaries = ""
        iterator = range(0, len(text), self.max_chars - self.char_overlap)

        for start_pos in iterator:
            end_pos = min(start_pos + self.max_chars, len(text))
            doc_part = text[start_pos:end_pos]
            response = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": f"{self.document_context.format(
                    doc_content=doc_part)}\n---\n {self.instruct}"}]
            )
            summary = response.choices[0].message.content.strip()

            if summary:
                partial_summaries += f" {summary}"

        return partial_summaries

    def split_text(self, documents: List[Dict[str, str]]) -> List[List[str]]:
        """
        Splits the text into chunks of equal size, then adds contextualizes each chunk by adding information that
        situates the chunk in the overall document. Overlong documents are split into smaller parts which are summarized
        and then given as the document context.

        :param documents: The documents to split.
        :return: The list of generated chunks.
        """
        chunks, extended_chunks = [], []
        chunks = self.splitter.split_text(documents)

        for doc_chunks, doc in zip(chunks, documents):
            context_chunks = []

            if len(doc_chunks) > 1:
                if len(doc["text"]) > self.max_chars:
                    click.echo(f"Encountered long document with {len(doc["text"])} chars. Summarizing parts to reduce "
                               f"document size.")
                    doc = self._create_partial_summaries(doc["text"]).strip()

                for chunk in doc_chunks:
                    response = self.client.chat.completions.create(
                        model=self.model, messages=[{"role": "user", "content": f"{self.document_context.format(
                            doc_content=doc)}\n---\n {self.chunk_context.format(chunk_content=chunk)}"}]
                    )
                    context = response.choices[0].message.content.strip()
                    context_chunks.append(f"{context} {chunk}")
                extended_chunks.append(context_chunks)
            else:
                extended_chunks.append(doc_chunks)

        return extended_chunks
