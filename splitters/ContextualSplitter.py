import asyncio
from typing import List, Dict

import click
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from splitters import AbstractSplitter, TokenSplitter


class ContextualSplitter(AbstractSplitter):
    """
    Chunker implementing contextual retrieval where each chunk is combined with additional information that situates it
    in the overall document context.
    """

    def __init__(self, api_key: str, url: str, model: str, tokenizer: AutoTokenizer, tokens_per_chunk: int = 512,
                 chunk_overlap: int = 25, max_parallel_requests: int = 20):
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
        :param max_parallel_requests: The maximum number of concurrent LLM requests.
        """
        assert 0 <= chunk_overlap < tokens_per_chunk
        assert max_parallel_requests > 0
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
        self.url = url
        self.api_key = api_key
        self.splitter = TokenSplitter(tokenizer, tokens_per_chunk, chunk_overlap)
        self.client = None
        self.model = model
        self.max_chars = 800_000
        self.char_overlap = 100
        self.counter = 0
        self.max_parallel_requests = max_parallel_requests
        self.semaphore = None

    async def _call_llm(self, content: str) -> str:
        """
        Prompts the LLM endpoint to generate a document summary or a context for a chunk.

        :param content: The prompt.
        :return: The summary or context.
        """
        if self.semaphore is None:
            raise RuntimeError("ContextualSplitter semaphore is not initialized.")

        async with self.semaphore:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
            )
            return response.choices[0].message.content.strip()

    async def _create_partial_summaries(self, text: str) -> str:
        """
        Summarizes parts of a long document and returns a concatenated string of these summaries.

        :param text: The long text to summarize.
        :return: The concatenated string of the partial summaries.
        """
        tasks = []
        iterator = range(0, len(text), self.max_chars - self.char_overlap)

        for start_pos in iterator:
            end_pos = min(start_pos + self.max_chars, len(text))
            doc_part = text[start_pos:end_pos]
            prompt = f"{self.document_context.format(doc_content=doc_part)}\n---\n {self.instruct}"
            tasks.append(self._call_llm(prompt))

        results = await asyncio.gather(*tasks)

        partial_summaries = ""
        for summary in results:
            if summary:
                partial_summaries += f" {summary}"

        return partial_summaries

    async def _process_doc_chunks(self, doc_chunks: List[str], doc: Dict[str, str]) -> List[str]:
        """
        Processes a list of chunks that were created from the given document. For each chunk, an LLM is prompted to
        generate a short context for it that situates the chunk within the larger document context. If there's only a
        single chunk, i.e. the entire document, then no context needs to be added. If the document length exceeds a
        pre-defined threshold, the document is first split into smaller parts which are summarized.

        :param doc_chunks: The list of chunks to process.
        :param doc: The document to process.
        :return: The list of contextualized chunks.
        """
        tasks = []

        if len(doc_chunks) == 1:
            return doc_chunks

        if len(doc["text"]) > self.max_chars:
            click.echo(
                f"Encountered long document with {len(doc['text'])} chars. Summarizing parts to reduce document size."
            )
            doc = await self._create_partial_summaries(doc["text"])

        for chunk in doc_chunks:
            prompt = f"{self.document_context.format(doc_content=doc)}\n---\n {self.chunk_context.format(
                chunk_content=chunk)}"
            tasks.append(self._call_llm(prompt))

        return await asyncio.gather(*tasks)

    async def split_text_async(self, chunks: List[List[str]], documents: List[Dict[str, str]]) -> List[List[str]]:
        """
        Takes a list of documents and chunks to generate contextual chunks.

        :param chunks: The initial document chunks.
        :param documents: The list of documents.
        :return: The list of contextualized chunks.
        """
        try:
            self.client = AsyncOpenAI(base_url=self.url, api_key=self.api_key)
            self.semaphore = asyncio.Semaphore(self.max_parallel_requests)
            tasks = [self._process_doc_chunks(chunk, doc) for chunk, doc in zip(chunks, documents)]
            return await asyncio.gather(*tasks)
        finally:
            await self.client.close()
            self.client = None
            self.semaphore = None

    def split_text(self, documents: List[Dict[str, str]]) -> List[List[str]]:
        """
        Splits the text into chunks of equal size, then adds contextualizes each chunk by adding information that
        situates the chunk in the overall document. Overlong documents are split into smaller parts which are summarized
        and then given as the document context.

        :param documents: The documents to split.
        :return: The list of generated chunks.
        """
        extended_chunks = []
        chunks = self.splitter.split_text(documents)
        contexts = asyncio.run(self.split_text_async(chunks, documents))

        for doc_chunks, gen_contexts in zip(chunks, contexts):
            context_chunks = []

            if len(doc_chunks) > 1:
                for chunk, context in zip(doc_chunks, gen_contexts):
                    context_chunks.append(f"{context} {chunk}")
                extended_chunks.append(context_chunks)
            else:
                extended_chunks.append(doc_chunks)

        return extended_chunks
