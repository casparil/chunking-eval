from typing import List, Dict

import click
from openai import OpenAI
from transformers import AutoTokenizer

from splitters import AbstractSplitter


class SummarySplitter(AbstractSplitter):
    """
    Chunker that summarizes a document to create a single chunk out of it.
    """

    def __init__(self, api_key: str, url: str, model: str, tokenizer: AutoTokenizer, num_words: int = 500):
        """
        Initializes the splitter class with the name of the parameters required to establish a connection to an LLM
        endpoint to generate contextual chunk information and tokenizer to cut down overlong summaries.

        :param api_key: The API key used for calling the LLM endpoint.
        :param url: The URL of the LLM endpoint.
        :param model: The name of the model to use for contextualizing chunks.
        :param num_words: Controls the length of the generated summaries by prompting the model to not exceed x words.
        :param tokenizer: The tokenizer to use for tokenizing summaries.
        """
        self.instruct = (f"Summarize the content of the document above for the purpose of improving search retrieval. "
                         f"Use no more than {num_words} words. Answer only with the summary and nothing else.")
        self.document_context = """<document>{doc_content}</document>"""
        self.instruct_partial = (f"Summarize the content of the partial text summaries above for the purpose of "
                                 f"improving search retrieval. Use no more than {num_words} words. Answer only with "
                                 f"the summary and nothing else.")
        self.partial_context = """<summary>{summary}</summary>"""
        self.client = OpenAI(base_url=url, api_key=api_key)
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = tokenizer.model_max_length
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
                partial_summaries += self.partial_context.format(summary=summary)

        return partial_summaries

    def split_text(self, documents: List[Dict[str, str]]) -> List[List[str]]:
        """
        Summarizes each document to create one chunk for each document. Overlong documents are split into smaller parts
        which are summarized individually before creating a single summary out of the partial ones. Finally, summaries
        that exceed the model's maximum token input length are truncated.

        :param documents: The documents to be summarized.
        :return: The list of generated summaries.
        """
        summaries = []

        for doc in documents:
            if len(doc["text"]) > self.max_chars:
                click.echo(f"Encountered long document with {len(doc["text"])} chars. Creating partial summaries.")
                partial_summaries = self._create_partial_summaries(doc["text"])
                response = self.client.chat.completions.create(
                    model=self.model, messages=[{"role": "user", "content": f"{self.document_context.format(
                        doc_content=partial_summaries)}\n---\n {self.instruct_partial}"}]
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model, messages=[{"role": "user", "content": f"{self.document_context.format(
                        doc_content=doc)}\n---\n {self.instruct}"}]
                )

            summary = response.choices[0].message.content.strip()

            if not summary:
                click.echo(f"Summary failed for document!")
                summaries.append("")
            elif len(summary) < 20:
                click.echo(f"Encountered small summary {summary}!")
                summaries.append([summary])
            else:
                # Ensure that summary fits into model
                tokens = self.tokenizer(summary)
                if len(tokens) > self.max_length:
                    click.echo(f"Encountered long summary with number of tokens {len(tokens)} greater than maximum "
                               f"{self.max_length}.")
                    summary = self.tokenizer.decode(tokens[0:self.max_length], skip_special_tokens=True)
                summaries.append([summary])

        return summaries
