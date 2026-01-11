from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable


@dataclass
class HyDE:
    """
    HyDE: Hypothetical Document Embeddings (query expansion).

    Given a query, generate a "hypothetical answer/document" using an LLM,
    then use that synthetic text as the retrieval query for dense/late-interaction retrievers.

    This class does NOT retrieve by itself; it just expands the query text.

    Plug it into your retriever as:
      expanded = hyde.expand(query)
      dense_hits = vs.search(expanded, k)
    """

    llm_generate: Callable[[str], str]
    prompt_template: str = (
        "Write a concise, factual passage that would directly answer the user query. "
        "Do not add citations. Do not mention that this is hypothetical.\n\nQuery:\n{query}\n\nPassage:"
    )
    max_chars: int = 2000

    def expand(self, query: str) -> str:
        prompt = self.prompt_template.format(query=query)
        text = str(self.llm_generate(prompt)).strip()
        if len(text) > self.max_chars:
            text = text[: self.max_chars]
        # If generation fails or empty, fallback to original query
        return text if text else query
