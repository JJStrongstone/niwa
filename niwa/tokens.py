"""niwa.tokens - Token counting utilities."""

import tiktoken

_encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count approximate tokens using cl100k_base encoding.

    Uses OpenAI's cl100k_base tokenizer as an approximation.
    Not exact for Claude's tokenizer but close enough for
    size estimation and context budgeting.
    """
    if not text:
        return 0
    return len(_encoder.encode(text))
