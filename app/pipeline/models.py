from dataclasses import dataclass


@dataclass(slots=True)
class RedactedChunk:
    chunk_index: int
    page_number: int
    text_redacted: str
    text_hash: str
    heading_guess: str = ""

