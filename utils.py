"""Local copy of split_solution_into_chunks from thought-anchors/utils.py."""

from typing import List


def split_solution_into_chunks(solution_text: str) -> List[str]:
    if "<think>" in solution_text:
        solution_text = solution_text.split("<think>")[1].strip()
    if "</think>" in solution_text:
        solution_text = solution_text.split("</think>")[0].strip()

    sentence_ending_tokens = [".", "?", "!"]
    paragraph_ending_patterns = ["\n\n", "\r\n\r\n"]

    chunks = []
    current_chunk = ""
    i = 0
    while i < len(solution_text):
        current_chunk += solution_text[i]

        is_paragraph_end = any(
            i + len(p) <= len(solution_text) and solution_text[i: i + len(p)] == p
            for p in paragraph_ending_patterns
        )
        is_sentence_end = (
            i < len(solution_text) - 1
            and solution_text[i] in sentence_ending_tokens
            and solution_text[i + 1] in (" ", "\n")
        )

        if is_paragraph_end or is_sentence_end:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""
        i += 1

    # Merge chunks shorter than 10 characters into neighbours
    i = 0
    while i < len(chunks):
        if len(chunks[i]) < 10:
            if i == len(chunks) - 1:
                if i > 0:
                    chunks[i - 1] += " " + chunks[i]
                    chunks.pop(i)
            else:
                chunks[i + 1] = chunks[i] + " " + chunks[i + 1]
                chunks.pop(i)
            if i == 0 and len(chunks) == 1:
                break
        else:
            i += 1

    return chunks
