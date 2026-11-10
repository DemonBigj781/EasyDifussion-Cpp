"""Prompt-mode normalization performed before a render enters the queue."""

import math
import re
from dataclasses import dataclass
from typing import List


NATURAL_PROMPT_MODE = "natural"
TAG_PROMPT_MODE = "tags"
PROMPT_MODES = frozenset((NATURAL_PROMPT_MODE, TAG_PROMPT_MODE))

_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


@dataclass
class _Piece:
    kind: str
    text: str
    depth: int = 0


def normalize_prompt_mode(mode: str) -> str:
    normalized = str(mode or NATURAL_PROMPT_MODE).strip().lower()
    if normalized not in PROMPT_MODES:
        raise ValueError(
            f"Unknown prompt mode {mode!r}; expected 'natural' or 'tags'."
        )
    return normalized


def _lex_tags(prompt: str) -> List[_Piece]:
    pieces = []
    depth = 0
    position = 0
    while position < len(prompt):
        current = prompt[position]
        if current == "\\" and position + 1 < len(prompt):
            pieces.append(_Piece("atom", prompt[position : position + 2], depth))
            position += 2
            continue
        if current.isspace():
            end = position + 1
            while end < len(prompt) and prompt[end].isspace():
                end += 1
            pieces.append(_Piece("space", prompt[position:end], depth))
            position = end
            continue
        if current in "([":
            pieces.append(_Piece("open", current, depth))
            depth += 1
            position += 1
            continue
        if current in ")]":
            depth = max(0, depth - 1)
            pieces.append(_Piece("close", current, depth))
            position += 1
            continue
        if current in ",;":
            pieces.append(_Piece("separator", current, depth))
            position += 1
            continue

        end = position + 1
        while end < len(prompt):
            value = prompt[end]
            if value.isspace() or value in "()[],;" or value == "\\":
                break
            end += 1
        text = prompt[position:end]
        if text in ("BREAK", "AND"):
            pieces.append(_Piece(text.lower(), text, depth))
        else:
            pieces.append(_Piece("atom", text, depth))
        position = end
    return pieces


def _format_weight(value: float) -> str:
    return format(value, ".8g")


def _has_weightable_name(text: str) -> bool:
    return any(value.isalpha() or value == "_" for value in text)


def _normalize_atom(piece: _Piece, fuse_relations: bool) -> _Piece:
    text = piece.text
    if text.startswith("\\"):
        return piece

    plus_count = len(text) - len(text.rstrip("+"))
    if plus_count and plus_count < len(text) and _has_weightable_name(text[:-plus_count]):
        base = text[:-plus_count]
        if fuse_relations:
            base = base.replace(":", " ")
        weight = _format_weight(math.pow(1.1, plus_count))
        weighted = f"{base}:{weight}" if piece.depth else f"({base}:{weight})"
        return _Piece("atom", weighted, piece.depth)

    fields = text.split(":")
    if len(fields) > 1 and all(fields):
        if _NUMBER.fullmatch(fields[-1]) and _has_weightable_name(":".join(fields[:-1])):
            base = " ".join(fields[:-1])
            if not fuse_relations:
                base = ":".join(fields[:-1])
            weighted = f"{base}:{fields[-1]}" if piece.depth else f"({base}:{fields[-1]})"
            return _Piece("atom", weighted, piece.depth)
        if fuse_relations:
            return _Piece("atom", " ".join(fields), piece.depth)

    return piece


def _neighbor_kind(pieces: List[_Piece], start: int, step: int):
    position = start + step
    while 0 <= position < len(pieces):
        if pieces[position].kind != "space":
            return pieces[position].kind
        position += step
    return None


def _interpret_tags(prompt: str) -> str:
    pieces = [_normalize_atom(piece, True) for piece in _lex_tags(prompt)]
    output = []
    for index, piece in enumerate(pieces):
        if piece.kind == "space":
            previous = _neighbor_kind(pieces, index, -1)
            following = _neighbor_kind(pieces, index, 1)
            if previous in ("atom", "close") and following in ("atom", "open"):
                output.append(", ")
            elif previous == "separator" and following not in (None, "close"):
                output.append(" ")
            continue
        if piece.kind == "separator":
            while output and output[-1].endswith(" "):
                output[-1] = output[-1].rstrip()
                if not output[-1]:
                    output.pop()
            output.append(",")
            continue
        if piece.kind == "and":
            while output and output[-1].endswith(" "):
                output[-1] = output[-1].rstrip()
            output.append(", ")
            continue
        if piece.kind == "break":
            while output and output[-1].endswith(" "):
                output[-1] = output[-1].rstrip()
            output.append(" BREAK ")
            continue
        output.append(piece.text)

    interpreted = "".join(output).strip(" ,")
    interpreted = re.sub(r",(?:\s*,)+", ",", interpreted)
    interpreted = re.sub(r",\s*", ", ", interpreted)
    interpreted = re.sub(r"\s+BREAK\s+", " BREAK ", interpreted)
    return interpreted


def _interpret_natural(prompt: str) -> str:
    return "".join(
        _normalize_atom(piece, False).text if piece.kind == "atom" else piece.text
        for piece in _lex_tags(prompt)
    )


def interpret_prompt(prompt: str, mode: str = NATURAL_PROMPT_MODE) -> str:
    """Return an immutable queue-ready prompt for the requested interpretation."""

    normalized_mode = normalize_prompt_mode(mode)
    if normalized_mode == TAG_PROMPT_MODE:
        return _interpret_tags(prompt)
    return _interpret_natural(prompt)
