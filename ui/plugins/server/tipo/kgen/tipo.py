"""TIPO v1 prompt protocol, adapted from Apache-2.0 KGen 0.2.0.

Copyright 2024 KBlueLeaf. This file has been modified for Easy Diffusion.
The full license is distributed in ``easy-diffusion-custom-licenses``.
"""

import random
import re

from .formatter import seperate_tags
from .metainfo import (
    TARGET_TIPO,
    TARGET_TIPO_MAX,
    TARGET_TIPO_NL,
    TARGET_TIPO_NL_MAX,
)


_RESULT_PATTERN = re.compile(r"\n([^:\n]+):(.*(?:\n(?![^:\n]+:).*)*)")
_TYPE_MAP = {"short": "extended", "long": "generated"}


def _deduplicate(items):
    output = []
    seen = set()
    for item in items:
        if item not in seen:
            output.append(item)
            seen.add(item)
    return output


def apply_tipo_prompt(meta, general, nl_prompt, mode, length, expand, gen_meta=False):
    content = {"tag": general}
    if nl_prompt and (mode is None or "short" not in mode):
        content["long"] = nl_prompt
    elif nl_prompt:
        content["short"] = nl_prompt

    prompt = ""
    target = ""
    for key, value in meta.items():
        if value:
            prompt += f"{key}: {value}\n"

    if length:
        target += f"<|{length}|>"
    if mode:
        content_order = mode.split("_to_")
        target += f" <|{mode}|>"
    else:
        content_order = ["tag"]
    if gen_meta:
        target += " <|gen_meta|>"
    prompt += f"target: {target.strip()}\n"

    last_key = "tag"
    for index, key in enumerate(content_order):
        last_key = key
        if key in content and content[key].strip():
            prompt += f"{key}: {content[key]}\n"
        elif index < len(content_order) - 1:
            prompt += f"{key}: \n"
    if expand:
        return prompt.strip()
    return prompt + f"{last_key}:"


def parse_tipo_result(result):
    result = "\n" + result.removeprefix("<s>").removesuffix("</s>").strip()
    parsed = {}
    for result_type, content in _RESULT_PATTERN.findall(result):
        result_type = result_type.strip()
        content = content.strip()
        mapped_type = _TYPE_MAP.get(result_type, result_type)
        if mapped_type in parsed:
            continue
        if result_type == "tag":
            tags = [item.strip() for item in content.split(",") if item.strip()]
            categories = seperate_tags(tags)
            for key, values in categories.items():
                values = [item for item in values if item]
                if key in parsed:
                    existing = parsed[key]
                    if not isinstance(existing, list):
                        existing = [existing]
                    parsed[key] = existing + values
                else:
                    parsed[key] = values
            parsed["tag"] = tags
        elif result_type in {"short", "long"}:
            if content and content != "<|empty|>":
                parsed[mapped_type] = content
        else:
            parsed[mapped_type] = [item.strip() for item in content.split(",") if item.strip()]
    return parsed


def parse_tipo_request(
    tag_map,
    nl_prompt="",
    tag_length="long",
    nl_length="long",
    generate_extra_nl_prompt=True,
):
    general = ", ".join(tag_map.get("special", []) + tag_map.get("general", [])).strip(", ")
    meta = {
        "meta": ", ".join(tag_map.get("meta", [])),
        "rating": ", ".join(tag_map.get("rating", [])) or None,
        "artist": ", ".join(tag_map.get("artist", [])) or None,
        "characters": ", ".join(tag_map.get("characters", [])) or None,
        "copyrights": ", ".join(tag_map.get("copyrights", [])) or None,
        "quality": ", ".join(tag_map.get("quality", [])),
    }

    operations = []
    extra_operation = None
    have_tags = bool(general.strip())
    have_nl = bool(nl_prompt.strip())
    if not have_tags and not have_nl:
        operations = [(None, tag_length, True)]
        extra_operation = ("tag_to_long", nl_length, True)
    elif have_tags and not have_nl:
        operations = [(None, tag_length, True)]
        extra_operation = ("tag_to_long", nl_length, False)
    elif not have_tags and have_nl:
        operations = [("long_to_tag", tag_length, True)]
        extra_operation = ("short_to_tag_to_long", nl_length, False)
    else:
        operations = [
            ("short_to_tag", tag_length, True),
            ("tag_to_long", nl_length, True),
        ]
        extra_operation = ("short_to_tag_to_long", nl_length, False)
    if generate_extra_nl_prompt and extra_operation:
        operations.append(extra_operation)
    return meta, operations, general, nl_prompt


def _allowed(value, ban_patterns):
    return not any(pattern.search(value) for pattern in ban_patterns)


def _post_process(parsed, general, nl_prompt, mode, length, rng, ban_patterns):
    if mode is None:
        parsed.pop("extended", None)
        parsed.pop("generated", None)
    else:
        if "long" not in mode:
            parsed.pop("generated", None)
        if "short" not in mode:
            parsed.pop("extended", None)
    if "generated" in parsed and nl_prompt and not parsed.get("extended", "").strip():
        parsed["extended"] = parsed.pop("generated")

    input_tags = [tag.strip() for tag in general.split(",") if tag.strip()]
    input_sentences = [sentence.strip() for sentence in nl_prompt.split(".") if sentence.strip()]
    input_general = [tag for tag in parsed.get("general", []) if tag in input_tags]
    output_general = [
        tag
        for tag in parsed.get("general", [])
        if _allowed(tag, ban_patterns) and tag not in input_tags
    ]
    rng.shuffle(output_general)
    output_sentences = [
        sentence.strip()
        for sentence in parsed.get("extended", "").split(".")
        if sentence.strip()
        and _allowed(sentence.strip(), ban_patterns)
        and sentence.strip() not in input_sentences
    ]
    if input_sentences and output_sentences and input_sentences[-1] in output_sentences[0]:
        input_sentences[-1] = output_sentences.pop(0)
    if output_sentences:
        last = output_sentences.pop()
        rng.shuffle(output_sentences)
        output_sentences.append(last)

    max_tags = TARGET_TIPO_MAX[length]
    max_sentences = TARGET_TIPO_NL_MAX[length]
    output_general = output_general[: max(max_tags - len(input_general), 0)]
    output_sentences = output_sentences[: max(max_sentences - len(input_sentences), 0)]

    generated = [
        sentence.strip()
        for sentence in parsed.get("generated", "").split(".")
        if sentence.strip() and _allowed(sentence.strip(), ban_patterns)
    ][:max_sentences]
    parsed["general"] = _deduplicate(input_general + output_general)
    parsed["extended"] = ". ".join(_deduplicate(input_sentences + output_sentences))
    if generated:
        parsed["generated"] = ". ".join(_deduplicate(generated))
    return parsed


def _enough_output(parsed, target, length):
    counts = {
        "tag": len(parsed.get("special", []) + parsed.get("general", [])),
        "short": len([item for item in parsed.get("extended", "").split(".") if item.strip()]),
        "long": len([item for item in parsed.get("generated", "").split(".") if item.strip()]),
    }
    minimums = {
        "tag": TARGET_TIPO[length],
        "short": TARGET_TIPO_NL[length],
        "long": TARGET_TIPO_NL[length],
    }
    if target == "long" and not parsed.get("generated"):
        target = "short"
    return counts.get(target, 0) >= minimums.get(target, 0)


def run_tipo(
    complete,
    tag_map,
    nl_prompt,
    tag_length,
    nl_length,
    prompt_format,
    aspect_ratio,
    ban_tags,
    seed,
    max_retry,
    max_same_output,
):
    try:
        ban_patterns = [re.compile(pattern) for pattern in ban_tags]
    except re.error as error:
        raise ValueError(f"Invalid ban-tag regular expression: {error}") from error

    meta, operations, general, nl_prompt = parse_tipo_request(
        tag_map,
        nl_prompt,
        tag_length,
        nl_length,
        generate_extra_nl_prompt=(not nl_prompt and "<|extended|>" in prompt_format)
        or "<|generated|>" in prompt_format,
    )
    meta["aspect_ratio"] = f"{aspect_ratio:.1f}"
    rng = random.Random(seed)
    parsed = {}

    for operation_index, (mode, length, expand) in enumerate(operations):
        is_last = operation_index == len(operations) - 1
        previous_outputs = set()
        same_output_count = 0
        for retry in range(max_retry + 1):
            prompt = apply_tipo_prompt(meta, general, nl_prompt, mode, length, expand, is_last)
            result = complete(prompt, seed + retry)
            parsed = parse_tipo_result(result)
            parsed = _post_process(parsed, general, nl_prompt, mode, length, rng, ban_patterns)
            target = mode.split("_to_")[-1] if mode else "tag"
            if _enough_output(parsed, target, length):
                break
            if result in previous_outputs:
                same_output_count += 1
                if same_output_count >= max_same_output:
                    break
            else:
                same_output_count = 0
                previous_outputs.add(result)

        if not is_last:
            if parsed.get("generated") and nl_prompt:
                parsed["extended"] = parsed.pop("generated")
            nl_prompt = parsed.get("generated") or parsed.get("extended") or nl_prompt
            general = ", ".join(parsed.get("special", []) + parsed.get("general", []))
            nl_prompt = nl_prompt.strip()
    return parsed
