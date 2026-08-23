# Adapted for Easy Diffusion from KGen 0.2.0 (Apache-2.0).
import os
import pathlib
import re

from .metainfo import SPECIAL, POSSIBLE_QUALITY_TAGS, RATING_TAGS


tag_list_folder = pathlib.Path(os.path.dirname(__file__)) / ".." / "tag-list"
tag_lists = {
    os.path.splitext(f)[0]: set(open(tag_list_folder / f).read().strip().split("\n"))
    for f in os.listdir(tag_list_folder)
    if f.endswith(".txt")
}
tag_lists["special"] = set(SPECIAL)
tag_lists["quality"] = set(POSSIBLE_QUALITY_TAGS)
tag_lists["rating"] = set(RATING_TAGS)


def seperate_tags(all_tags):
    all_tags = [tag.strip() for tag in all_tags if tag.strip()]
    tag_map = {cate: [] for cate in tag_lists.keys()}
    tag_map["general"] = []
    for tag in all_tags:
        for cate in tag_lists.keys():
            if tag in tag_lists[cate]:
                tag_map[cate].append(tag)
                break
            if tag.replace("_", " ") in tag_lists[cate]:
                tag_map[cate].append(tag.replace("_", " "))
                break
        else:
            tag_map["general"].append(tag if len(tag) < 4 else tag.replace("_", " "))
    return tag_map


def apply_format(tag_map, format):
    if "<|extended|>" in format and not tag_map.get("extended", ""):
        format = format.replace("<|extended|>", "<|generated|>")
    for type in tag_map:
        if f"<|{type}|>" in format:
            if not tag_map[type]:
                format = format.replace(f"<|{type}|>,", "")
                format = format.replace(f"<|{type}|>", "")
            else:
                data = tag_map[type]
                if isinstance(data, list):
                    data = ", ".join(data)
                format = format.replace(f"<|{type}|>", data)
    format = re.sub(r"<\|(?:(?!<\|.*\|>).)*\|>", "", format)
    format = re.sub(r",\s*,+", ",", format)
    format = re.sub(r",[ \t]*\n", "\n", format)
    format = re.sub(r"(?m)^[ \t]*[,.]+[ \t]*$", "", format)
    format = re.sub(r"[ \t]+\.", ".", format)
    format = re.sub(r"\n +", "\n", format)
    format = re.sub(r"\n\n\n+", "\n\n", format)
    format = re.sub(r"  +", " ", format)
    return format.strip().strip(",")


def apply_dtg_prompt(tag_map, target="", aspect_ratio=1.0):
    special_tags = ", ".join(tag_map.get("special", []))
    rating = ", ".join(tag_map.get("rating", []))
    artist = ", ".join(tag_map.get("artist", []))
    characters = ", ".join(tag_map.get("characters", []))
    copyrights = ", ".join(tag_map.get("copyrights", []))
    general = ", ".join(tag_map.get("general", []))
    aspect_ratio = aspect_ratio
    prompt = f"""
rating: {rating or '<|empty|>'}
artist: {artist.strip() or '<|empty|>'}
characters: {characters.strip() or '<|empty|>'}
copyrights: {copyrights.strip() or '<|empty|>'}
aspect ratio: {f"{aspect_ratio:.1f}" or '<|empty|>'}
target: {'<|' + target + '|>' if target else '<|long|>'}
general: {special_tags}, {general.strip().strip(",")}<|input_end|>
""".strip()

    return prompt


if __name__ == "__main__":
    from json import dumps

    print(tag_lists.keys())
    print([len(t) for t in tag_lists.values()])
    print(
        dumps(
            tag_map := seperate_tags(
                [
                    "1girl",
                    "fukuro daizi",
                    "kz oji",
                    "henreader",
                    "ask (askzy)",
                    "aki99",
                    "masterpiece",
                    "newest",
                    "absurdres",
                    "loli",
                    "solo",
                    "dragon girl",
                    "dragon horns",
                    "white dress",
                    "long hair",
                    "side up",
                    "river",
                    "tree",
                    "forest",
                    "pointy ears",
                    ":3",
                    "blue hair",
                    "blush",
                    "breasts",
                    "collarbone",
                    "dress",
                    "eyes visible through hair",
                    "fang",
                    "looking at viewer",
                    "nature",
                    "off shoulder",
                    "open mouth",
                    "orange eyes",
                    "tail",
                    "twintails",
                    "wings",
                ]
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    print()
    print(
        apply_format(
            tag_map,
            """<|special|>, 
<|characters|>, <|copyrights|>, 
<|artist|>, 

<|general|>, 

<|quality|>, <|meta|>, <|rating|>""",
        )
    )
    print()
    print()
    print(apply_dtg_prompt(tag_map, 1.0))
