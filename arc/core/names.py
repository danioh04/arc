import re
import unicodedata

KNOWN_ALIASES = {
    "herb jones": "herbert jones",
    "cam thomas": "cameron thomas",
    "cam johnson": "cameron johnson",
    "cam reddish": "cameron reddish",
    "cam boozer": "cameron boozer",
    "cam whitmore": "cameron whitmore",
    "cam christie": "cameron christie",
    "cam spencer": "cameron spencer",
    "cam payne": "cameron payne",
    "moe wagner": "moritz wagner",
    "edrice adebayo": "bam adebayo",
    "gg jackson": "gregory jackson",
    "mo bamba": "mohamed bamba",
    "moe harkless": "maurice harkless",
    "jeffery taylor": "jeff taylor",
    "roy devyn marble": "devyn marble",
    "joseph young": "joe young",
    "kahlil felder": "kay felder",
    "wesley iwundu": "wes iwundu",
    "sviatoslav mykhailiuk": "svi mykhailiuk",
    "nicolas claxton": "nic claxton",
    "dewan huell": "dewan hernandez",
    "nahshon hyland": "bones hyland",
    "carlton carrington": "bub carrington",
    "rob dillingham": "robert dillingham",
    "mo speights": "marreese speights",
    "iggy brazdeikis": "ignas brazdeikis",
    "mitch mcgary": "mitchell mcgary",
    "ish smith": "ishmael smith",
    "oso ighodaro": "osasere ighodaro",
    "johnuel fland": "boogie fland",
    "rejean ellis": "boogie ellis",
    "phil pressey": "phl pressey",
    "adamaalpha bal": "adama bal",
}

_SUFFIXES = (r"\bjr\.?\b", r"\bsr\.?\b", r"\bii\b", r"\biii\b", r"\biv\b")


def clean_name(name: str) -> str:
    nfkd_form = unicodedata.normalize("NFKD", name)
    text = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    text = text.lower().strip()
    for suffix in _SUFFIXES:
        text = re.sub(suffix, "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return KNOWN_ALIASES.get(text, text)
