"""
Ottoman Turkish Transliterator
================================
Self-contained engine that converts Modern Turkish text into Ottoman
Arabic-script (حروف عثمانیه).  No Jupyter / IPython dependencies.

Public API
----------
    from app.transliterator import OttomanTransliterator

    engine = OttomanTransliterator()
    result = engine.transliterate("vermemişlerdir.")
    # result.ottoman, result.tokens, result.confidence
"""

from __future__ import annotations

import csv
import logging
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import zeyrek

# Patch: bypass nltk punkt_tab requirement (Zeyrek passes single tokens anyway)
import zeyrek.morphology as _zm
def _simple_tokenize(text: str) -> list[str]:
    return [text.strip()] if text.strip() else []
_zm._tokenize_text = _simple_tokenize

logging.getLogger("zeyrek").setLevel(logging.ERROR)

# Module-level cache (instance lru_cache doesn't work across calls)
_module_analyzer = None
def _get_module_analyzer():
    global _module_analyzer
    if _module_analyzer is None:
        _module_analyzer = zeyrek.MorphAnalyzer()
    return _module_analyzer

@lru_cache(maxsize=8192)
def _cached_zeyrek_analyze(word: str):
    return list(_get_module_analyzer().analyze(word))


# ============================================================
# §1  CONFIGURATION
# ============================================================

ENABLE_HISTORICAL_ORTHOGRAPHY: bool = True

SCORE_MAP: dict[str, float] = {
    "exact":    1.0,
    "override": 1.0,
    "tags":     1.0,
    "punct":    1.0,
    "english":  0.7,
    "auto":     0.6,
    "missing":  0.0,
}

# Vowel sets
KALIN_UNLULER    = set("aıou")
INCE_UNLULER     = set("eiöü")
YUVARLAK_UNLULER = set("ouöü")
DUZ_UNLULER      = set("aeıi")
_TUM_UNLULER     = KALIN_UNLULER | INCE_UNLULER

PHONEME_MAP: dict[str, str] = {
    "a": "ا", "e": "ه", "ı": "ی", "i": "ی",
    "o": "و", "ö": "و", "u": "و", "ü": "و",
    "b": "ب", "c": "ج", "ç": "چ", "d": "د",
    "f": "ف", "g": "گ", "ğ": "غ", "h": "ه",
    "j": "ژ", "l": "ل", "m": "م", "n": "ن",
    "p": "پ", "r": "ر", "ş": "ش", "v": "و",
    "y": "ی", "z": "ز", "k": "ک", "q": "ق",
}
PUNCTUATION_MAP: dict[str, str] = {",": "،", ";": "؛", "?": "؟", "%": "٪"}

TR_LOWER_MAP     = str.maketrans("IİÇĞÖŞÜ", "ıiçğöşü")
TR_NORMALIZE_MAP = str.maketrans("ÂâÎîÛû",  "AaİiUu")

_TR_WORD_CHARS = r"a-zA-Z0-9_ğüşıöçâîûÂÎÛĞÜŞİÖÇ"
_TR_WORD       = f"[{_TR_WORD_CHARS}]"
_TOKEN_RE      = re.compile(
    rf"{_TR_WORD}+(?:[\u2018\u2019']{_TR_WORD}+)?|[^\s{_TR_WORD_CHARS}]+"
)
_WORD_TOKEN_RE = re.compile(
    rf"^{_TR_WORD}+(?:[\u2018\u2019']{_TR_WORD}+)?$"
)

# ============================================================
# §2  FSM STATES
# ============================================================
ROOT               = "ROOT"
DERIVATION         = "DERIVATION"
DERIVATION_NOMINAL = "DERIVATION_NOMINAL"
VOICE              = "VOICE"
NEGATION           = "NEGATION"
TENSE              = "TENSE"
PERSON             = "PERSON"
CASE               = "CASE"
PLURAL_ST          = "PLURAL"
POSSESSIVE         = "POSSESSIVE"
COPULA_ST          = "COPULA"

ROOT_POS_TAGS          = {"NOUN", "VERB", "ADJ", "ADV", "PRON", "NUM", "QUES"}
EMPTY_TAGS             = {"NOM", "A3SG"}
PERSON_TAGS            = {"A1SG", "A2SG", "A3SG", "A1PL", "A2PL", "A3PL"}
POSSESSIVE_TAGS        = {"P1SG", "P2SG", "P3SG", "P1PL", "P2PL", "P3PL"}
CASE_TAGS              = {"NOM", "ACC", "DAT", "LOC", "ABL", "GEN", "INS", "REL", "REL_LOC"}
VERBAL_TENSE_TAGS      = {"PAST", "NARR", "PROG", "FUTURE", "FUT_PART", "AOR", "COND", "NECES", "OPT"}
VOICE_TAGS             = {"PASSIVE", "CAUSATIVE", "RECIPROCAL", "REFLEXIVE"}
VERBAL_DERIVATION_TAGS = {"ABLE", "UNABLE", "INF1", "INF2", "PART", "CONV", "CONV_SINCE", "FUT_PART"}
NOMINAL_DERIVATION_TAGS = {
    "NOM_DER_LIK", "NOM_DER_LI", "NOM_DER_SIZ",
    "NOM_DER_SEL", "NOM_DER_CI", "NOM_DER_DAS", "NOM_DER_MSI",
}

CASE_ST = "CASE"

TAG_FSM: dict[str, list[str]] = {
    ROOT:               [DERIVATION, DERIVATION_NOMINAL, VOICE, NEGATION, TENSE,
                         PERSON, PLURAL_ST, POSSESSIVE, CASE_ST, COPULA_ST],
    DERIVATION:         [DERIVATION, VOICE, NEGATION, TENSE, PLURAL_ST, POSSESSIVE, CASE_TAGS],
    DERIVATION_NOMINAL: [DERIVATION_NOMINAL, PLURAL_ST, POSSESSIVE, CASE_TAGS, COPULA_ST],
    VOICE:              [VOICE, NEGATION, TENSE, PLURAL_ST, POSSESSIVE, CASE_TAGS],
    NEGATION:           [VOICE, DERIVATION, TENSE],
    TENSE:              [TENSE, PERSON, COPULA_ST, CASE_TAGS],
    PERSON:             [COPULA_ST, CASE_TAGS],
    PLURAL_ST:          [POSSESSIVE, CASE_TAGS, COPULA_ST],
    POSSESSIVE:         [CASE_TAGS, PLURAL_ST, COPULA_ST],
    CASE_ST:          [CASE_ST, PLURAL_ST, COPULA_ST],
    COPULA_ST:          [TENSE, PERSON, CASE_TAGS],
}

TAG_TO_STATE: dict[str, str] = {
    "ABLE": DERIVATION, "UNABLE": DERIVATION, "INF1": DERIVATION,
    "INF2": DERIVATION, "PART": DERIVATION, "CONV": DERIVATION,
    "CONV_SINCE": DERIVATION, "FUT_PART": DERIVATION,
    "PASSIVE": VOICE, "CAUSATIVE": VOICE, "RECIPROCAL": VOICE, "REFLEXIVE": VOICE,
    "NEG": NEGATION,
    "PAST": TENSE, "NARR": TENSE, "PROG": TENSE, "FUTURE": TENSE,
    "AOR": TENSE, "COND": TENSE, "NECES": TENSE, "OPT": TENSE,
    "A1SG": PERSON, "A2SG": PERSON, "A3SG": PERSON,
    "A1PL": PERSON, "A2PL": PERSON, "A3PL": PERSON,
    "PLURAL": PLURAL_ST,
    "P1SG": POSSESSIVE, "P2SG": POSSESSIVE, "P3SG": POSSESSIVE,
    "P1PL": POSSESSIVE, "P2PL": POSSESSIVE, "P3PL": POSSESSIVE,
    "NOM": CASE_ST, "ACC": CASE_ST, "DAT": CASE_ST, "LOC": CASE_ST,
    "ABL": CASE_ST, "GEN": CASE_ST, "INS": CASE_ST,
    "REL": CASE_ST, "REL_LOC": CASE_ST,
    "COPULA": COPULA_ST, "COPULA_ASSERT": COPULA_ST,
    "NOM_DER_LIK": DERIVATION_NOMINAL, "NOM_DER_LI": DERIVATION_NOMINAL,
    "NOM_DER_SIZ": DERIVATION_NOMINAL, "NOM_DER_SEL": DERIVATION_NOMINAL,
    "NOM_DER_CI": DERIVATION_NOMINAL, "NOM_DER_DAS": DERIVATION_NOMINAL,
    "NOM_DER_MSI": DERIVATION_NOMINAL,
}

# ============================================================
# §3  BASIC HELPERS
# ============================================================

def normalize_tr_text(t: str) -> str:
    return t.translate(TR_NORMALIZE_MAP) if t else t

def lower_tr(t: str) -> str:
    return normalize_tr_text(t).translate(TR_LOWER_MAP).lower() if t else t

def fold_tr(t: str) -> str:
    if not t:
        return t
    t = lower_tr(t)
    return (t.replace("ç","c").replace("ğ","g").replace("ı","i")
             .replace("ö","o").replace("ş","s").replace("ü","u"))

def is_word_token(token: str) -> bool:
    return bool(_WORD_TOKEN_RE.fullmatch(token))

def convert_ottoman_punctuation(text: str) -> str:
    return "".join(PUNCTUATION_MAP.get(c, c) for c in text)

def last_vowel(text: str) -> str:
    for ch in reversed(lower_tr(text or "")):
        if ch in _TUM_UNLULER:
            return ch
    return "a"

def is_vowel(ch: str) -> bool:
    return lower_tr(ch or "")[:1] in _TUM_UNLULER

def starts_with_vowel(t: str) -> bool:
    return bool(t) and is_vowel(t[0])

def ends_with_vowel(t: str) -> bool:
    return bool(t) and is_vowel(t[-1])

def choose_harmony_A(s: str) -> str:
    return "a" if last_vowel(s) in KALIN_UNLULER else "e"

def choose_harmony_I(s: str) -> str:
    v = last_vowel(s)
    return {"a":"ı","ı":"ı","e":"i","i":"i","o":"u","u":"u"}.get(v,"ü")

def choose_initial_D(s: str) -> str:
    return "t" if s and lower_tr(s[-1]) in {"ç","f","h","k","p","s","ş","t"} else "d"

def choose_initial_C(s: str) -> str:
    return "ç" if s and lower_tr(s[-1]) in {"ç","f","h","k","p","s","ş","t"} else "c"

def strip_infinitive_from_ottoman(w: str) -> str:
    for sfx in ("مق","مك","ماق","مەك","mak","mek"):
        if w.endswith(sfx):
            return w[:-len(sfx)]
    return w

def normalize_surface_ascii(t: str) -> str:
    t = lower_tr(t)
    return t.replace("\u2019","'").replace("\u2018","'").replace("'","")

# ============================================================
# §4  MORPHOPHONEMIC REPRESENTATIONS
# ============================================================

UNDERLYING_MORPHS: dict[str, str] = {
    "PASSIVE":"Il","CAUSATIVE":"DIr","RECIPROCAL":"Iş","REFLEXIVE":"In",
    "ABLE":"(y)Abil","UNABLE":"mA","NEG":"mA","INF1":"mAk","INF2":"mA",
    "PART":"","CONV":"","CONV_SINCE":"(y)AlI",
    "PAST":"DI","NARR":"mIş","PROG":"Iyor","FUTURE":"(y)AcAk",
    "FUT_PART":"(y)AcAk","AOR":"Ar","COND":"sA","NECES":"mAlI","OPT":"(y)A",
    "A1SG":"Im","A2SG":"sIn","A3SG":"","A1PL":"Iz","A2PL":"sInIz","A3PL":"lAr",
    "PLURAL":"lAr",
    "P1SG":"(I)m","P2SG":"(I)n","P3SG":"sI","P1PL":"(I)mIz","P2PL":"(I)nIz","P3PL":"lArI",
    "NOM":"","ACC":"(y)I","DAT":"(y)A","LOC":"DA","ABL":"DAn",
    "GEN":"(n)In","INS":"(y)lA","REL":"ki","REL_LOC":"DAki",
    "COPULA":"i","COPULA_ASSERT":"DIr",
    "NOM_DER_LIK":"lIk","NOM_DER_LI":"lI","NOM_DER_SIZ":"sIz",
    "NOM_DER_SEL":"sAl","NOM_DER_CI":"CI","NOM_DER_DAS":"DAş","NOM_DER_MSI":"ImsI",
}

OTTOMAN_SURFACE_OVERRIDES: dict[str, str] = {
    "lar":"لر","ler":"لر",
    "ları":"لری","leri":"لری",
    "lara":"لره","lere":"لره","lardan":"لردن","lerden":"لردن",
    "larla":"لرله","lerle":"لرله","larda":"لرده","lerde":"لرده",
    "ların":"لرڭ","lerin":"لرڭ",
    "larını":"لرینی","lerini":"لرینی","larına":"لرینه","lerine":"لرینه",
    "larında":"لریندە","lerinde":"لریندە","larından":"لریندن","lerinden":"لریندن",
    "larının":"لرینڭ","lerinin":"لرینڭ",
    "ımız":"مز","imiz":"مز","umuz":"مز","ümüz":"مز",
    "mız":"مز","miz":"مز","muz":"مز","müz":"مز",
    "ım":"م","im":"م","um":"م","üm":"م","m":"م",
    "ın":"ڭ","in":"ڭ","un":"ڭ","ün":"ڭ",
    "nız":"ڭز","niz":"ڭز","nuz":"ڭز","nüz":"ڭز",
    "ınız":"ڭز","iniz":"ڭز","unuz":"ڭز","ünüz":"ڭز",
    "nın":"نڭ","nin":"نڭ","nun":"نڭ","nün":"نڭ",
    "sı":"سی","si":"سی","su":"سی","sü":"سی",
    "ı":"ی","i":"ی","u":"ی","ü":"ی",
    "yı":"یی","yi":"یی","yu":"یو","yü":"یو",
    "a":"ه","e":"ه","ya":"یه","ye":"یه",
    "da":"ده","de":"ده","ta":"ده","te":"ده",
    "dan":"دن","den":"دن","tan":"دن","ten":"دن",
    "la":"له","le":"له","yla":"یله","yle":"یله",
    "daki":"دەكی","deki":"دەكی","taki":"دەكی","teki":"دەكی",
    "ndaki":"ندەكی","ndeki":"ندەكی","ki":"كی",
    "dı":"دی","di":"دی","du":"دی","dü":"دی",
    "tı":"دی","ti":"دی","tu":"دی","tü":"دی",
    "tır":"در","tir":"در","tur":"در","tür":"در",
    "dır":"در","dir":"در","dur":"در","dür":"در",
    "mış":"مش","miş":"مش","muş":"مش","müş":"مش",
    "mamış":"ممش","memiş":"ممش",
    "acak":"اجق","ecek":"هجك","yacak":"یاجق","yecek":"یهجك",
    "acağ":"اجغ","eceğ":"هجگ","yacağ":"یاجغ","yeceğ":"یهجگ",
    "sın":"سڭ","sin":"سڭ","sun":"سڭ","sün":"سڭ",
    "sınız":"سڭز","siniz":"سڭز","sunuz":"سڭز","sünüz":"سڭز",
    "ız":"ز","iz":"ز","uz":"ز","üz":"ز",
    "yız":"یز","yiz":"یز","yuz":"یوز","yüz":"یوز",
    "r":"ر","ar":"ار","er":"ر",
    "sa":"سە","se":"سە","malı":"ملی","meli":"ملی",
    "yor":"یور","ıyor":"یور","iyor":"یور","uyor":"یور","üyor":"یور",
    "iken":"ایکن","yken":"ایکن","ise":"ایسه","ysa":"ایسه","yse":"ایسه",
    "idi":"ایدی","ydı":"ایدی","ydi":"ایدی","ydu":"ایدی","ydü":"ایدی",
    "imiş":"ایمش","ymış":"ایمش","ymiş":"ایمش",
    "ydım":"ایدم","ydim":"ایدم","ydın":"ایدڭ","ydin":"ایدڭ",
    "ydık":"ایدق","ydik":"ایدک","ydınız":"ایدڭز","ydiniz":"ایدڭز",
    "ydılar":"ایدلر","ydiler":"ایدلر",
    "iyordu":"ایوردی","iyormuş":"ایورمش","iyorsa":"ایورسه",
    "ıyordu":"ایوردی","ıyormuş":"ایورمش","ıyorsa":"ایورسه",
    "ndan":"ندن","nden":"ندن","nda":"نده","nde":"نده","na":"نه","ne":"نه",
    "mayalı":"مایالی","meyeli":"میەلی","mamak":"مەمق","memek":"مەمك",
}

VOWEL_DROP_WORDS = {
    "burun","ağız","karın","oğul","gönül","omuz",
    "akıl","şehir","nehir","sabır","ömür",
}

WORD_OVERRIDES: dict[str, str] = {
    "Apple":"آپپلە","başka":"باشقە","da":"دە","de":"دە",
    "değişikliğe":"دگیشیكلگە","ile":"ایلە","lise":"لیسه",
    "Messi":"مسّی","elektrik":"الكتریگ",
}

# ============================================================
# §5  MORPHOPHONEMIC RULES
# ============================================================

def apply_vowel_harmony(morph: str, root: str) -> str:
    return morph.replace("A", choose_harmony_A(root)).replace("I", choose_harmony_I(root))

def apply_vowel_drop(root: str, morph: str) -> str:
    if lower_tr(root) not in VOWEL_DROP_WORDS:
        return root
    if not morph or not starts_with_vowel(morph):
        return root
    return root[:-2] + root[-1] if len(root) >= 3 else root

def apply_consonant_softening(root: str, morph: str) -> str:
    if not morph or not starts_with_vowel(morph):
        return root
    lr = lower_tr(root)
    if lr.endswith("nk"):
        return root[:-1] + "g"
    rep = {"p":"b","ç":"c","t":"d","k":"ğ"}
    last = lr[-1] if lr else ""
    return root[:-1] + rep[last] if last in rep else root

def apply_buffer_consonants(prev: str, morph: str) -> str:
    for token, char in [("(y)","y"),("(n)","n"),("(s)","s"),("(ş)","ş")]:
        if morph.startswith(token):
            buf = char if ends_with_vowel(prev) else ""
            return buf + morph[3:]
    return morph

def resolve_nominal_possessive(tag: str, prev: str) -> str:
    hi, ha, ev = choose_harmony_I(prev), choose_harmony_A(prev), ends_with_vowel(prev)
    if tag == "P1SG": return "m"           if ev else hi + "m"
    if tag == "P2SG": return "n"           if ev else hi + "n"
    if tag == "P3SG": return "s" + hi      if ev else hi
    if tag == "P1PL": return "m"+hi+"z"    if ev else hi+"m"+hi+"z"
    if tag == "P2PL": return "n"+hi+"z"    if ev else hi+"n"+hi+"z"
    if tag == "P3PL": return "l"+ha+"r"+choose_harmony_I(prev)
    return ""

def resolve_nominal_case(tag: str, prev: str, possessed: bool = False) -> str:
    hi, ha, ev = choose_harmony_I(prev), choose_harmony_A(prev), ends_with_vowel(prev)
    if tag == "NOM":    return ""
    if tag == "ACC":    return ("n"+hi)   if possessed else (("y" if ev else "")+hi)
    if tag == "DAT":    return ("n"+ha)   if possessed else (("y" if ev else "")+ha)
    if tag == "LOC":    return choose_initial_D(prev) + ha
    if tag == "ABL":    return choose_initial_D(prev) + ha + "n"
    if tag == "GEN":    return (("n" if ev or possessed else "") + hi + "n")
    if tag == "INS":    return ("y" if ev else "") + "l" + ha
    if tag == "REL":    return "ki"
    if tag == "REL_LOC":return choose_initial_D(prev) + ha + "ki"
    return ""

def resolve_copula_variant(follow_tag: str, prev: str) -> str:
    buf = "y" if ends_with_vowel(prev) else ""
    if follow_tag == "PAST":  return buf + choose_initial_D(prev) + choose_harmony_I(prev)
    if follow_tag == "NARR":  return buf + "m" + choose_harmony_I(prev) + "ş"
    if follow_tag == "COND":  return buf + "s" + choose_harmony_A(prev)
    if follow_tag == "NECES": return (buf or "i") + "d" + choose_harmony_I(prev) + "r"
    return ""

def fuse_tag_strings(tags: list[str]) -> list[str]:
    out, i = [], 0
    while i < len(tags):
        if tags[i:i+2] == ["LOC","REL"]:
            out.append("REL_LOC"); i += 2
        else:
            out.append(tags[i]); i += 1
    return out

def fuse_realized_morphs(seq: list[dict]) -> list[dict]:
    out, i = [], 0
    while i < len(seq):
        item = dict(seq[i])
        if out:
            prev = out[-1]
            if prev["tag"] == "NEG" and item["tag"] in {"CONV_SINCE","NARR","INF1"}:
                prev["tag"] = item["tag"]
                prev["surface"] += item["surface"]
                i += 1; continue
            if (prev["tag"] in {"FUTURE","FUT_PART"}
                    and item["surface"][:1] in _TUM_UNLULER
                    and prev["surface"].endswith(("acak","ecek","yacak","yecek"))):
                prev["surface"] = prev["surface"][:-1] + "ğ"
            if prev["tag"] == "REL" and item["tag"] == "DAT":
                prev["surface"] = prev["surface"][:-2] + "ye"; i += 1; continue
        out.append(item); i += 1
    return out

def build_underlying_morphs(tags: list[str]) -> list[dict]:
    out = []
    for tag in tags:
        if tag in ROOT_POS_TAGS or tag in EMPTY_TAGS:
            continue
        u = UNDERLYING_MORPHS.get(tag)
        if u is not None:
            out.append({"tag": tag, "underlying": u})
    return out

def realize_single_morph(
    prev: str, morph: dict,
    next_tag: Optional[str] = None,
    copula_mode: bool = False,
    possessed: bool = False,
) -> Optional[str]:
    tag, underlying = morph["tag"], morph["underlying"]
    if tag in POSSESSIVE_TAGS: return resolve_nominal_possessive(tag, prev)
    if tag in CASE_TAGS:       return resolve_nominal_case(tag, prev, possessed=possessed)
    if tag == "PLURAL":        return "l" + choose_harmony_A(prev) + "r"
    if tag == "A1SG":          return choose_harmony_I(prev) + "m"
    if tag == "A1PL":          return choose_harmony_I(prev) + "z"
    if tag == "A2SG":          return "s" + choose_harmony_I(prev) + "n"
    if tag == "A2PL":
        hi = choose_harmony_I(prev); return "s"+hi+"n"+hi+"z"
    if tag == "A3PL":          return "l" + choose_harmony_A(prev) + "r"
    if copula_mode and tag in {"PAST","NARR","COND","NECES"}:
        return resolve_copula_variant(tag, prev)
    if tag == "PAST":    return choose_initial_D(prev) + choose_harmony_I(prev)
    if tag == "NARR":    return "m" + choose_harmony_I(prev) + "ş"
    if tag == "PROG":    return choose_harmony_I(prev) + "yor"
    if tag in {"FUTURE","FUT_PART"}:
        ha = choose_harmony_A(prev)
        return ("y" if ends_with_vowel(prev) else "") + ha + "c" + ha + "k"
    if tag == "AOR":
        return "r" if ends_with_vowel(prev) else choose_harmony_A(prev) + "r"
    if tag == "COND":  return "s" + choose_harmony_A(prev)
    if tag == "NECES": return "m" + choose_harmony_A(prev) + "l" + choose_harmony_I(prev)
    if tag == "OPT":   return ("y" if ends_with_vowel(prev) else "") + choose_harmony_A(prev)
    if tag == "NEG":   return "m" + choose_harmony_A(prev)
    if tag == "UNABLE":return "m" + choose_harmony_A(prev)
    if tag == "ABLE":  return ("y" if ends_with_vowel(prev) else "") + choose_harmony_A(prev) + "bil"
    if tag == "PASSIVE":    return choose_harmony_I(prev) + "l"
    if tag == "CAUSATIVE":  return choose_initial_D(prev) + choose_harmony_I(prev) + "r"
    if tag == "RECIPROCAL": return choose_harmony_I(prev) + "ş"
    if tag == "REFLEXIVE":  return choose_harmony_I(prev) + "n"
    if tag in {"PART","CONV"}: return ""
    if tag == "INF1": return "m" + choose_harmony_A(prev) + "k"
    if tag == "INF2": return "m" + choose_harmony_A(prev)
    if tag == "NOM_DER_LIK": return "l" + choose_harmony_I(prev) + "k"
    if tag == "NOM_DER_LI":  return "l" + choose_harmony_I(prev)
    if tag == "NOM_DER_SIZ": return "s" + choose_harmony_I(prev) + "z"
    if tag == "NOM_DER_SEL": return "s" + choose_harmony_A(prev) + "l"
    if tag == "NOM_DER_CI":  return choose_initial_C(prev) + choose_harmony_I(prev)
    if tag == "NOM_DER_DAS": return choose_initial_D(prev) + choose_harmony_A(prev) + "ş"
    if tag == "NOM_DER_MSI": return choose_harmony_I(prev) + "ms" + choose_harmony_I(prev)
    realized = apply_vowel_harmony(underlying, prev)
    return apply_buffer_consonants(prev, realized)

def realize_allomorphs(root_surface: str, morphs: list[dict]) -> tuple[str, list[dict]]:
    current_root = lower_tr(root_surface)
    realized: list[dict] = []
    copula_mode = False
    has_possessive = False
    for idx, morph in enumerate(morphs):
        tag      = morph["tag"]
        next_tag = morphs[idx + 1]["tag"] if idx + 1 < len(morphs) else None
        if tag == "COPULA":
            copula_mode = True; continue
        prev = current_root + "".join(p["surface"] for p in realized)
        piece = realize_single_morph(prev, morph, next_tag, copula_mode, has_possessive)
        if piece is None:
            continue
        if not realized and starts_with_vowel(piece):
            current_root = apply_vowel_drop(current_root, piece)
            if tag not in {"FUTURE","FUT_PART"}:
                current_root = apply_consonant_softening(current_root, piece)
            prev  = current_root + "".join(p["surface"] for p in realized)
            piece = realize_single_morph(prev, morph, next_tag, copula_mode, has_possessive)
        piece = normalize_surface_ascii(piece)
        realized.append({"tag": tag, "surface": piece})
        if tag in POSSESSIVE_TAGS:
            has_possessive = True
    return current_root, fuse_realized_morphs(realized)

# ============================================================
# §6  OTTOMAN RENDER LAYER
# ============================================================

def render_ottoman(surface: str, historical: bool = False) -> str:
    if not surface:
        return ""
    normalized = normalize_surface_ascii(surface)
    if historical and normalized in OTTOMAN_SURFACE_OVERRIDES:
        return OTTOMAN_SURFACE_OVERRIDES[normalized]
    result = ""
    skip_first = False
    if normalized and normalized[0] in _TUM_UNLULER:
        result += "ا"
        skip_first = True  # leading alef written; skip first char in loop
    first_v = next((c for c in normalized if c in _TUM_UNLULER), "a")
    current_harmony = "kalin" if first_v in KALIN_UNLULER else "ince"
    for idx, ch in enumerate(normalized):
        if ch in _TUM_UNLULER:
            current_harmony = "kalin" if ch in KALIN_UNLULER else "ince"
        if skip_first and idx == 0:
            continue  # alef already written above
        if ch == "k":
            result += "ق" if current_harmony == "kalin" else "ک"
        elif ch == "t":
            result += "ط" if current_harmony == "kalin" else "ت"
        elif ch == "s":
            result += "ص" if current_harmony == "kalin" else "س"
        else:
            result += PHONEME_MAP.get(ch, ch)
    return result

def merge_ottoman(root_ot: str, suffixes: list[dict]) -> str:
    rendered    = [p["ottoman"] for p in suffixes if p["ottoman"]]
    visible_tags= [p["tag"]    for p in suffixes if p["ottoman"]]
    first_vis   = rendered[0] if rendered else ""
    if first_vis.startswith("لر") and root_ot.endswith("ه"):
        root_ot = root_ot[:-1] + "ە"
    if visible_tags[:1] == ["REL_LOC"] and root_ot.endswith("ه"):
        root_ot = root_ot[:-1] + "ە"
    if (visible_tags[:2] == ["P3SG","ACC"] and rendered[:2] == ["سی","نی"]):
        if root_ot.endswith("ه"): root_ot = root_ot[:-1] + "ە"
        rendered = ["سنی"] + rendered[2:]
    if first_vis.startswith("ی") and root_ot.endswith("ه"):
        root_ot = root_ot[:-1] + "ە"
    return root_ot + "".join(rendered)

# ============================================================
# §7  ENGLISH TRANSLITERATION
# ============================================================

_ENG_VOWELS = set("aeiou")

_ENG_PATTERNS: list[tuple[str, str]] = [
    ("tion","شن"),("sion","شن"),("ture","چر"),("ough","و"),("augh","اف"),("ight","ایت"),
    ("tch","چ"),("dge","ج"),("sch","ش"),("ght","ت"),("gue","گ"),("que","ک"),
    ("igh","ای"),("ssi","ش"),("sci","ش"),("kno","نو"),("wri","ری"),("psy","سی"),
    ("pneu","نو"),("rhy","ری"),
    ("ph","ف"),("sh","ش"),("ch","چ"),("th","ث"),("wh","و"),("ck","ک"),
    ("ng","ڭ"),("nk","ڭک"),("qu","کو"),("kn","ن"),("wr","ر"),("gn","ن"),
    ("ps","س"),("mb","م"),("gh",""),("ee","ی"),("ea","ی"),("oo","و"),
    ("ou","او"),("ow","او"),("oa","و"),("ai","ای"),("ay","ای"),("ei","ی"),
    ("ie","ی"),("oi","وی"),("oy","وی"),("au","او"),("aw","او"),("ew","یو"),
    ("ue","یو"),("ae","ی"),("oe","و"),("ui","وی"),("ia","یه"),("io","یو"),
    ("ua","وه"),("lk","لک"),("lm","لم"),("mn","م"),("rh","ر"),("xc","کس"),
    ("ww","و"),("ss","س"),("ll","ل"),("tt","ت"),("nn","ن"),("rr","ر"),
    ("pp","پ"),("bb","ب"),("dd","د"),("ff","ف"),("gg","گ"),("cc","ک"),
    ("mm","م"),("zz","ز"),
]

_ENG_SINGLE: dict[str, str] = {
    "a":"ه","b":"ب","c":"ک","d":"د","e":"ه","f":"ف","g":"گ","h":"ه",
    "i":"ی","j":"ج","k":"ک","l":"ل","m":"م","n":"ن","o":"و","p":"پ",
    "q":"ق","r":"ر","s":"س","t":"ت","u":"ا","v":"ڤ","w":"و","x":"کس",
    "y":"ی","z":"ز",
}

ENGLISH_WORD_OVERRIDES: dict[str, str] = {
    "internet":"اینترنت","computer":"کمپیوتر","digital":"دیجیتل",
    "software":"سافتویر","hardware":"هاردویر","network":"نتورک",
    "telephone":"تلفون","television":"تلویزیون","radio":"رادیو",
    "video":"ویدیو","photo":"فوطو","camera":"کامره","email":"ایمیل",
    "website":"ویب سایت","password":"پاسورد","download":"داونلود",
    "upload":"آپلود","london":"لندره","paris":"پاریس","berlin":"برلین",
    "moscow":"مسکو","new york":"نیویورک","ok":"اوکی","okay":"اوکی",
    "yes":"یس","no":"نو","hello":"هللو","bye":"بای",
    "english":"انگیلیزچه","french":"فرانسزچه","german":"الماندجه",
    "america":"امریقا","europe":"اوروپا","asia":"آسیا","africa":"افریقا",
}

def is_likely_english(word: str) -> bool:
    w = word.lower()
    if any(c in w for c in "ğüşıöç"): return False
    if any(p in w for p in ("wh","ph","tch","tion","ght","ough","kn","wr")): return True
    if "w" in w or "x" in w: return True
    for sfx in ("tion","sion","ness","ment","ful","less","ive","ous","ing","ance",
                "ence","able","ible","ity","ify","ize","ise","ism","ist","ish","ward"):
        if w.endswith(sfx) and len(w) > len(sfx) + 1: return True
    return False

def render_english_ottoman(word: str) -> str:
    wl = word.lower()
    if wl in ENGLISH_WORD_OVERRIDES:
        return ENGLISH_WORD_OVERRIDES[wl]
    s, n, out, i = wl, len(wl), "", 0
    while i < n:
        matched = False
        for pattern, ottoman in _ENG_PATTERNS:
            pl = len(pattern)
            if i + pl <= n and s[i:i+pl] == pattern:
                if pattern == "gh" and i == 0:           ottoman = "غ"
                elif pattern == "mb" and i + pl < n:     ottoman = "مب"
                elif pattern == "ou" and i+pl<n and s[i+pl] in "lr": ottoman = "ور"
                out += ottoman; i += pl; matched = True; break
        if matched: continue
        ch = s[i]
        if ch == "c":
            out += "س" if i+1<n and s[i+1] in "eiy" else "ک"
        elif ch == "g":
            out += "ج" if i+1<n and s[i+1] in "eiy" else "گ"
        elif ch == "e" and i == n-1 and i > 0 and s[i-1] not in _ENG_VOWELS:
            pass
        elif ch == "a" and i+2<n and s[i+1] not in _ENG_VOWELS and s[i+2]=="e" and i+3>=n:
            out += "ای"
        elif ch == "i" and i+2<n and s[i+1] not in _ENG_VOWELS and s[i+2]=="e" and i+3>=n:
            out += "ای"
        else:
            out += _ENG_SINGLE.get(ch, ch)
        i += 1
    if out and out[0] in {"ه","ی","و"}: out = "ا" + out
    return out

# ============================================================
# §8  ZEYREK ANALYSIS
# ============================================================

ZEYREK_TAG_MAP: dict[str, str] = {
    "Passive":"PASSIVE","Caus":"CAUSATIVE","Recip":"RECIPROCAL","Reflex":"REFLEXIVE",
    "Able":"ABLE","Unable":"UNABLE","Neg":"NEG","Inf1":"INF1","Inf2":"INF2",
    "Part":"PART","FutPart":"FUT_PART","Conv":"CONV","SinceDoingSo":"CONV_SINCE",
    "Past":"PAST","Narr":"NARR","Prog1":"PROG","Prog2":"PROG","Fut":"FUTURE",
    "Aor":"AOR","Cond":"COND","Neces":"NECES","Opt":"OPT","Cop":"COPULA_ASSERT",
    "A1sg":"A1SG","A2sg":"A2SG","A3sg":"A3SG","A1pl":"A1PL","A2pl":"A2PL","A3pl":"A3PL",
    "P1sg":"P1SG","P2sg":"P2SG","P3sg":"P3SG","P1pl":"P1PL","P2pl":"P2PL","P3pl":"P3PL",
    "Nom":"NOM","Acc":"ACC","Dat":"DAT","Loc":"LOC","Abl":"ABL","Gen":"GEN","Ins":"INS",
    "Rel":"REL","With":"NOM_DER_LI","Without":"NOM_DER_SIZ","Agt":"NOM_DER_CI",
    "FitFor":"NOM_DER_LIK",
}

_APOSTROPHE_SUFFIXES: set[str] = {
    "m","n","nın","nin","nun","nün",
    "mız","miz","muz","müz","nız","niz","nuz","nüz",
    "ım","im","um","üm","ın","in","un","ün",
    "ımız","imiz","umuz","ümüz","ınız","iniz","unuz","ünüz",
    "sı","si","su","sü","ları","leri",
    "ı","i","u","ü","yı","yi","yu","yü",
    "a","e","ya","ye","da","de","ta","te",
    "dan","den","tan","ten","la","le","yla","yle",
    "dır","dir","dur","dür","tır","tir","tur","tür",
    "ydı","ydi","ydu","ydü","ydım","ydim","ydın","ydin",
    "ydık","ydik","ydınız","ydiniz","ymış","ymiş",
    "ysa","yse","lar","ler","ların","lerin","lara","lere",
    "lardan","lerden","larla","lerle","larda","lerde",
}


class OttomanTransliterator:
    """Thread-safe transliteration engine.

    Instantiate once at application startup; the underlying Zeyrek
    analyzer is loaded on __init__.
    """

    def __init__(
        self,
        lookup_file: str = "manual_lookup.tsv",
        abbrev_file: str = "abbrev_lookup.tsv",
        historical: bool = True,
    ) -> None:
        self.historical = historical
        self._analyzer  = zeyrek.MorphAnalyzer()
        self._lookup    = self._load_tsv(lookup_file)
        self._abbrev    = self._load_tsv(abbrev_file)
        self._lookup_folded: dict[str, str] = {}
        for k, v in self._lookup.items():
            f = fold_tr(k)
            if f and f not in self._lookup_folded:
                self._lookup_folded[f] = v

    # ── public ────────────────────────────────────────────────────────────

    def transliterate(self, text: str) -> "TransliterationResult":
        tokens: list[dict] = []
        ot_parts: list[str] = []
        sources:  list[str] = []

        for tok, ot, src, dbg in self._tokenize(text):
            ot_parts.append(ot)
            if src != "whitespace":
                tokens.append({
                    "token":   tok,
                    "ottoman": ot,
                    "source":  src,
                    "debug":   dbg,
                })
                sources.append(src)

        ottoman   = "".join(ot_parts)
        scores    = [SCORE_MAP.get(s, 0.0) for s in sources]
        confidence = round(sum(scores) / len(scores), 3) if scores else 1.0

        return TransliterationResult(
            turkish    = text,
            ottoman    = ottoman,
            confidence = confidence,
            tokens     = tokens,
        )

    # ── tokenization ──────────────────────────────────────────────────────

    def _tokenize(self, line: str):
        cursor = 0
        for match in _TOKEN_RE.finditer(line):
            start, end = match.span()
            if start > cursor:
                gap = line[cursor:start]
                yield gap, gap, "whitespace", gap
            token = match.group(0)
            if not is_word_token(token):
                ot = convert_ottoman_punctuation(token)
                yield token, ot, "punct", token
            else:
                ot, src, dbg = self._transliterate_token(token)
                yield token, ot, src, dbg
            cursor = end
        if cursor < len(line):
            tail = line[cursor:]
            yield tail, tail, "whitespace", tail

    def _transliterate_token(self, token: str) -> tuple[str, str, str]:
        # 1. Hard override
        if token in WORD_OVERRIDES:
            return WORD_OVERRIDES[token], "override", token
        lw = lower_tr(token)
        if lw in WORD_OVERRIDES:
            return WORD_OVERRIDES[lw], "override", token

        # 2. Apostrophe split
        norm = token.replace("\u2019","'").replace("\u2018","'")
        if "'" in norm:
            base, suffix = norm.split("'", 1)
            found = self._lookup_root_entry(base) or self._lookup_root_entry(lower_tr(base))
            if found and lower_tr(suffix) in _APOSTROPHE_SUFFIXES:
                base_ot = found[0]
                suf_ot  = OTTOMAN_SURFACE_OVERRIDES.get(lower_tr(suffix)) or \
                          render_ottoman(lower_tr(suffix), self.historical)
                return base_ot + suf_ot, "override", token

        # 3. Dictionary / number
        direct, src, form = self._lookup_word(token)
        if direct is not None:
            return direct, src, form

        # 4. English
        if is_likely_english(token):
            return render_english_ottoman(token), "english", token

        # 5. Zeyrek morphological analysis
        analysis_word = token.replace("'","").replace("\u2019","")
        selected = self._select_parse(analysis_word)
        if selected:
            pred = self._predicative_inf(selected, analysis_word)
            if pred:
                return pred["result"], "tags", f"{selected['lemma']} :: PRED_INF"
            gen = self._generate(selected["root_ot"], selected["tags"], selected["surface_root"])
            if gen:
                sfx = " + ".join(p["surface"] for p in gen["suffixes"])
                dbg = f"{selected['lemma']} :: {'+'.join(selected['tags'])} :: {selected['surface_root']}+{sfx}"
                return gen["result"], "tags", dbg

        # 6. Auto fallback
        parses = self._flatten(analysis_word)
        if parses:
            pos = str(getattr(parses[0], "pos", ""))
            if not any(x in pos for x in ("Prop","Abbrv","Unk")):
                return render_ottoman(token, False), "auto", token

        # 7. Unresolved
        return f"[{token}]", "missing", token

    # ── dictionary helpers ────────────────────────────────────────────────

    @staticmethod
    def _load_tsv(filepath: str) -> dict[str, str]:
        data: dict[str, str] = {}
        if not os.path.exists(filepath):
            return data
        with open(filepath, encoding="utf-8-sig") as f:
            for row in csv.reader(f, delimiter="\t"):
                if len(row) >= 2 and not row[0].startswith("#"):
                    k, v = row[0].strip(), row[1].strip()
                    if k.lower() == "word" and v.lower() == "ottoman":
                        continue
                    data[k] = v
                    data[lower_tr(k)] = v
        return data

    def _lookup_root_entry(self, key: str):
        if not key: return None
        for cand, score in [(key,32),(lower_tr(key),28)]:
            if cand in WORD_OVERRIDES: return WORD_OVERRIDES[cand], cand, score
        for cand, score in [(key,30),(lower_tr(key),24)]:
            if cand in self._lookup: return self._lookup[cand], cand, score
        f = fold_tr(key)
        if f in self._lookup_folded: return self._lookup_folded[f], lower_tr(key), 18
        return None

    def _lookup_word(self, word: str):
        if all(c.isdigit() or c in ".,-" for c in word) and any(c.isdigit() for c in word):
            return word.translate(str.maketrans("0123456789","٠١٢٣٤٥٦٧٨٩")), "exact", word
        if word in self._lookup: return self._lookup[word], "exact", word
        lw = lower_tr(word)
        if lw in self._lookup:   return self._lookup[lw],  "exact", lw
        f = fold_tr(word)
        if f in self._lookup_folded: return self._lookup_folded[f], "exact", lw
        return None, None, None

    # ── Zeyrek wrappers ───────────────────────────────────────────────────

    def _analyze(self, word: str):
        return _cached_zeyrek_analyze(normalize_tr_text(word))

    def _flatten(self, word: str) -> list:
        return [p for grp in self._analyze(word) for p in grp]

    def _parse_pos(self, parse) -> str:
        fmt   = getattr(parse, "formatted", "") or ""
        m     = re.match(r"\[([^:]+):([^\],]+)", fmt)
        return m.group(2).strip().upper() if m else str(getattr(parse,"pos","")).upper()

    def _surface_root(self, parse) -> str:
        fmt = getattr(parse,"formatted","") or ""
        if "] " in fmt:
            tail = fmt.split("] ",1)[1]
            seg  = re.split(r"[+|]",tail)[0]
            if ":" in seg: return lower_tr(seg.split(":",1)[0].strip())
        lemma = lower_tr(getattr(parse,"lemma","") or "")
        return lemma[:-3] if lemma.endswith(("mak","mek")) else lemma

    def _resolve_root(self, parse):
        lemma = getattr(parse,"lemma",None)
        if not lemma or lemma == "Unk": return None
        base_pos = self._parse_pos(parse)
        sr       = self._surface_root(parse)
        cands    = []
        if base_pos == "VERB" and sr: cands.append(sr)
        cands.extend([lemma, lower_tr(lemma)])
        if base_pos == "VERB" and lower_tr(lemma).endswith(("mak","mek")):
            cands.append(lower_tr(lemma)[:-3])
        if sr and sr not in cands: cands.append(sr)
        seen: set = set()
        for cand in cands:
            if not cand or cand in seen: continue
            seen.add(cand)
            found = self._lookup_root_entry(cand)
            if found:
                root_ot, dict_form, ls = found
                return {"root_ot":root_ot,"lemma":lemma,"dict_form":dict_form,
                        "lookup_score":ls,"base_pos":base_pos,"surface_root":sr or lower_tr(cand)}
        return None

    def _zeyrek_tags(self, parse) -> list[str]:
        base_pos    = self._parse_pos(parse)
        morphemes   = list(getattr(parse,"morphemes",[]) or [])
        tags        = [base_pos]
        nominal_ctx = base_pos in {"NOUN","ADJ","PRON","NUM","QUES"}
        verbal_ctx  = base_pos == "VERB"
        for raw in morphemes[1:]:
            if raw == "Zero": continue
            if raw in {"Noun","Adj","Adv","Pron","Num"}:
                nominal_ctx = True; verbal_ctx = False; continue
            if raw == "Verb":
                if not verbal_ctx: tags.append("COPULA")
                verbal_ctx = True; nominal_ctx = False; continue
            if raw == "A3pl":
                tags.append("PLURAL" if nominal_ctx else "A3PL"); continue
            if raw == "A3sg" and not nominal_ctx:
                tags.append("A3SG"); continue
            n = ZEYREK_TAG_MAP.get(raw)
            if n: tags.append(n)
        return fuse_tag_strings(tags)

    def _score(self, parse, ntags: list[str], ri: dict, amb: int) -> float:
        score = float(ri["lookup_score"])
        score += 40 if self._validate_tags(ntags) else -80
        if ri["base_pos"] == str(getattr(parse,"pos","")).upper(): score += 10
        if "PROP" in (getattr(parse,"formatted","") or "").upper(): score -= 20
        deriv = sum(1 for t in ntags if t in VERBAL_DERIVATION_TAGS or t in NOMINAL_DERIVATION_TAGS)
        non_d = [t for t in ntags if t not in VERBAL_DERIVATION_TAGS and t not in NOMINAL_DERIVATION_TAGS]
        score -= deriv * 4
        score -= max(len(non_d) - 3, 0)
        score -= max(amb - 1, 0) * 2
        if ri["dict_form"] == lower_tr(getattr(parse,"lemma","") or ""): score += 6
        return score

    def _validate_tags(self, tags: list[str]) -> bool:
        root_pos = tags[0] if tags else ROOT
        current  = ROOT
        for tag in tags:
            if tag in ROOT_POS_TAGS: continue
            target = TAG_TO_STATE.get(tag)
            if target is None: return False
            if target != current and target not in TAG_FSM.get(current, []):
                return False
            current = target
        return True

    def _select_parse(self, word: str):
        parses = self._flatten(word)
        cands  = []
        for parse in parses:
            ri = self._resolve_root(parse)
            if not ri: continue
            nt = self._zeyrek_tags(parse)
            cands.append((self._score(parse, nt, ri, len(parses)), parse, ri, nt))
        if not cands: return None
        cands.sort(key=lambda x: (x[0], -len(x[3]), len(x[2]["surface_root"])), reverse=True)
        _, bp, ri, tags = cands[0]
        return {"parse":bp,"root_ot":ri["root_ot"],"lemma":ri["lemma"],
                "dict_form":ri["dict_form"],"base_pos":ri["base_pos"],
                "surface_root":ri["surface_root"],"tags":tags}

    def _generate(self, root_ot: str, tags: list[str], root_surface: str):
        ntags = fuse_tag_strings(tags)
        if not self._validate_tags(ntags): return None
        morphs = build_underlying_morphs(ntags)
        realized_root, realized_sfx = realize_allomorphs(root_surface, morphs)
        rendered_sfx = [
            {"tag":p["tag"],"surface":p["surface"],
             "ottoman": render_ottoman(p["surface"], self.historical)}
            for p in realized_sfx
        ]
        if ntags and ntags[0] == "VERB" and rendered_sfx:
            stripped = strip_infinitive_from_ottoman(root_ot)
            root_ot  = (WORD_OVERRIDES.get(realized_root) or
                        render_ottoman(realized_root, self.historical)
                        if normalize_surface_ascii(realized_root) != normalize_surface_ascii(root_surface)
                        else stripped)
        return {"surface_root":realized_root,"suffixes":rendered_sfx,
                "result": merge_ottoman(root_ot, rendered_sfx)}

    def _predicative_inf(self, sel: dict, word: str):
        morphemes = list(getattr(sel["parse"],"morphemes",[]) or [])
        if not all(x in morphemes for x in ("Inf1","Pres","Cop")): return None
        lemma = lower_tr(sel.get("lemma") or "")
        w     = lower_tr(word or "")
        if not lemma or not w.startswith(lemma): return None
        cop_surface = w[len(lemma):]
        if not cop_surface: return None
        cop_ot = render_ottoman(cop_surface, self.historical)
        return {"result": sel["root_ot"] + cop_ot}


# ============================================================
# §9  RESULT MODEL
# ============================================================

@dataclass
class TransliterationResult:
    turkish:    str
    ottoman:    str
    confidence: float
    tokens:     list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "turkish":    self.turkish,
            "ottoman":    self.ottoman,
            "confidence": self.confidence,
            "tokens":     self.tokens,
        }
