"""
Ottoman Turkish Transliterator  v2.0.3
======================================
Synced from: Claude_2may_koklu_corpus_maker_improved.ipynb
API-only changes kept:
  - punkt bypass patch
  - module-level LRU cache (instance lru_cache doesn't persist)
  - skip_first fix in render_ottoman (prevents double-alef)
  - OttomanTransliterator class wrapper
  - "auto" fallback instead of "[token]" brackets
"""
from __future__ import annotations
import csv, logging, os, re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import zeyrek
import zeyrek.morphology as _zm

# ── Bypass nltk punkt_tab (single tokens only, punkt unused) ─────────────────
def _simple_tokenize(text: str) -> list[str]:
    return [text.strip()] if text.strip() else []
_zm._tokenize_text = _simple_tokenize
logging.getLogger("zeyrek").setLevel(logging.ERROR)
def normalize_ottoman_lookup_value(text: str) -> str:
    """Normalise Ottoman script: unify glyph variants, strip thin spaces."""
    if not text: return text
    t = text.replace("ي", "ی")
    if len(t) > 1:
        t = t[:-1].replace("ك", "ک") + t[-1]
    t = re.sub(r"[\u2009\u200a\u200b\u202f]+", " ", t)
    t = t.replace("جه", "جە").replace("چه", "چە")
    return t



# ── Module-level analyzer + LRU cache ────────────────────────────────────────
_module_analyzer: Optional[zeyrek.MorphAnalyzer] = None
def _get_module_analyzer() -> zeyrek.MorphAnalyzer:
    global _module_analyzer
    if _module_analyzer is None:
        _module_analyzer = zeyrek.MorphAnalyzer()
    return _module_analyzer

@lru_cache(maxsize=8192)
def _cached_zeyrek_analyze(word: str):
    return list(_get_module_analyzer().analyze(word))

# ══════════════════════════════════════════════════════════════════════════════
# §1  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
SCORE_MAP: dict[str, float] = {
    "exact": 1.0, "override": 1.0, "surface": 1.0, "tags": 1.0, "punct": 1.0,
    "english": 0.7, "auto": 0.6,
}

KALIN_UNLULER    = set("aıou")
INCE_UNLULER     = set("eiöü")
YUVARLAK_UNLULER = set("ouöü")
DUZ_UNLULER      = set("aeıi")
_TUM_UNLULER     = KALIN_UNLULER | INCE_UNLULER

PHONEME_MAP: dict[str, str] = {
    "a":"ا","e":"ە","ı":"ی","i":"ی","o":"و","ö":"و","u":"و","ü":"و",
    "b":"ب","c":"ج","ç":"چ","d":"د","f":"ف","g":"گ","ğ":"غ","h":"ه",
    "j":"ژ","l":"ل","m":"م","n":"ن","p":"پ","r":"ر","ş":"ش","v":"و",
    "y":"ی","z":"ز","k":"ک","q":"ق",
}
PUNCTUATION_MAP: dict[str, str] = {",":"،",";":"؛","?":"؟","%":"٪"}
TR_LOWER_MAP     = str.maketrans("IİÇĞÖŞÜ","ıiçğöşü")
TR_NORMALIZE_MAP = str.maketrans("ÂâÎîÛû","AaİiUu")

_TR_WORD_CHARS = r"a-zA-Z0-9_ğüşıöçâîûÂÎÛĞÜŞİÖÇ"
_TR_WORD       = f"[{_TR_WORD_CHARS}]"
_TOKEN_RE      = re.compile(rf"{_TR_WORD}+(?:[\u2018\u2019']{_TR_WORD}+)?|[^\s{_TR_WORD_CHARS}]+")
_WORD_TOKEN_RE = re.compile(rf"^{_TR_WORD}+(?:[\u2018\u2019']{_TR_WORD}+)?$")

# ══════════════════════════════════════════════════════════════════════════════
# §2  FSM STATES
# ══════════════════════════════════════════════════════════════════════════════
ROOT="ROOT"; DERIVATION="DERIVATION"; DERIVATION_NOMINAL="DERIVATION_NOMINAL"
VOICE="VOICE"; NEGATION="NEGATION"; TENSE_ST="TENSE"; PERSON_ST="PERSON"
CASE_ST="CASE"; PLURAL_ST="PLURAL"; POSSESSIVE_ST="POSSESSIVE"; COPULA_ST="COPULA"

ROOT_POS_TAGS   = {"NOUN","VERB","ADJ","ADV","PRON","NUM","QUES"}
EMPTY_TAGS      = {"NOM","A3SG"}
PERSON_TAGS     = {"A1SG","A2SG","A3SG","A1PL","A2PL","A3PL"}
POSSESSIVE_TAGS = {"P1SG","P2SG","P3SG","P1PL","P2PL","P3PL"}
CASE_TAGS       = {"NOM","ACC","DAT","LOC","ABL","GEN","INS","REL","REL_LOC"}
VOICE_TAGS      = {"PASSIVE","CAUSATIVE","RECIPROCAL","REFLEXIVE"}
VERBAL_DERIVATION_TAGS = {
    "ABLE","UNABLE","ACQUIRE","INF1","INF2","PART","PAST_PART","CONV",
    "CONV_AFTER","CONV_BY","CONV_SINCE","CONV_ASLONGAS","CONV_WHILE","FUT_PART",
}
NOMINAL_DERIVATION_TAGS = {
    "NOM_DER_LIK","NOM_DER_LI","NOM_DER_SIZ",
    "NOM_DER_SEL","NOM_DER_CI","NOM_DER_DAS","NOM_DER_MSI",
}

TAG_FSM: dict[str,list[str]] = {
    ROOT:               [DERIVATION,DERIVATION_NOMINAL,VOICE,NEGATION,TENSE_ST,PERSON_ST,PLURAL_ST,POSSESSIVE_ST,CASE_ST,COPULA_ST],
    DERIVATION:         [DERIVATION,VOICE,NEGATION,TENSE_ST,PLURAL_ST,POSSESSIVE_ST,CASE_ST],
    DERIVATION_NOMINAL: [DERIVATION_NOMINAL,PLURAL_ST,POSSESSIVE_ST,CASE_ST,COPULA_ST],
    VOICE:              [VOICE,DERIVATION,NEGATION,TENSE_ST,PLURAL_ST,POSSESSIVE_ST,CASE_ST],
    NEGATION:           [VOICE,DERIVATION,TENSE_ST],
    TENSE_ST:           [TENSE_ST,PERSON_ST,COPULA_ST,CASE_ST],
    PERSON_ST:          [COPULA_ST,CASE_ST],
    PLURAL_ST:          [POSSESSIVE_ST,CASE_ST,COPULA_ST],
    POSSESSIVE_ST:      [CASE_ST,PLURAL_ST,COPULA_ST],
    CASE_ST:            [CASE_ST,PLURAL_ST,COPULA_ST],
    COPULA_ST:          [TENSE_ST,PERSON_ST,CASE_ST],
}
TAG_TO_STATE: dict[str,str] = {
    "ABLE":DERIVATION,"UNABLE":DERIVATION,"ACQUIRE":DERIVATION,
    "INF1":DERIVATION,"INF2":DERIVATION,"PART":DERIVATION,"PAST_PART":DERIVATION,
    "CONV":DERIVATION,"CONV_AFTER":DERIVATION,"CONV_BY":DERIVATION,
    "CONV_SINCE":DERIVATION,"CONV_ASLONGAS":DERIVATION,"CONV_WHILE":DERIVATION,
    "FUT_PART":DERIVATION,
    "PASSIVE":VOICE,"CAUSATIVE":VOICE,"RECIPROCAL":VOICE,"REFLEXIVE":VOICE,
    "NEG":NEGATION,
    "PAST":TENSE_ST,"NARR":TENSE_ST,"PROG":TENSE_ST,"PROG2":TENSE_ST,
    "FUTURE":TENSE_ST,"AOR":TENSE_ST,"COND":TENSE_ST,"NECES":TENSE_ST,"OPT":TENSE_ST,
    "A1SG":PERSON_ST,"A2SG":PERSON_ST,"A3SG":PERSON_ST,
    "A1PL":PERSON_ST,"A2PL":PERSON_ST,"A3PL":PERSON_ST,
    "PLURAL":PLURAL_ST,
    "P1SG":POSSESSIVE_ST,"P2SG":POSSESSIVE_ST,"P3SG":POSSESSIVE_ST,
    "P1PL":POSSESSIVE_ST,"P2PL":POSSESSIVE_ST,"P3PL":POSSESSIVE_ST,
    "NOM":CASE_ST,"ACC":CASE_ST,"DAT":CASE_ST,"LOC":CASE_ST,
    "ABL":CASE_ST,"GEN":CASE_ST,"INS":CASE_ST,"REL":CASE_ST,"REL_LOC":CASE_ST,
    "COPULA":COPULA_ST,"COPULA_ASSERT":COPULA_ST,
    "NOM_DER_LIK":DERIVATION_NOMINAL,"NOM_DER_LI":DERIVATION_NOMINAL,
    "NOM_DER_SIZ":DERIVATION_NOMINAL,"NOM_DER_SEL":DERIVATION_NOMINAL,
    "NOM_DER_CI":DERIVATION_NOMINAL,"NOM_DER_DAS":DERIVATION_NOMINAL,
    "NOM_DER_MSI":DERIVATION_NOMINAL,
    "EQU":CASE_ST,
}

# ══════════════════════════════════════════════════════════════════════════════
# §3  BASIC HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def normalize_tr_text(t:str)->str: return t.translate(TR_NORMALIZE_MAP) if t else t
def lower_tr(t:str)->str: return normalize_tr_text(t).translate(TR_LOWER_MAP).lower() if t else t
def fold_tr(t:str)->str:
    if not t: return t
    t=lower_tr(t)
    return t.replace("ç","c").replace("ğ","g").replace("ı","i").replace("ö","o").replace("ş","s").replace("ü","u")
def is_word_token(t:str)->bool: return bool(_WORD_TOKEN_RE.fullmatch(t))
def convert_ottoman_punctuation(t:str)->str: return "".join(PUNCTUATION_MAP.get(c,c) for c in t)
def last_vowel(t:str)->str:
    for ch in reversed(lower_tr(t or "")):
        if ch in _TUM_UNLULER: return ch
    return "a"
def is_vowel(ch:str)->bool: return lower_tr(ch or "")[:1] in _TUM_UNLULER
def starts_with_vowel(t:str)->bool: return bool(t) and is_vowel(t[0])
def ends_with_vowel(t:str)->bool: return bool(t) and is_vowel(t[-1])
def choose_harmony_A(s:str)->str: return "a" if last_vowel(s) in KALIN_UNLULER else "e"
def choose_harmony_I(s:str)->str:
    v=last_vowel(s); return {"a":"ı","ı":"ı","e":"i","i":"i","o":"u","u":"u"}.get(v,"ü")
def choose_harmony_U(s:str)->str: return choose_harmony_I(s)
def choose_initial_D(s:str)->str:
    return "t" if s and lower_tr(s[-1]) in {"ç","f","h","k","p","s","ş","t"} else "d"
def choose_initial_C(s:str)->str:
    return "ç" if s and lower_tr(s[-1]) in {"ç","f","h","k","p","s","ş","t"} else "c"
def strip_infinitive_from_ottoman(w:str)->str:
    for sfx in ("مق","مك","مک","ماق","ماک","مەك","مەک","mak","mek"):
        if w.endswith(sfx): return w[:-len(sfx)]
    return w
def normalize_surface_ascii(t:str)->str:
    t=lower_tr(t); return t.replace("\u2019","'").replace("\u2018","'").replace("'","")

# ══════════════════════════════════════════════════════════════════════════════
# §4  MORPHOPHONEMIC REPRESENTATIONS
# ══════════════════════════════════════════════════════════════════════════════
UNDERLYING_MORPHS: dict[str,str] = {
    "PASSIVE":"Il","CAUSATIVE":"DIr","RECIPROCAL":"Iş","REFLEXIVE":"In",
    "ABLE":"(y)Abil","UNABLE":"mA","ACQUIRE":"lAn","NEG":"mA",
    "INF1":"mAk","INF2":"mA","PART":"","PAST_PART":"DIk",
    "CONV":"","CONV_AFTER":"Ip","CONV_BY":"(y)ArAk",
    "CONV_SINCE":"(y)AlI","CONV_ASLONGAS":"DIkçA","CONV_WHILE":"ken",
    "PAST":"DI","NARR":"mIş","PROG":"Iyor","PROG2":"mAktA",
    "FUTURE":"(y)AcAk","FUT_PART":"(y)AcAk","AOR":"Ar","COND":"sA","NECES":"mAlI","OPT":"(y)A",
    "A1SG":"Im","A2SG":"sIn","A3SG":"","A1PL":"Iz","A2PL":"sInIz","A3PL":"lAr","PLURAL":"lAr",
    "P1SG":"(I)m","P2SG":"(I)n","P3SG":"sI","P1PL":"(I)mIz","P2PL":"(I)nIz","P3PL":"lArI",
    "NOM":"","ACC":"(y)I","DAT":"(y)A","LOC":"DA","ABL":"DAn","GEN":"(n)In","INS":"(y)lA",
    "REL":"ki","REL_LOC":"DAki","COPULA":"i","COPULA_ASSERT":"DIr",
    "NOM_DER_LIK":"lIk","NOM_DER_LI":"lI","NOM_DER_SIZ":"sIz",
    "NOM_DER_SEL":"sAl","NOM_DER_CI":"CI","NOM_DER_DAS":"DAş","NOM_DER_MSI":"ImsI",
    "EQU":"cA",
}

OTTOMAN_SURFACE_OVERRIDES: dict[str,str] = {
    "lar":"لر","ler":"لر","ları":"لری","leri":"لری",
    "lara":"لره","lere":"لره","lardan":"لردن","lerden":"لردن",
    "larla":"لرله","lerle":"لرله","larda":"لرده","lerde":"لرده",
    "ların":"لرڭ","lerin":"لرڭ",
    "larını":"لرینی","lerini":"لرینی","larına":"لرینە","lerine":"لرینە",
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
    # Dative — notebook: "a"→"ە", "e"→"ە", "ye"→"یە"
    "a":"ە","e":"ە","ya":"یه","ye":"یە",
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
    "acak":"اجق","ecek":"ەجك","yacak":"یاجق","yecek":"یهجك",
    "acağ":"اجغ","eceğ":"ەجگ","yacağ":"یاجغ","yeceğ":"یهجگ",
    "sın":"سڭ","sin":"سڭ","sun":"سڭ","sün":"سڭ",
    "sınız":"سڭز","siniz":"سڭز","sunuz":"سڭز","sünüz":"سڭز",
    "ız":"ز","iz":"ز","uz":"ز","üz":"ز",
    "ır":"یر","ir":"یر","ur":"یر","ür":"یر",
    "yız":"یز","yiz":"یز","yuz":"یوز","yüz":"یوز",
    "r":"ر","ar":"ار","er":"ر",
    "sa":"سە","se":"سە","malı":"ملی","meli":"ملی",
    "yor":"یور","ıyor":"ییور","iyor":"ییور","uyor":"ییور","üyor":"ییور",
    "makta":"مقدە","mekte":"مكدە",
    "abil":"ابیل","ebil":"ەبیل",
    "ıl":"یل","il":"یل","ul":"ول","ül":"ول",
    "an":"ان","en":"ان",
    "ıp":"یب","ip":"یب","up":"وب","üp":"وب",
    # PAST_PART suffixes
    "dık":"دق","dik":"دك","duk":"دق","dük":"دك",
    "tık":"دق","tik":"دك","tuk":"دق","tük":"دك",
    # Agent noun
    "ıcı":"یجی","ici":"یجی","ucu":"یجی","ücü":"یجی",
    "arak":"ارق","erek":"ارك",
    "lan":"لان","len":"لن",
    "lık":"لك","lik":"لك","luk":"لك","lük":"لك",
    "yın":"ین","yin":"ین","yun":"یون","yün":"یون",
    "maz":"مز","mez":"مز",
    "maman":"مامن","memen":"مەمن",
    "mamak":"مەمق","memek":"مەمك",
    "yalı":"یالی","yeli":"یەلی",
    "me":"مە",
    # Optative / prohibitive
    "yalım":"یالم",
    "ayım":"ایم",
    "eyim":"ەیم","yelim":"یەلم","alım":"الم","elim":"الم",
    "mayın":"مایڭ","meyin":"مەیڭ",
    # Copula tense
    "iken":"ایکن","yken":"ایکن","ise":"ایسه","ysa":"ایسه","yse":"ایسه",
    "idi":"ایدی",
    # notebook: ydı/ydi → "یدی" (not "ایدی")
    "ydı":"یدی","ydi":"یدی","ydu":"یدی","ydü":"یدی",
    "imiş":"ایمش","ymış":"ایمش","ymiş":"ایمش",
    "ydım":"ایدم","ydim":"ایدم","ydın":"ایدڭ","ydin":"ایدڭ",
    "ydık":"ایدق","ydik":"ایدک","ydınız":"ایدڭز","ydiniz":"ایدڭز",
    # notebook: ydılar → "یدیلر" (not "ایدلر")
    "ydılar":"یدیلر","ydiler":"یدیلر",
    "iyordu":"ایوردی","iyormuş":"ایورمش","iyorsa":"ایورسه",
    "ıyordu":"ایوردی","ıyormuş":"ایورمش","ıyorsa":"ایورسه",
    "ndan":"ندن","nden":"ندن","nda":"نده","nde":"نده","na":"نه","ne":"نه",
    "mayalı":"مایالی","meyeli":"میەلی",
    # CONV_WHILE / CONV_ASLONGAS
    "ken":"كن",
    "dıkça":"دقجە","dikçe":"دكجە","dukça":"دقجە","dükçe":"دكجە",
    "tıkça":"دقجە","tikçe":"دكجە","tukça":"دقجە","tükçe":"دكجە",
}

VOWEL_DROP_WORDS = {
    "burun","ağız","karın","oğul","gönül","omuz","akıl","şehir","nehir","sabır","ömür",
}

WORD_OVERRIDES: dict[str,str] = {
    "Apple":"آپپلە","Almanya'da":"آلمانیەدە",
    "ağır":"آغير","bağır":"باغر",
    "binalarının":"بنالرینڭ","başka":"باشقە",
    "çabaladığınız":"چابالادیغڭز",
    "da":"دە","de":"دە","del":"دل",
    "değişikliğe":"دگیشیكلگە","deneyin":"دڭیڭ",
    "için":"ایچون","ile":"ایلە","lise":"لیسه",
    "Messi":"مسّی","oyunda":"اویوندە","oynama":"اوينامە",
    "belirtmek":"بلیرتمك","belirtildi":"بلیرتیلدی",
    "Şikayetinizi":"شكایتیڭزی",
    "araştırıp":"آراشدیریپ","ar":"آری","azalt":"آزالت",
    "çalışacağız":"چالیشاجغز","çözmeye":"چوزمەیە",
    "başar":"باشار","bilgilen":"بیلگیلن",
    "ağlayan":"آغلايان","bulur":"بولور","bulduğum":"بولديغم",
    "denizdir":"دڭزدر","düş":"دوش","kazan":"قزان",
    "umutsuzluğa":"اوموتسزلغه",
    "yaşa":"یاشا","yaşam":"یاشام",
    "yaşayabilmesinin":"یاشایابیلمەسنڭ","yaşayarak":"یاشایارق",
    "yaşasaydınız":"یاشاسەیدیڭز","yaşayacaklar":"یاشایاجقلر",
    "vazgeçmek":"واز گچمك","vazgeç":"واز گچ",
    "tesislerinin":"تأسیسلرینڭ","çatılarına":"چاتیلارینە",
    "eklemeyi":"اكلمەیی","yerleştirilecek":"یرلشدیریلەجك",
    "megavat":"مغه وات","mısınız":"میسڭز",
    "yalvar":"یالوار","yetecek":"یتەجك","zarfında":"ظرفندە",
    "zorla":"زورلا","paylaş":"پایلاش",
    "elektrik":"الكتریگ","üretecek":"أورتجك",
    "gelir":"گلیر","misin":"میسڭ",
    "güven":          "گوگن",
    "mürettebatlı":   "مرتّباتلی",
    "istemişti":      "ایستەمشدی",
    "seyrediyorum":"سیر ایدییورم","izleyen":"ایزلەین",
    "gidilen":"گیدیلن","gelince":"گلینجە",
    "adamak":"آدامق","adadı":"آدادی",
    "meslektaşlarımızı":"مسلكداشلریمزی",
    "hayat":          "حیات",
    "burhanettin":    "برهان الدّین",
    "sultan":         "سلطان",
    "ama":            "اما",
    "rahmet":         "رحمت",
    "başsağlığı":     "باشساغلغی",
    "allah":          "اللّٰه",
    "almanya":        "آلمانیە",
    "antalya":        "آنطالیە",
    "azerbaycan":     "آذربيجان",
    "istanbul":       "استانبول",
    "kütahya":        "كوتاهیە",
    "erzincan":       "ارزنجان",
    "gazze":          "غزّە",
    "hollanda":       "هوللانده",
    "israil":         "اسرائیل",
    "lübnan":         "لبنان",
    "medine":         "مدينه",
    "mekke":          "مكه",
    "mısır":          "مصر",
    "pakistan":       "پاكستان",
    "sapanca":        "صپانجە",
    "tekne":          "تكنە",
    "türkiye":        "توركیه",
    "dile":           "دیلە",
    "yakın":          "یاقین",
    "ver":            "ویر",
    "avrupa":         "آوروپە",
    "savunurken":     "صاوونیركن",
    "vurguluyor":     "وورغولییور",
    "çöküş":          "چوكوش",
    "çöküşten":       "چوكوشدن",
}
WORD_OVERRIDES = {k: normalize_ottoman_lookup_value(v) for k, v in WORD_OVERRIDES.items()}

# ══════════════════════════════════════════════════════════════════════════════
# §5  MORPHOPHONEMIC RULES
# ══════════════════════════════════════════════════════════════════════════════
def apply_vowel_harmony(morph:str, root:str)->str:
    return morph.replace("A",choose_harmony_A(root)).replace("I",choose_harmony_I(root))
def apply_vowel_drop(root:str, morph:str)->str:
    if lower_tr(root) not in VOWEL_DROP_WORDS or not morph or not starts_with_vowel(morph): return root
    return root[:-2]+root[-1] if len(root)>=3 else root
def apply_consonant_softening(root:str, morph:str)->str:
    if not morph or not starts_with_vowel(morph): return root
    lr=lower_tr(root)
    if lr.endswith("nk"): return root[:-1]+"g"
    rep={"p":"b","ç":"c","t":"d","k":"ğ"}
    last=lr[-1] if lr else ""
    return root[:-1]+rep[last] if last in rep else root
def apply_buffer_consonants(prev:str, morph:str)->str:
    for tok,char in [("(y)","y"),("(n)","n"),("(s)","s"),("(ş)","ş")]:
        if morph.startswith(tok):
            return (char if ends_with_vowel(prev) else "")+morph[3:]
    return morph

def resolve_nominal_possessive(tag:str, prev:str)->str:
    hi,ha,ev=choose_harmony_I(prev),choose_harmony_A(prev),ends_with_vowel(prev)
    if tag=="P1SG": return "m"        if ev else hi+"m"
    if tag=="P2SG": return "n"        if ev else hi+"n"
    if tag=="P3SG": return "s"+hi     if ev else hi
    if tag=="P1PL": return "m"+hi+"z" if ev else hi+"m"+hi+"z"
    if tag=="P2PL": return "n"+hi+"z" if ev else hi+"n"+hi+"z"
    if tag=="P3PL": return "l"+ha+"r"+choose_harmony_I(prev)
    return ""

def resolve_nominal_case(tag:str, prev:str, possessed:bool=False)->str:
    hi,ha,ev=choose_harmony_I(prev),choose_harmony_A(prev),ends_with_vowel(prev)
    D=choose_initial_D(prev)
    if tag=="NOM":    return ""
    if tag=="ACC":    return ("n"+hi)           if possessed else (("y" if ev else "")+hi)
    if tag=="DAT":    return ("n"+ha)           if possessed else (("y" if ev else "")+ha)
    if tag=="LOC":    return ("n"+D+ha)         if possessed else (D+ha)
    if tag=="ABL":    return ("n"+D+ha+"n")     if possessed else (D+ha+"n")
    if tag=="GEN":    return (("n" if ev or possessed else "")+hi+"n")
    if tag=="INS":    return ("y" if ev else "")+"l"+ha
    if tag=="REL":    return "ki"
    if tag=="REL_LOC":return D+ha+"ki"
    return ""

def resolve_copula_variant(follow_tag:str, prev:str)->str:
    buf="y" if ends_with_vowel(prev) else ""
    if follow_tag=="PAST":  return buf+choose_initial_D(prev)+choose_harmony_I(prev)
    if follow_tag=="NARR":  return buf+"m"+choose_harmony_I(prev)+"ş"
    if follow_tag=="COND":  return buf+"s"+choose_harmony_A(prev)
    if follow_tag=="NECES": return (buf or "i")+"d"+choose_harmony_I(prev)+"r"
    return ""

def fuse_tag_strings(tags:list[str])->list[str]:
    out,i=[],0
    while i<len(tags):
        if tags[i:i+2]==["LOC","REL"]: out.append("REL_LOC"); i+=2
        else: out.append(tags[i]); i+=1
    return out

def fuse_realized_morphs(seq:list[dict])->list[dict]:
    out,i=[],0
    while i<len(seq):
        item=dict(seq[i])
        if out:
            prev=out[-1]
            if prev["tag"]=="NEG" and item["tag"] in {"CONV_SINCE","NARR","INF1"}:
                prev["tag"]=item["tag"]; prev["surface"]+=item["surface"]; i+=1; continue
            if prev["tag"]=="COND" and item["tag"]=="PAST":
                item["surface"]="y"+choose_initial_D(prev["surface"])+choose_harmony_I(prev["surface"])
            if prev["tag"]=="COND" and item["tag"] in {"A1SG","A2SG","A1PL","A2PL"}:
                hi=choose_harmony_I(prev["surface"])
                item["surface"]={"A1SG":"m","A2SG":"n","A1PL":"k","A2PL":"n"+hi+"z"}[item["tag"]]
            # OPT + A1PL → yalım/yelim
            if prev["tag"]=="OPT" and item["tag"]=="A1PL":
                prev["surface"]=prev["surface"]+"l"+choose_harmony_I(prev["surface"])+"m"
                i+=1; continue
            if prev["tag"]=="OPT" and item["tag"]=="A1SG":
                prev["surface"]=prev["surface"]+"y"+choose_harmony_I(prev["surface"])+"m"
                i+=1; continue
            if prev["tag"]=="UNABLE" and item["tag"]=="AOR":
                item["surface"]="z"
            if prev["tag"]=="NEG" and item["tag"]=="AOR":
                prev["tag"]="AOR"; prev["surface"]+="z"; i+=1; continue
            if (prev["tag"]=="NEG" and item["tag"]=="INF2"
                    and i+1<len(seq) and seq[i+1]["tag"]=="P2SG"):
                prev["tag"]="INF2"; prev["surface"]+=item["surface"]+seq[i+1]["surface"]; i+=2; continue
            # INF2 + P3SG: drop last char of inf2
            if prev["tag"]=="INF2" and item["tag"]=="P3SG":
                prev["surface"]=prev["surface"][:-1]
            # FUTURE/FUT_PART: acak→acağ before vowel
            if (prev["tag"] in {"FUTURE","FUT_PART"}
                    and item["surface"][:1] in _TUM_UNLULER
                    and prev["surface"].endswith(("acak","ecek","yacak","yecek"))):
                prev["surface"]=prev["surface"][:-1]+"ğ"
            # PAST_PART: dık→dığ before vowel
            if (prev["tag"]=="PAST_PART"
                    and item["surface"][:1] in _TUM_UNLULER
                    and prev["surface"].endswith(("dık","dik","duk","dük","tık","tik","tuk","tük"))):
                prev["surface"]=prev["surface"][:-1]+"ğ"
            if prev["tag"]=="PAST" and item["tag"] in {"A1SG","A2SG","A1PL","A2PL"}:
                hi=choose_harmony_I(prev["surface"])
                item["surface"]={"A1SG":"m","A2SG":"n","A1PL":"k","A2PL":"n"+hi+"z"}[item["tag"]]
            if prev["tag"]=="REL" and item["tag"]=="DAT":
                prev["surface"]=prev["surface"][:-2]+"ye"; i+=1; continue
        out.append(item); i+=1
    return out

def build_underlying_morphs(tags:list[str])->list[dict]:
    out=[]
    for tag in tags:
        if tag in ROOT_POS_TAGS or tag in EMPTY_TAGS: continue
        u=UNDERLYING_MORPHS.get(tag)
        if u is not None: out.append({"tag":tag,"underlying":u})
    return out

def realize_single_morph(
    prev:str, morph:dict,
    next_tag:Optional[str]=None,
    copula_mode:bool=False,
    possessed:bool=False,
)->Optional[str]:
    tag,underlying=morph["tag"],morph["underlying"]
    if tag in POSSESSIVE_TAGS: return resolve_nominal_possessive(tag,prev)
    if tag in CASE_TAGS:       return resolve_nominal_case(tag,prev,possessed=possessed)
    if tag=="PLURAL":          return "l"+choose_harmony_A(prev)+"r"
    if tag=="A1SG":            return choose_harmony_I(prev)+"m"
    if tag=="A1PL":            return choose_harmony_I(prev)+"z"
    if tag=="A2SG":            return "s"+choose_harmony_I(prev)+"n"
    if tag=="A2PL":
        hi=choose_harmony_I(prev); return "s"+hi+"n"+hi+"z"
    if tag=="A3PL":            return "l"+choose_harmony_A(prev)+"r"
    if copula_mode and tag in {"PAST","NARR","COND","NECES"}:
        return resolve_copula_variant(tag,prev)
    if tag=="PAST":   return choose_initial_D(prev)+choose_harmony_I(prev)
    if tag=="NARR":   return "m"+choose_harmony_I(prev)+"ş"
    if tag=="PROG":   return choose_harmony_I(prev)+"yor"
    if tag=="PROG2":
        ha=choose_harmony_A(prev); return "m"+ha+"kt"+ha
    if tag in {"FUTURE","FUT_PART"}:
        ha=choose_harmony_A(prev); return ("y" if ends_with_vowel(prev) else "")+ha+"c"+ha+"k"
    if tag=="AOR":
        ns=normalize_surface_ascii(prev)
        if ns in {"bul"}: return choose_harmony_I(prev)+"r"
        if lower_tr(prev).endswith(("dır","dir","dur","dür","tır","tir","tur","tür")):
            return choose_harmony_I(prev)+"r"
        if ends_with_vowel(prev): return "r"
        vc=sum(1 for ch in lower_tr(prev) if ch in _TUM_UNLULER)
        return choose_harmony_I(prev)+"r" if vc>1 else choose_harmony_A(prev)+"r"
    if tag=="COND":   return "s"+choose_harmony_A(prev)
    if tag=="NECES":  return "m"+choose_harmony_A(prev)+"l"+choose_harmony_I(prev)
    if tag=="OPT":    return ("y" if ends_with_vowel(prev) else "")+choose_harmony_A(prev)
    if tag=="NEG":
        if next_tag=="PROG": return "m"
        return "m"+choose_harmony_A(prev)
    if tag=="UNABLE":
        ha=choose_harmony_A(prev)
        return "y"+ha+"m"+ha if ends_with_vowel(prev) else "m"+ha
    if tag=="ACQUIRE": return "l"+choose_harmony_A(prev)+"n"
    if tag=="ABLE":    return ("y" if ends_with_vowel(prev) else "")+choose_harmony_A(prev)+"bil"
    if tag=="PASSIVE":
        return "n" if ends_with_vowel(prev) or lower_tr(prev).endswith("l") \
               else choose_harmony_I(prev)+"l"
    if tag=="CAUSATIVE":
        return "t" if ends_with_vowel(prev) else choose_initial_D(prev)+choose_harmony_I(prev)+"r"
    if tag=="RECIPROCAL": return choose_harmony_I(prev)+"ş"
    if tag=="REFLEXIVE":  return choose_harmony_I(prev)+"n"
    if tag in {"PART","CONV"}: return ""
    if tag=="PAST_PART":   return "d"+choose_harmony_I(prev)+"k"  # softened in fuse step
    if tag=="CONV_AFTER":  return choose_harmony_U(prev)+"p"
    if tag=="CONV_BY":
        return ("y" if ends_with_vowel(prev) else "")+choose_harmony_A(prev)+"r"+choose_harmony_A(prev)+"k"
    if tag=="CONV_WHILE":    return "ken"
    if tag=="CONV_ASLONGAS":
        return choose_initial_D(prev)+choose_harmony_I(prev)+"kç"+choose_harmony_A(prev)
    if tag=="INF1": return "m"+choose_harmony_A(prev)+"k"
    if tag=="INF2": return "m"+choose_harmony_A(prev)
    if tag=="NOM_DER_LIK": return "l"+choose_harmony_I(prev)+"k"
    if tag=="NOM_DER_LI":  return "l"+choose_harmony_I(prev)
    if tag=="NOM_DER_SIZ": return "s"+choose_harmony_I(prev)+"z"
    if tag=="NOM_DER_SEL": return "s"+choose_harmony_A(prev)+"l"
    if tag=="NOM_DER_CI":
        if lower_tr(prev).endswith(("t","d")):
            hi=choose_harmony_I(prev); return hi+"c"+hi
        return choose_initial_C(prev)+choose_harmony_I(prev)
    if tag=="NOM_DER_DAS": return choose_initial_D(prev)+choose_harmony_A(prev)+"ş"
    if tag=="NOM_DER_MSI": return choose_harmony_I(prev)+"ms"+choose_harmony_I(prev)
    if tag=="EQU": return "c"+choose_harmony_A(prev)
    realized=apply_vowel_harmony(underlying,prev)
    return apply_buffer_consonants(prev,realized)

def realize_allomorphs(root_surface:str, morphs:list[dict])->tuple[str,list[dict]]:
    current_root=lower_tr(root_surface); realized:list[dict]=[]; copula_mode=False; has_possessive=False
    for idx,morph in enumerate(morphs):
        tag=morph["tag"]; next_tag=morphs[idx+1]["tag"] if idx+1<len(morphs) else None
        if tag=="COPULA": copula_mode=True; continue
        prev=current_root+"".join(p["surface"] for p in realized)
        piece=realize_single_morph(prev,morph,next_tag,copula_mode,has_possessive)
        if piece is None: continue
        if not realized and tag=="PROG" and current_root and current_root[-1] in "aeıioöuü":
            current_root=current_root[:-1]
        if not realized and starts_with_vowel(piece):
            current_root=apply_vowel_drop(current_root,piece)
            # Extended exemption list from notebook
            if tag not in {"PASSIVE","FUTURE","FUT_PART","CONV_AFTER","ABLE","OPT","AOR"}:
                current_root=apply_consonant_softening(current_root,piece)
            prev=current_root+"".join(p["surface"] for p in realized)
            piece=realize_single_morph(prev,morph,next_tag,copula_mode,has_possessive)
        piece=normalize_surface_ascii(piece)
        realized.append({"tag":tag,"surface":piece})
        if tag in POSSESSIVE_TAGS: has_possessive=True
    return current_root, fuse_realized_morphs(realized)

# ══════════════════════════════════════════════════════════════════════════════
# §6  OTTOMAN RENDER LAYER
# ══════════════════════════════════════════════════════════════════════════════
def render_ottoman(surface:str, historical:bool=False)->str:
    if not surface: return ""
    normalized=normalize_surface_ascii(surface)
    if historical and normalized in OTTOMAN_SURFACE_OVERRIDES:
        return OTTOMAN_SURFACE_OVERRIDES[normalized]
    result=""; skip_first=False
    if normalized and normalized[0] in _TUM_UNLULER:
        result+="ا"; skip_first=True
    first_v=next((c for c in normalized if c in _TUM_UNLULER),"a")
    harmony="kalin" if first_v in KALIN_UNLULER else "ince"
    for idx,ch in enumerate(normalized):
        if ch in _TUM_UNLULER: harmony="kalin" if ch in KALIN_UNLULER else "ince"
        if skip_first and idx==0: continue
        if ch=="k":
            if idx==len(normalized)-1: result+="ك"
            else: result+="ق" if harmony=="kalin" else "ک"
        elif ch=="t": result+="ط" if harmony=="kalin" else "ت"
        elif ch=="s": result+="ص" if harmony=="kalin" else "س"
        else:         result+=PHONEME_MAP.get(ch,ch)
    return result

def render_suffix_ottoman(tag:str, surface:str, historical:bool=True)->str:
    """Render a realized suffix, with tag-specific overrides."""
    normalized=normalize_surface_ascii(surface)
    if tag=="EQU": return "جە"
    # FUTURE yecek/yeceğ: use ە not ه
    if tag in {"FUTURE","FUT_PART"} and normalized in {"yecek","yeceğ"}:
        return {"yecek":"یەجك","yeceğ":"یەجگ"}[normalized]
    # PAST_PART diğ/tiğ: unified form
    if tag=="PAST_PART" and normalized in {"diğ","tiğ"}:
        return "دیگ"
    # CAUSATIVE dır/tır: bypass historical overrides
    if tag=="CAUSATIVE" and normalized=="t":
        return "ت"
    if tag=="CAUSATIVE" and normalized in {"dır","dir","dur","dür","tır","tir","tur","tür"}:
        return render_ottoman(surface,historical=False)
    return render_ottoman(surface,historical=historical)

def merge_ottoman(root_ot:str, suffixes:list[dict])->str:
    rendered=[p["ottoman"] for p in suffixes if p["ottoman"]]
    vtags=[p["tag"] for p in suffixes if p["ottoman"]]
    first=rendered[0] if rendered else ""
    if first.startswith("لر") and root_ot.endswith("ه"): root_ot=root_ot[:-1]+"ە"
    if first.startswith("لر") and len(rendered)>=2 and rendered[1]=="ه": rendered[1]="ە"
    if vtags[:1]==["REL_LOC"] and root_ot.endswith("ه"): root_ot=root_ot[:-1]+"ە"
    # P3SG + ACC
    if vtags[:2]==["P3SG","ACC"] and rendered[:2]==["سی","نی"]:
        if root_ot.endswith("ه"): root_ot=root_ot[:-1]+"ە"
        rendered=["سنی"]+rendered[2:]
    elif vtags[:2]==["P3SG","ACC"] and rendered[:2]==["ی","نی"]:
        rendered=["ینی"]+rendered[2:]
    # P3SG + DAT
    if vtags[:2]==["P3SG","DAT"] and rendered[:2]==["ی","نه"]:
        rendered=["نە"]+rendered[2:]
    # P3SG + LOC
    if vtags[:2]==["P3SG","LOC"] and rendered[:2]==["ی","نده"]:
        rendered=["نده"]+rendered[2:]
    # P3SG + GEN
    if vtags[:2]==["P3SG","GEN"] and rendered[:2]==["سی","نڭ"]:
        rendered=["سنڭ"]+rendered[2:]
    # P3SG + ABL
    if vtags[:2]==["P3SG","ABL"] and rendered[:2]==["ی","ندن"]:
        rendered=["ندن"]+rendered[2:]
    # P2SG + ACC
    if vtags[:2]==["P2SG","ACC"] and rendered[:2]==["ن","نی"]:
        if root_ot.endswith("ه"): root_ot=root_ot[:-1]+"ە"
        rendered=["ڭی"]+rendered[2:]
    # P1SG + ACC
    if vtags[:2]==["P1SG","ACC"] and rendered[:2]==["م","نی"]:
        rendered=["می"]+rendered[2:]
    # PAST_PART combinations (scanned at any depth)
    for idx in range(len(vtags)-2):
        if vtags[idx:idx+3]==["PAST_PART","P2SG","LOC"] and rendered[idx+1:idx+3]==["ڭ","نده"]:
            rendered=rendered[:idx+1]+["ندە"]+rendered[idx+3:]; break
        if vtags[idx:idx+3]==["PAST_PART","P3SG","LOC"] and rendered[idx+1:idx+3]==["ی","نده"]:
            rendered=rendered[:idx+1]+["ندە"]+rendered[idx+3:]; break
        if vtags[idx:idx+3]==["PAST_PART","P3SG","ACC"] and rendered[idx+1:idx+3] in (["ی","نی"],["ینی"]):
            rendered=rendered[:idx+1]+["نی"]+rendered[idx+3 if rendered[idx+1:idx+3]==["ی","نی"] else idx+2:]; break
        if vtags[idx:idx+3]==["PAST_PART","P3SG","ABL"] and rendered[idx+1:idx+3]==["ی","ندن"]:
            rendered=rendered[:idx+1]+["ندن"]+rendered[idx+3:]; break
        if vtags[idx:idx+3]==["PAST_PART","P3SG","GEN"] and rendered[idx+1:idx+3]==["سی","نڭ"]:
            rendered=rendered[:idx+1]+["سنڭ"]+rendered[idx+3:]; break
    # CAUSATIVE + PROG
    if vtags[:2]==["CAUSATIVE","PROG"] and rendered[:2] in (["ت","یور"],["ط","یور"]):
        rendered=["تییور"]+rendered[2:]
    if first.startswith("ی") and root_ot.endswith("ه"): root_ot=root_ot[:-1]+"ە"
    return root_ot+"".join(rendered)

def adjust_etmek_auxiliary_output(lemma:Optional[str], realized_root:str, rendered_sfx:list[dict])->tuple[Optional[str],list[dict]]:
    if lower_tr(lemma or "")!="etmek": return None,rendered_sfx
    if normalize_surface_ascii(realized_root)!="ed" or not rendered_sfx: return None,rendered_sfx
    adjusted=[dict(p) for p in rendered_sfx]
    fs=normalize_surface_ascii(adjusted[0].get("surface",""))
    if not starts_with_vowel(fs): return None,adjusted
    aux="ایدی" if fs.startswith(("iyor","ıyor","uyor","üyor")) else "اید"
    fo=adjusted[0].get("ottoman","")
    if fo.startswith("ه"): adjusted[0]["ottoman"]="ە"+fo[1:]
    return aux,adjusted

def adjust_past_person_rendering(rendered_sfx:list[dict])->list[dict]:
    adjusted=[dict(p) for p in rendered_sfx]
    for i in range(1,len(adjusted)):
        prev,item=adjusted[i-1],adjusted[i]
        if prev.get("tag")!="PAST": continue
        if item.get("tag") not in {"A1SG","A2SG","A1PL","A2PL"}: continue
        ps=normalize_surface_ascii(prev.get("surface",""))
        syd=ps.startswith("yd")
        prev["ottoman"]="ید" if syd else "د"
        if item.get("tag")=="A2SG":  item["ottoman"]="ڭ"
        elif item.get("tag")=="A1PL": item["ottoman"]="ق" if syd else "ك"
        elif item.get("tag")=="A2PL": prev["ottoman"]="یدی" if syd else "دی"; item["ottoman"]="ڭز"
    return adjusted

def adjust_softened_nominal_root_ottoman(root_ot:str, lemma:str, realized_root:str)->str:
    base=normalize_surface_ascii(lemma or ""); realized=normalize_surface_ascii(realized_root or "")
    if not base or not realized or len(base)!=len(realized) or base[:-1]!=realized[:-1]: return root_ot
    pair=(base[-1],realized[-1])
    if pair==("p","b") and root_ot.endswith("پ"): return root_ot[:-1]+"ب"
    if pair==("ç","c") and root_ot.endswith("چ"):  return root_ot[:-1]+"ج"
    if pair==("t","d") and root_ot.endswith(("ت","ط")):
        if root_ot.endswith("ات"): return root_ot
        return root_ot[:-1]+"د"
    if pair==("k","ğ"):
        if root_ot.endswith("ق"): return root_ot[:-1]+"غ"
        if root_ot.endswith(("ک","ك")): return root_ot[:-1]+"گ"
    return root_ot

# ══════════════════════════════════════════════════════════════════════════════
# §7  ENGLISH TRANSLITERATION
# ══════════════════════════════════════════════════════════════════════════════
_ENG_VOWELS=set("aeiou")
_ENG_PATTERNS:list[tuple[str,str]]=[
    ("tion","شن"),("sion","شن"),("ture","چر"),("ough","و"),("augh","اف"),("ight","ایت"),
    ("tch","چ"),("dge","ج"),("sch","ش"),("ght","ت"),("gue","گ"),("que","ک"),
    ("igh","ای"),("ssi","ش"),("sci","ش"),("kno","نو"),("wri","ری"),("psy","سی"),("pneu","نو"),("rhy","ری"),
    ("ph","ف"),("sh","ش"),("ch","چ"),("th","ث"),("wh","و"),("ck","ک"),
    ("ng","ڭ"),("nk","ڭک"),("qu","کو"),("kn","ن"),("wr","ر"),("gn","ن"),("ps","س"),
    ("mb","م"),("gh",""),("ee","ی"),("ea","ی"),("oo","و"),("ou","او"),("ow","او"),("oa","و"),
    ("ai","ای"),("ay","ای"),("ei","ی"),("ie","ی"),("oi","وی"),("oy","وی"),("au","او"),("aw","او"),
    ("ew","یو"),("ue","یو"),("ae","ی"),("oe","و"),("ui","وی"),("ia","یه"),("io","یو"),("ua","وه"),
    ("lk","لک"),("lm","لم"),("mn","م"),("rh","ر"),("xc","کس"),("ww","و"),
    ("ss","س"),("ll","ل"),("tt","ت"),("nn","ن"),("rr","ر"),("pp","پ"),("bb","ب"),
    ("dd","د"),("ff","ف"),("gg","گ"),("cc","ک"),("mm","م"),("zz","ز"),
]
_ENG_SINGLE:dict[str,str]={
    "a":"ه","b":"ب","c":"ک","d":"د","e":"ه","f":"ف","g":"گ","h":"ه","i":"ی","j":"ج",
    "k":"ک","l":"ل","m":"م","n":"ن","o":"و","p":"پ","q":"ق","r":"ر","s":"س","t":"ت",
    "u":"ا","v":"ڤ","w":"و","x":"کس","y":"ی","z":"ز",
}
ENGLISH_WORD_OVERRIDES:dict[str,str]={
    "internet":"اینترنت","computer":"کمپیوتر","digital":"دیجیتل","software":"سافتویر",
    "hardware":"هاردویر","network":"نتورک","telephone":"تلفون","television":"تلویزیون",
    "radio":"رادیو","video":"ویدیو","photo":"فوطو","camera":"کامره","email":"ایمیل",
    "website":"ویب سایت","password":"پاسورد","download":"داونلود","upload":"آپلود",
    "london":"لندره","paris":"پاریس","berlin":"برلین","moscow":"مسکو","new york":"نیویورک",
    "ok":"اوکی","okay":"اوکی","yes":"یس","no":"نو","hello":"هللو","bye":"بای",
    "english":"انگیلیزچه","french":"فرانسزچه","german":"الماندجه",
    "america":"امریقا","europe":"اوروپا","asia":"آسیا","africa":"افریقا",
}
def is_likely_english(word:str)->bool:
    w=word.lower()
    if any(c in w for c in "ğüşıöç"): return False
    if any(p in w for p in ("wh","ph","tch","tion","ght","ough")): return True
    if any(w.startswith(p) for p in ("kn","wr","gn","ps")): return True
    if "w" in w or "x" in w: return True
    for sfx in ("tion","sion","ness","ment","ful","less","ive","ous","ing","ance",
                "ence","able","ible","ity","ify","ize","ise","ism","ist","ish","ward"):
        if w.endswith(sfx) and len(w)>len(sfx)+1: return True
    return False
def render_english_ottoman(word:str)->str:
    wl=word.lower()
    if wl in ENGLISH_WORD_OVERRIDES: return ENGLISH_WORD_OVERRIDES[wl]
    s,n,out,i=wl,len(wl),"",0
    while i<n:
        matched=False
        for pattern,ottoman in _ENG_PATTERNS:
            pl=len(pattern)
            if i+pl<=n and s[i:i+pl]==pattern:
                if pattern=="gh" and i==0:        ottoman="غ"
                elif pattern=="mb" and i+pl<n:    ottoman="مب"
                elif pattern=="ou" and i+pl<n and s[i+pl] in "lr": ottoman="ور"
                out+=ottoman; i+=pl; matched=True; break
        if matched: continue
        ch=s[i]
        if ch=="c":   out+="س" if i+1<n and s[i+1] in "eiy" else "ک"
        elif ch=="g": out+="ج" if i+1<n and s[i+1] in "eiy" else "گ"
        elif ch=="e" and i==n-1 and i>0 and s[i-1] not in _ENG_VOWELS: pass
        elif ch=="a" and i+2<n and s[i+1] not in _ENG_VOWELS and s[i+2]=="e" and i+3>=n: out+="ای"
        elif ch=="i" and i+2<n and s[i+1] not in _ENG_VOWELS and s[i+2]=="e" and i+3>=n: out+="ای"
        else: out+=_ENG_SINGLE.get(ch,ch)
        i+=1
    if out and out[0] in {"ه","ی","و"}: out="ا"+out
    return out

# ══════════════════════════════════════════════════════════════════════════════
# §8  ZEYREK TAG MAP + SURFACE FALLBACKS
# ══════════════════════════════════════════════════════════════════════════════
ZEYREK_TAG_MAP:dict[str,str]={
    "Passive":"PASSIVE","Pass":"PASSIVE","Caus":"CAUSATIVE",
    "Recip":"RECIPROCAL","Reflex":"REFLEXIVE",
    "Able":"ABLE","Unable":"UNABLE","Acquire":"ACQUIRE","Neg":"NEG",
    "Inf1":"INF1","Inf2":"INF2","Part":"PART","PastPart":"PAST_PART","FutPart":"FUT_PART",
    "Conv":"CONV","AfterDoingSo":"CONV_AFTER","ByDoingSo":"CONV_BY",
    "SinceDoingSo":"CONV_SINCE","AsLongAs":"CONV_ASLONGAS","While":"CONV_WHILE",
    "Past":"PAST","Narr":"NARR","Prog1":"PROG","Prog2":"PROG2",
    "Fut":"FUTURE","Aor":"AOR","Cond":"COND","Desr":"COND","Neces":"NECES","Opt":"OPT",
    "Cop":"COPULA_ASSERT",
    "A1sg":"A1SG","A2sg":"A2SG","A3sg":"A3SG","A1pl":"A1PL","A2pl":"A2PL","A3pl":"A3PL",
    "P1sg":"P1SG","P2sg":"P2SG","P3sg":"P3SG","P1pl":"P1PL","P2pl":"P2PL","P3pl":"P3PL",
    "Nom":"NOM","Acc":"ACC","Dat":"DAT","Loc":"LOC","Abl":"ABL","Gen":"GEN","Ins":"INS","Rel":"REL",
    "With":"NOM_DER_LI","Without":"NOM_DER_SIZ","Agt":"NOM_DER_CI",
    "Ness":"NOM_DER_LIK","FitFor":"NOM_DER_LIK",
    "Related":"NOM_DER_SEL",
    "Equ":"EQU","Ly":"EQU","AsIf":"EQU",
}

_APOSTROPHE_SUFFIXES:set[str]={
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
    "ydık","ydik","ydınız","ydiniz","ymış","ymiş","ysa","yse",
    "lar","ler","ların","lerin","lara","lere",
    "lardan","lerden","larla","lerle","larda","lerde",
}

# Surface-only fallback for verb + frozen suffix combos
SURFACE_VERB_FALLBACK_SUFFIXES:dict[str,str]={
    "irken":"یركن","rken":"ركن",
    "sınız":"سینز","siniz":"سینز","sunuz":"سینز","sünüz":"سینز",
}
SURFACE_FUTURE_CHAIN_SUFFIXES:dict[str,str]={
    "tim":"دم","tin":"دڭ","ti":"دی","tık":"دق","tik":"دك","tuk":"دق","tük":"دك",
    "tiniz":"دیڭز","tiler":"دیلر","lar":"لر","ler":"لر",
    "mış":"مش","miş":"مش","muş":"مش","müş":"مش",
}


# ── Apostrophe suffix renderer ────────────────────────────────────────────────
def _render_apostrophe_suffix(suffix: str) -> str:
    normalized = lower_tr(suffix)
    if normalized in {"ya","ye"}: return "یە"
    if normalized in {"da","de","ta","te"}: return "دە"
    return OTTOMAN_SURFACE_OVERRIDES.get(normalized) or render_ottoman(normalized, historical=True)

# ── Lexicalized nominal overrides ────────────────────────────────────────────
LEXICALIZED_NOMINAL_STEM_OVERRIDES: dict[str,str] = {
    "ameliyat":"عملیات", "hayat":"حیات", "kaybeden":"غائب ایدن",
}
LEXICALIZED_NOMINAL_SUFFIX_OVERRIDES: dict[str,str] = {
    "ı":"ی","i":"ی","u":"ی","ü":"ی",
    "ını":"نی","ini":"نی","unu":"نی","ünü":"نی",
    "ının":"نڭ","inin":"نڭ","unun":"نڭ","ünün":"نڭ",
    "ına":"نە","ine":"نە","una":"نە","üne":"نە",
    "ından":"ندن","inden":"ندن","undan":"ندن","ünden":"ندن",
    "lar":"لر","ler":"لر","lara":"لرە","lere":"لرە",
}

# ── LIK-drop surface overrides ───────────────────────────────────────────────
LIK_DROP_STEM_OVERRIDES: dict[str,str] = {
    "bilge":"بیلگە","hakim":"حاكم","lider":"لیدر","değişik":"دگیشیك",
    "beraber":"برابر","bakan":"باقان","çöp":"چوپ","genç":"گنچ",
    "müdür":"مدير","önder":"اوندهر","özgür":"اوزگور","sağ":"صاغ",
}
LIK_DROP_STEM_SUFFIX_OVERRIDES: dict[tuple,str] = {
    ("hasta","lığı"):"خستەلقی",("hasta","lığında"):"خستەلقڭده",
    ("hasta","lığına"):"خستەلقڭه",("hasta","lığından"):"خستەلقندن",
    ("nite","liği"):"نیتهلیغی",("nite","liğinde"):"نیتهلیغینده",
    ("nite","liğine"):"نیتهلیغینه",("nite","liğinden"):"نیتهلیغیندن",
    ("nite","liğindeki"):"نیتهلیغیندهکی",
    ("sağ","lığında"):"صاغلقڭده",("sağ","lığına"):"صاغلقڭه",
    ("sağ","lığından"):"صاغلقندن",
    ("savcı","lığından"):"صاوجیلغندن",
    ("başsavcı","lığından"):"باشصاوجیلغندن",
}
LIK_DROP_SURFACE_OVERRIDES: dict[str,str] = {
    "liği":"لگی","lığı":"لگی","luğu":"لگی","lüğü":"لگی",
    "liğe":"لگە","lığa":"لگە","luğa":"لگە","lüğe":"لگە",
    "liğinde":"لگینده","lığında":"لگینده","luğunda":"لگینده","lüğünde":"لگینده",
    "liğine":"لگینه","lığına":"لگنە","luğuna":"لگنە","lüğüne":"لگنە",
    "liğinden":"لگیندن","lığından":"لگیندن","luğundan":"لگیندن","lüğünden":"لگیندن",
    "liğindeki":"لگیندهکی","lığındaki":"لگیندهکی","luğundaki":"لگیندهکی","lüğündeki":"لگیندهکی",
}

# ── Related-adjective overrides ──────────────────────────────────────────────
RELATED_ADJ_STEM_OVERRIDES: dict[str,str] = {
    "para":"پارە", "katılım":"قاتیلیم",
}

# ── Equative -ca/-ce normalizer ──────────────────────────────────────────────
def normalize_ca_stem_ottoman(stem: str, stem_ottoman: str) -> str:
    st = normalize_ottoman_lookup_value(stem_ottoman)
    if st.startswith("اە"): st = "ا" + st[2:]
    if stem.endswith(("ma","me")):
        if st.endswith(("ما","مه","مق","مک")): st = st[:-1] + "ە"
        if stem and stem[0] not in _TUM_UNLULER and len(st)>=2 and st[1]=="ە":
            st = st[0] + st[2:]
    if (stem.endswith(("an","en","ın","in","un","ün"))
            and len(st)>=2 and st[-1]=="ن" and st[-2] in "اەیو"):
        st = st[:-2] + "ن"
    return st

# ── Loanword post-processing ─────────────────────────────────────────────────
def normalize_auto_loanword_surface(token: str, ottoman: str) -> str:
    lw = lower_tr(token or "")
    if lw.startswith("zayıf") and ottoman.startswith("زاییف"):
        return "ضعیف" + ottoman[len("زاییف"):]
    return ottoman

def normalize_latin_terminal_s(token: str, ottoman: str) -> str:
    if re.fullmatch(r"[A-Za-z]+", token or "") and lower_tr(token).endswith("s") and ottoman.endswith(("ش","ص")):
        return ottoman[:-1] + "س"
    return ottoman

# ── Question particle suffix list (for concatenated-question recovery) ───────
QUESTION_PARTICLE_SUFFIXES = (
    "mıyım","miyim","muyum","müyüm",
    "mıyız","miyiz","muyuz","müyüz",
    "mısın","misin","musun","müsün",
    "mısınız","misiniz","musunuz","müsünüz",
)

# ══════════════════════════════════════════════════════════════════════════════
# §9  TRANSLITERATOR CLASS
# ══════════════════════════════════════════════════════════════════════════════
class OttomanTransliterator:
    """Thread-safe Ottoman transliteration engine. Instantiate once at startup."""

    def __init__(self, lookup_file:str="manual_lookup.tsv",
                 abbrev_file:str="abbrev_lookup.tsv", historical:bool=True)->None:
        self.historical=historical
        self._lookup=self._load_tsv(lookup_file); self._abbrev=self._load_tsv(abbrev_file)
        self._lookup_folded:dict[str,tuple]={}
        for k,v in self._lookup.items():
            f=fold_tr(k)
            if f and f not in self._lookup_folded: self._lookup_folded[f]=(k,v)
        _get_module_analyzer()

    # ── public ────────────────────────────────────────────────────────────
    def transliterate(self, text:str)->"TransliterationResult":
        tokens:list[dict]=[]; ot_parts:list[str]=[]; sources:list[str]=[]
        for tok,ot,src,dbg in self._tokenize(text):
            ot_parts.append(ot)
            if src!="whitespace":
                tokens.append({"token":tok,"ottoman":ot,"source":src,"debug":dbg})
                sources.append(src)
        ottoman="".join(ot_parts)
        scores=[SCORE_MAP.get(s,0.0) for s in sources]
        confidence=round(sum(scores)/len(scores),3) if scores else 1.0
        return TransliterationResult(turkish=text,ottoman=ottoman,confidence=confidence,tokens=tokens)

    # ── tokenization ──────────────────────────────────────────────────────
    def _tokenize(self, line:str):
        cursor=0
        for match in _TOKEN_RE.finditer(line):
            start,end=match.span()
            if start>cursor:
                gap=line[cursor:start]; yield gap,gap,"whitespace",gap
            token=match.group(0)
            if not is_word_token(token):
                yield token,convert_ottoman_punctuation(token),"punct",token
            else:
                ot,src,dbg=self._transliterate_token(token); yield token,ot,src,dbg
            cursor=end
        if cursor<len(line):
            tail=line[cursor:]; yield tail,tail,"whitespace",tail

    def _render_lexicalized_nominal_surface(self, token:str):
        word=lower_tr(token)
        for stem,stem_ot in sorted(LEXICALIZED_NOMINAL_STEM_OVERRIDES.items(),key=lambda x:-len(x[0])):
            if not word.startswith(stem): continue
            suffix=word[len(stem):]
            if not suffix: return stem_ot,stem,""
            suffix_ot=(LEXICALIZED_NOMINAL_SUFFIX_OVERRIDES.get(suffix) or
                       OTTOMAN_SURFACE_OVERRIDES.get(suffix) or
                       render_ottoman(suffix,self.historical))
            return stem_ot+suffix_ot,stem,suffix
        return None

    def _render_lik_drop_surface(self, token:str):
        word=lower_tr(token)
        for suffix,suffix_ot in sorted(LIK_DROP_SURFACE_OVERRIDES.items(),key=lambda x:-len(x[0])):
            if not word.endswith(suffix) or len(word)<=len(suffix): continue
            stem=token[:-len(suffix)]; stem_lw=lower_tr(stem)
            override=LIK_DROP_STEM_SUFFIX_OVERRIDES.get((stem_lw,suffix))
            if override is not None: return override,stem,suffix
            stem_ot=LIK_DROP_STEM_OVERRIDES.get(stem_lw)
            if stem_ot is None:
                r=self._transliterate_token(stem)
                if r[1]=="missing": continue
                stem_ot=r[0]
            return stem_ot+suffix_ot,stem,suffix
        return None

    def _render_related_adj_surface(self, token:str):
        word=lower_tr(token)
        if not word.endswith(("sal","sel")) or len(word)<=3: return None
        for parse in self._flatten(token):
            if "Related" not in (list(getattr(parse,"morphemes",[]) or [])): continue
            stem=token[:-3]; stem_lw=lower_tr(stem)
            stem_ot=RELATED_ADJ_STEM_OVERRIDES.get(stem_lw)
            if stem_ot is None:
                found=self._lookup_root_entry(stem)
                stem_ot=found[0] if found else render_ottoman(stem,self.historical)
            return stem_ot+"سال",stem,word[-3:]
        return None

    def _render_ca_surface(self, token:str):
        word=lower_tr(token)
        for suffix in ("ça","çe","ca","ce"):
            if not word.endswith(suffix) or len(word)<=len(suffix): continue
            stem=token[:-len(suffix)]; stem_lw=lower_tr(stem)
            if not stem_lw.endswith(("ma","me","n")): continue
            found=self._lookup_word(stem)
            stem_ot=found[0] if found[0] else render_ottoman(stem,self.historical)
            stem_ot=normalize_ca_stem_ottoman(stem_lw,stem_ot)
            if stem_ot: return stem_ot+"جە",stem,suffix
        return None

    def _render_concatenated_question_particle(self, token:str):
        """Recover tokens where a question particle was written without a space."""
        word=lower_tr(token)
        for suffix in QUESTION_PARTICLE_SUFFIXES:
            if not word.endswith(suffix) or len(word)<=len(suffix): continue
            stem=token[:-len(suffix)]
            stem_ot,stem_src,_=self._transliterate_token(stem)
            # Only use the split if the stem resolved properly
            if stem_src in {"missing"}: continue
            suffix_ot,_,_=self._transliterate_token(suffix)
            return stem_ot+suffix_ot, stem, suffix
        return None

    def _transliterate_token(self, token:str)->tuple[str,str,str]:
        analysis_word=token.replace("'","").replace("\u2019","")
        imperative_like_suffixes=(
            "sın","sin","sun","sün",
            "sınız","siniz","sunuz","sünüz",
            "sınlar","sinler","sunlar","sünler",
        )
        # 1. Hard override
        if token in WORD_OVERRIDES: return WORD_OVERRIDES[token],"override",token
        lw=lower_tr(token)
        if lw=="belirt": return "بلیرت","override",token
        if lw in WORD_OVERRIDES: return WORD_OVERRIDES[lw],"override",token

        # 2. Apostrophe split
        norm=token.replace("\u2019","'").replace("\u2018","'")
        if "'" in norm:
            base,suffix=norm.split("'",1)
            if base.isdigit() and suffix:
                base_ot=base.translate(str.maketrans("0123456789","٠١٢٣٤٥٦٧٨٩"))
                return base_ot+self._render_numeric_apostrophe_suffix(suffix),"override",token
            found=self._lookup_root_entry(base) or self._lookup_root_entry(lower_tr(base))
            if found and lower_tr(suffix) in _APOSTROPHE_SUFFIXES:
                return found[0]+_render_apostrophe_suffix(lower_tr(suffix)),"override",token
            if base and suffix and lower_tr(suffix) in _APOSTROPHE_SUFFIXES:
                base_ot=render_ottoman(base,self.historical)
                return base_ot+_render_apostrophe_suffix(lower_tr(suffix)),"surface",f"{base}+'{suffix}"

        # 2b. Inline numeral+suffix (7lik → ٧لك)
        num_match=re.fullmatch(rf"(\d+)([{_TR_WORD_CHARS}]+)",norm)
        if num_match:
            base_n,sfx_n=num_match.groups(); sfx_lw=lower_tr(sfx_n)
            if sfx_lw in _APOSTROPHE_SUFFIXES or sfx_lw in {"lık","lik","luk","lük"}:
                base_ot=base_n.translate(str.maketrans("0123456789","٠١٢٣٤٥٦٧٨٩"))
                return base_ot+self._render_numeric_apostrophe_suffix(sfx_lw),"override",token

        # 2c. Concatenated question particle
        cq=self._render_concatenated_question_particle(token)
        if cq: return cq[0],"surface",f"{cq[1]} + {cq[2]}"

        # 2d. Lexicalized nominal surface
        lex=self._render_lexicalized_nominal_surface(token)
        if lex: return lex[0],"surface",(f"{lex[1]} + {lex[2]}" if lex[2] else lex[1])

        # 2e. LIK-drop surface
        lik=self._render_lik_drop_surface(token)
        if lik: return lik[0],"surface",f"{lik[1]} + {lik[2]}"

        # 2f. Related-adjective surface
        rel=self._render_related_adj_surface(token)
        if rel: return rel[0],"surface",f"{rel[1]} + {rel[2]}"

        # 3. Imperative-like surfaces before dict lookup
        selected=None
        if lw.endswith(imperative_like_suffixes):
            selected=self._select_parse(analysis_word)
            if selected:
                imp=self._generate_imperative(selected,analysis_word)
                if imp: return imp["result"],"tags",f"{selected['lemma']} :: IMP"

        # 4. Surface verb / future-chain fallbacks
        sfb=self._generate_surface_verb_fallback(token)
        if sfb: return sfb[0],"surface",f"surface_verb:{sfb[1]}+{sfb[2]}"
        sfc=self._generate_surface_future_chain_fallback(token)
        if sfc: return sfc[0],"surface",f"surface_future:{sfc[1]}+{sfc[2]}"

        # 5. Dictionary / number lookup
        direct,src,form=self._lookup_word(token)
        if direct is not None:
            direct=normalize_latin_terminal_s(token,direct)
            return direct,src,form

        # 5b. CA-surface after dict to avoid false positives
        ca=self._render_ca_surface(token)
        if ca: return ca[0],"surface",f"{ca[1]} + {ca[2]}"

        # 6. English
        if is_likely_english(token):
            eng_ot=render_english_ottoman(token)
            eng_ot=normalize_latin_terminal_s(token,eng_ot)
            return eng_ot,"english",token

        # 7. Zeyrek morphological analysis
        if selected is None:
            selected=self._select_parse(analysis_word)
        if selected:
            imp=self._generate_imperative(selected,analysis_word)
            if imp: return imp["result"],"tags",f"{selected['lemma']} :: IMP"
            pp=self._generate_present_participle(selected,analysis_word)
            if pp: return pp["result"],"tags",f"{selected['lemma']} :: PRES_PART"
            pc=self._generate_participle_copula(selected,analysis_word)
            if pc: return pc["result"],"tags",f"{selected['lemma']} :: PART_COP"
            pred=self._generate_predicative_inf(selected,analysis_word)
            if pred: return pred["result"],"tags",f"{selected['lemma']} :: PRED_INF"
            gen=self._generate(selected["root_ot"],selected["tags"],
                               selected["surface_root"],lemma=selected.get("lemma"))
            if gen:
                sfx=" + ".join(p["surface"] for p in gen["suffixes"])
                return gen["result"],"tags",f"{selected['lemma']} :: {'+'.join(selected['tags'])} :: {selected['surface_root']}+{sfx}"

        # 8. Auto fallback (no brackets)
        parses=self._flatten(analysis_word)
        if parses:
            pos=str(getattr(parses[0],"pos",""))
            if not any(x in pos for x in ("Prop","Abbrv","Unk")):
                auto_ot=render_ottoman(token,False)
                return normalize_latin_terminal_s(token,normalize_auto_loanword_surface(token,auto_ot)),"auto",token
        auto_ot=render_ottoman(token,False)
        return normalize_latin_terminal_s(token,normalize_auto_loanword_surface(token,auto_ot)),"auto",token

    # ── dictionary helpers ────────────────────────────────────────────────
    @staticmethod
    def _load_tsv(filepath:str)->dict[str,str]:
        data:dict[str,str]={}
        if not os.path.exists(filepath): return data
        with open(filepath,encoding="utf-8-sig") as f:
            for row in csv.reader(f,delimiter="\t"):
                if len(row)>=2 and not row[0].startswith("#"):
                    k,v=row[0].strip(),row[1].strip()
                    if k.lower()=="word" and v.lower()=="ottoman": continue
                    v=normalize_ottoman_lookup_value(v)
                    data[k]=v; data[lower_tr(k)]=v
        return data

    def _lookup_root_entry(self, key:str):
        if not key: return None
        for cand,score in [(key,32),(lower_tr(key),28)]:
            if cand in WORD_OVERRIDES: return WORD_OVERRIDES[cand],cand,score
        for cand,score in [(key,30),(lower_tr(key),24)]:
            if cand in self._lookup: return self._lookup[cand],cand,score
        f=fold_tr(key)
        if f in self._lookup_folded:
            mk,mv=self._lookup_folded[f]
            return mv,lower_tr(mk),18
        return None

    def _lookup_word(self, word:str):
        if all(c.isdigit() or c in ".,-" for c in word) and any(c.isdigit() for c in word):
            return word.translate(str.maketrans("0123456789","٠١٢٣٤٥٦٧٨٩")),"exact",word
        if word in self._lookup: return self._lookup[word],"exact",word
        lw=lower_tr(word)
        if lw in self._lookup: return self._lookup[lw],"exact",lw
        f=fold_tr(word)
        if f in self._lookup_folded:
            _,fv=self._lookup_folded[f]
            return fv,"exact",lw
        return None,None,None

    def _render_numeric_apostrophe_suffix(self, suffix:str)->str:
        normalized=lower_tr(suffix)
        overrides={
            "da":"دە","de":"دە","ta":"دە","te":"دە","a":"ە","e":"ە",
            "ya":"یه","ye":"یە",
            "ı":"ی","i":"ی","u":"ی","ü":"ی",
            "lık":"لیق","lik":"ليك","luk":"لوق","lük":"لوك",
            "sı":"سی","si":"سی","su":"سی","sü":"سی",
            "nı":"نی","ni":"نی","nu":"نی","nü":"نی",
            "ını":"نی","ini":"نی","unu":"نی","ünü":"نی",
            "sını":"سنی","sini":"سنی","sunu":"سنی","sünü":"سنی",
            "ına":"نه","ine":"نه","una":"نه","üne":"نه",
            "sına":"سنه","sine":"سنه","suna":"سنه","süne":"سنه",
            "ından":"ندن","inden":"ندن","undan":"ندن","ünden":"ندن",
            "sından":"سندن","sinden":"سندن","sundan":"سندن","sünden":"سندن",
            "ıydı":"یدی","iydi":"یدی","uydu":"یدی","üydü":"یدی",
            "sıydı":"سیدی","siydi":"سیدی","suydu":"سیدی","süydü":"سیدی",
            "ındaydı":"ندەیدی","indeydi":"یندەیدی","undaydı":"ندەیدی","ündeydi":"یندەیدی",
            "sındaydı":"سندەیدی","sindeydi":"سیندەیدی","sundaydı":"سندەیدی","sündeydi":"سیندەیدی",
            "ındeymiş":"یندەیمش","indeymiş":"یندەیمش","undaymış":"ندەیمش","ündeymiş":"یندەیمش",
        }
        return overrides.get(normalized) or OTTOMAN_SURFACE_OVERRIDES.get(normalized) or \
               render_ottoman(normalized,self.historical)

    # ── Zeyrek ───────────────────────────────────────────────────────────
    def _analyze(self, word:str): return _cached_zeyrek_analyze(normalize_tr_text(word))
    def _flatten(self, word:str)->list: return [p for grp in self._analyze(word) for p in grp]

    def _parse_pos(self, parse)->str:
        fmt=getattr(parse,"formatted","") or ""
        m=re.match(r"\[([^:]+):([^\],]+)",fmt)
        return m.group(2).strip().upper() if m else str(getattr(parse,"pos","")).upper()

    def _surface_root(self, parse)->str:
        fmt=getattr(parse,"formatted","") or ""
        if "] " in fmt:
            tail=fmt.split("] ",1)[1]; seg=re.split(r"[+|]",tail)[0]
            if ":" in seg: return lower_tr(seg.split(":",1)[0].strip())
        lemma=lower_tr(getattr(parse,"lemma","") or "")
        return lemma[:-3] if lemma.endswith(("mak","mek")) else lemma

    def _resolve_root(self, parse):
        lemma=getattr(parse,"lemma",None)
        if not lemma or lemma=="Unk": return None
        base_pos=self._parse_pos(parse); sr=self._surface_root(parse)
        cands=[]
        if base_pos=="VERB" and sr: cands.append(sr)
        cands.extend([lemma,lower_tr(lemma)])
        if base_pos=="VERB" and lower_tr(lemma).endswith(("mak","mek")): cands.append(lower_tr(lemma)[:-3])
        if sr and sr not in cands: cands.append(sr)
        seen:set=set()
        for cand in cands:
            if not cand or cand in seen: continue
            seen.add(cand)
            found=self._lookup_root_entry(cand)
            if found:
                root_ot,dict_form,ls=found
                # Skip folded VERB matches where the dict form doesn't match the candidate
                if base_pos=="VERB" and ls<=18 and lower_tr(dict_form)!=lower_tr(cand):
                    continue
                return {"root_ot":root_ot,"lemma":lemma,"dict_form":dict_form,
                        "lookup_score":ls,"base_pos":base_pos,"surface_root":sr or lower_tr(cand)}
        # Verb fallback: synthesise root from surface form when no dict entry found
        if base_pos=="VERB" and sr:
            root_ot=render_ottoman(sr,historical=self.historical)
            return {"root_ot":root_ot,"lemma":lemma,"dict_form":lower_tr(lemma),
                    "lookup_score":0,"base_pos":base_pos,"surface_root":sr}
        # EQU/Ly/AsIf/Related fallback
        morphemes=list(getattr(parse,"morphemes",[]) or [])
        if any(m in morphemes for m in ("Equ","Ly","AsIf","Related")) and sr:
            root_ot=render_ottoman(sr,historical=self.historical)
            return {"root_ot":root_ot,"lemma":lemma,"dict_form":lower_tr(lemma),
                    "lookup_score":0,"base_pos":base_pos,"surface_root":sr}
        return None

    def _zeyrek_tags(self, parse)->list[str]:
        base_pos=self._parse_pos(parse); morphemes=list(getattr(parse,"morphemes",[]) or [])
        tags=[base_pos]; nominal_ctx=base_pos in {"NOUN","ADJ","PRON","NUM","QUES"}
        verbal_ctx=base_pos=="VERB"; sr=self._surface_root(parse)
        prev_raw=morphemes[0] if morphemes else None
        for index,raw in enumerate(morphemes[1:],start=1):
            next_raw=morphemes[index+1] if index+1<len(morphemes) else None
            if raw=="Zero": prev_raw=raw; continue
            if raw in {"Noun","Adj","Adv","Pron","Num"}:
                nominal_ctx=True; verbal_ctx=False; prev_raw=raw; continue
            if raw=="Verb":
                if not verbal_ctx and prev_raw=="Zero": tags.append("COPULA")
                elif not verbal_ctx: tags[0]="VERB"
                verbal_ctx=True; nominal_ctx=False; prev_raw=raw; continue
            if raw=="A3pl":
                if nominal_ctx and next_raw=="Become": prev_raw=raw; continue
                tags.append("PLURAL" if nominal_ctx else "A3PL"); prev_raw=raw; continue
            if raw=="A3sg" and nominal_ctx and next_raw=="Become": prev_raw=raw; continue
            if raw=="A3sg" and not nominal_ctx: tags.append("A3SG"); prev_raw=raw; continue
            if raw=="Recip" and sr.endswith("ş"): prev_raw=raw; continue
            n=ZEYREK_TAG_MAP.get(raw)
            if n: tags.append(n)
            prev_raw=raw
        return fuse_tag_strings(tags)

    def _score(self, parse, ntags:list[str], ri:dict, amb:int)->float:
        score=float(ri["lookup_score"])
        score+=40 if self._validate_tags(ntags) else -80
        base_pos=ri["base_pos"]; actual_pos=str(getattr(parse,"pos","")).upper()
        if base_pos==actual_pos: score+=10
        elif base_pos=="VERB" and actual_pos in {"ADJ","ADV"} and len(ntags)==1: score-=24
        if "PROP" in (getattr(parse,"formatted","") or "").upper(): score-=20
        deriv=sum(1 for t in ntags if t in VERBAL_DERIVATION_TAGS or t in NOMINAL_DERIVATION_TAGS)
        non_d=[t for t in ntags if t not in VERBAL_DERIVATION_TAGS and t not in NOMINAL_DERIVATION_TAGS]
        score-=deriv*4; score-=max(len(non_d)-3,0); score-=max(amb-1,0)*2
        sw=lower_tr(getattr(parse,"word","") or ""); morphemes=list(getattr(parse,"morphemes",[]) or [])
        if sw.endswith(("sa","se")): score+=6 if "COND" in ntags else -6
        if sw.endswith(("mış","miş","muş","müş")): score+=10 if "NARR" in ntags else -10
        if ("Dim" in morphemes
                and sw.endswith(("ceğiz","cağız","ceğim","cağım","ceksin","caksın","ceksiniz","caksınız"))):
            score-=24
        if ("Fut" in morphemes
                and any(t in ntags for t in {"A1SG","A1PL","A2SG","A2PL"})
                and sw.endswith(("ceğim","cağım","ceğiz","cağız","ceksin","caksın","ceksiniz","caksınız"))):
            score+=18
        if "ByDoingSo" in morphemes: score+=32 if ri["base_pos"]=="VERB" else -32
        if "Recip" in morphemes and ri["surface_root"].endswith("ş"): score+=10
        if ntags[-2:]==["P3SG","DAT"] and sw.endswith(("ına","ine","una","üne")): score+=1
        if ntags[-2:]==["P3SG","ACC"] and sw.endswith(("ını","ini","unu","ünü")): score+=1
        if ntags[-2:]==["P3SG","ABL"] and sw.endswith(("ından","inden","undan","ünden")): score+=1
        if ntags[-2:]==["P3SG","LOC"] and sw.endswith(("ında","inde","unda","ünde","nda","nde")): score+=1
        if ntags[-2:]==["P3SG","GEN"] and sw.endswith(("ının","inin","unun","ünün")): score+=1
        if ri["dict_form"]==lower_tr(getattr(parse,"lemma","") or ""): score+=6
        return score

    def _validate_tags(self, tags:list[str])->bool:
        root_pos=tags[0] if tags else ROOT
        if root_pos=="QUES":
            if not all(t in PERSON_TAGS|{"A3SG"} for t in tags[1:]): return False
        if root_pos in {"NOUN","ADJ","PRON","NUM"}:
            seen_cop=False
            for tag in tags[1:]:
                if tag=="COPULA": seen_cop=True; continue
                if tag in VOICE_TAGS or tag in {"NEG","PROG","FUTURE","AOR","OPT","ABLE","UNABLE",
                                                 "CONV_AFTER","CONV_BY","CONV_SINCE","CONV_ASLONGAS","CONV_WHILE"}: return False
                if tag in {"PAST","NARR","COND","NECES"} and not seen_cop: return False
                if tag in EMPTY_TAGS: continue
                if tag in PERSON_TAGS and not seen_cop and tag not in {"A3SG","A3PL"}: return False
        if root_pos=="VERB":
            nominalized=False
            for tag in tags[1:]:
                if tag in {"INF1","INF2","PART","PAST_PART","CONV","CONV_AFTER","CONV_BY",
                           "CONV_SINCE","CONV_ASLONGAS","CONV_WHILE","FUT_PART"}:
                    nominalized=True; continue
                if tag in NOMINAL_DERIVATION_TAGS: nominalized=True; continue
                if (tag in POSSESSIVE_TAGS|CASE_TAGS|{"PLURAL"}
                        and not nominalized and "COPULA" not in tags and tag!="A3PL"):
                    return False
        current=ROOT
        for tag in tags:
            if tag in ROOT_POS_TAGS or tag in EMPTY_TAGS: continue
            target=TAG_TO_STATE.get(tag)
            if target is None: return False
            if target!=current and target not in TAG_FSM.get(current,[]): return False
            current=target
        return True

    def _select_parse(self, word:str):
        parses=self._flatten(word); cands=[]
        for parse in parses:
            ri=self._resolve_root(parse)
            if not ri: continue
            nt=self._zeyrek_tags(parse)
            cands.append((self._score(parse,nt,ri,len(parses)),parse,ri,nt))
        if not cands: return None
        cands.sort(key=lambda x:(x[0],-len(x[3]),len(x[2]["surface_root"])),reverse=True)
        _,bp,ri,tags=cands[0]
        return {"parse":bp,"root_ot":ri["root_ot"],"lemma":ri["lemma"],
                "dict_form":ri["dict_form"],"base_pos":ri["base_pos"],
                "surface_root":ri["surface_root"],"tags":tags}

    # ── generation ────────────────────────────────────────────────────────
    def _generate(self, root_ot:str, tags:list[str], root_surface:str, lemma:Optional[str]=None):
        ntags=fuse_tag_strings(tags)
        if not self._validate_tags(ntags): return None
        # Bare imperative: VERB + A2SG → just the stripped root
        if ntags==["VERB","A2SG"]:
            return {"surface_root":root_surface,"suffixes":[],"result":strip_infinitive_from_ottoman(root_ot)}
        morphs=build_underlying_morphs(ntags)
        realized_root,realized_sfx=realize_allomorphs(root_surface,morphs)
        rendered_sfx=[
            {"tag":p["tag"],"surface":p["surface"],
             "ottoman":render_suffix_ottoman(p["tag"],p["surface"],self.historical)}
            for p in realized_sfx
        ]
        rendered_sfx=adjust_past_person_rendering(rendered_sfx)
        if ntags and ntags[0]=="VERB" and rendered_sfx:
            stripped=strip_infinitive_from_ottoman(root_ot)
            if lower_tr(lemma or "")=="belirtmek":
                stripped=strip_infinitive_from_ottoman("بلیرتمك")
            if rendered_sfx[0].get("tag") in NOMINAL_DERIVATION_TAGS:
                root_ot=stripped
            elif normalize_surface_ascii(realized_root)==normalize_surface_ascii(root_surface):
                root_ot=stripped
            else:
                alt=WORD_OVERRIDES.get(realized_root) or render_ottoman(realized_root,self.historical)
                root_ot=alt if alt!=stripped else stripped
            aux_root_ot,rendered_sfx=adjust_etmek_auxiliary_output(lemma,realized_root,rendered_sfx)
            if aux_root_ot: root_ot=aux_root_ot
            elif (rendered_sfx and rendered_sfx[0].get("tag")=="PASSIVE"
                    and normalize_surface_ascii(rendered_sfx[0].get("surface",""))=="n"
                    and normalize_surface_ascii(realized_root).endswith(("la","le"))
                    and root_ot.endswith(("ه","ە"))):
                root_ot=root_ot[:-1]
            elif (rendered_sfx
                    and normalize_surface_ascii(rendered_sfx[0].get("surface","")) in {"yor","ıyor","iyor","uyor","üyor"}
                    and root_ot.endswith(("لا","لە","له"))):
                root_ot=root_ot[:-1]
        elif (ntags and ntags[0] in {"NOUN","ADJ","PRON","NUM"} and rendered_sfx and lemma
                and normalize_surface_ascii(realized_root)!=normalize_surface_ascii(lemma)):
            root_ot=adjust_softened_nominal_root_ottoman(root_ot,lemma,realized_root)
        return {"surface_root":realized_root,"suffixes":rendered_sfx,
                "result":merge_ottoman(root_ot,rendered_sfx)}

    def _generate_predicative_inf(self, sel:dict, word:str):
        morphemes=list(getattr(sel["parse"],"morphemes",[]) or [])
        if not all(x in morphemes for x in ("Inf1","Pres","Cop")): return None
        lemma=lower_tr(sel.get("lemma") or ""); w=lower_tr(word or "")
        if not lemma or not w.startswith(lemma): return None
        cop_surface=w[len(lemma):]
        if not cop_surface: return None
        return {"result":sel["root_ot"]+render_ottoman(cop_surface,self.historical)}

    def _generate_participle_copula(self, sel:dict, word:str):
        morphemes=list(getattr(sel["parse"],"morphemes",[]) or [])
        if "PresPart" not in morphemes or "Cop" not in morphemes: return None
        root_surface=sel.get("surface_root") or lower_tr(sel.get("lemma") or "")
        w=lower_tr(word or "")
        part_surface=("y" if ends_with_vowel(root_surface) else "")+choose_harmony_A(root_surface)+"n"
        expected=root_surface+part_surface
        if not w.startswith(expected): return None
        cop_surface=w[len(expected):]
        if not cop_surface: return None
        root_ot=strip_infinitive_from_ottoman(sel["root_ot"])
        part_ot=OTTOMAN_SURFACE_OVERRIDES.get(part_surface) or render_ottoman(part_surface,self.historical)
        cop_ot=("دیر" if normalize_surface_ascii(cop_surface) in
                         {"dır","dir","dur","dür","tır","tir","tur","tür"}
                else render_suffix_ottoman("COPULA_ASSERT",cop_surface,self.historical))
        return {"result":root_ot+part_ot+cop_ot}

    def _generate_present_participle(self, sel:dict, word:str):
        morphemes=list(getattr(sel["parse"],"morphemes",[]) or [])
        if "PresPart" not in morphemes or "Cop" in morphemes: return None
        w=lower_tr(word or "")
        root_surface=sel.get("surface_root") or lower_tr(sel.get("lemma") or "")
        if not w.startswith(root_surface): return None
        part_surface=w[len(root_surface):]
        if not part_surface: return None
        root_ot=strip_infinitive_from_ottoman(sel["root_ot"])
        part_ot=OTTOMAN_SURFACE_OVERRIDES.get(part_surface) or render_ottoman(part_surface,self.historical)
        return {"result":root_ot+part_ot}

    def _generate_imperative(self, sel:dict, word:str):
        morphemes=list(getattr(sel["parse"],"morphemes",[]) or [])
        if "Imp" not in morphemes: return None
        root_surface=sel.get("surface_root") or lower_tr(sel.get("lemma") or "")
        w=lower_tr(word or "")
        if not w.startswith(root_surface): return None
        suffix_surface=w[len(root_surface):]
        if not suffix_surface: return None
        root_ot=strip_infinitive_from_ottoman(sel["root_ot"])
        # Imperative-specific overrides (different from regular sın/siniz)
        imp_overrides={
            "sın":"سین","sin":"سین","sun":"سین","sün":"سین",
            "sınız":"سینز","siniz":"سینز","sunuz":"سینز","sünüz":"سینز",
            "sınlar":"سینلر","sinler":"سینلر","sunlar":"سینلر","sünler":"سینلر",
        }
        suffix_ot=imp_overrides.get(suffix_surface) or \
                  OTTOMAN_SURFACE_OVERRIDES.get(suffix_surface) or \
                  render_ottoman(suffix_surface,self.historical)
        return {"result":root_ot+suffix_ot}

    def _generate_surface_verb_fallback(self, token:str):
        word=lower_tr(token)
        for suffix,suffix_ot in sorted(SURFACE_VERB_FALLBACK_SUFFIXES.items(),key=lambda x:-len(x[0])):
            if not word.endswith(suffix) or len(word)<=len(suffix): continue
            stem=word[:-len(suffix)]
            root_entry=self._lookup_root_entry(stem)
            sel=self._select_parse(stem)
            if not root_entry and not sel: continue
            if not any(self._parse_pos(p)=="VERB" for p in self._flatten(stem)): continue
            root_ot=strip_infinitive_from_ottoman((root_entry[0] if root_entry else sel["root_ot"]))
            return root_ot+suffix_ot, stem, suffix
        return None

    def _generate_surface_future_chain_fallback(self, token:str):
        word=lower_tr(token)
        future_bases=("acak","ecek","yacak","yecek")
        for suffix,suffix_ot in sorted(SURFACE_FUTURE_CHAIN_SUFFIXES.items(),key=lambda x:-len(x[0])):
            if not word.endswith(suffix) or len(word)<=len(suffix): continue
            future_form=word[:-len(suffix)]
            if not any(future_form.endswith(b) for b in future_bases): continue
            future_ot,future_src,_=self._transliterate_token(future_form)
            if future_src=="auto" and future_ot.startswith("["): continue
            return future_ot+suffix_ot, future_form, suffix
        return None

# ══════════════════════════════════════════════════════════════════════════════
# §10  RESULT MODEL
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class TransliterationResult:
    turkish:    str
    ottoman:    str
    confidence: float
    tokens:     list[dict] = field(default_factory=list)
    def to_dict(self)->dict:
        return {"turkish":self.turkish,"ottoman":self.ottoman,
                "confidence":self.confidence,"tokens":self.tokens}
