"""PSA framework classifier — FROZEN.

Scoring classifier calibrated against the lecturer's applied standard from
"PSA FRAMEWORK.pdf" (methodology: reports/framework_audit.md). The regexes
and scoring rules below are a frozen contract: do not tune them without
updating the methodology documentation and the must-pass reference examples
in tests/test_psa_classify.py.

Scoring (all regexes Python ``re`` with ``re.I``):
    KEEP_SIG match +2; first word in IMPER_START +2;
    "To <verb>..." start WITH KEEP_SIG hit +1; AUDIENCE +1;
    CONNECTIVE start -1; PRESS_NEWS -2; LEGAL3 -3;
    ENCYC -2 but only when score<2 before applying.

Label: score>=2 -> "PSA"; elif LEGAL3 and score<0 -> "Legal";
    elif PRESS_NEWS and score<=0 -> "PressRelease"; else "Informational";
    empty text -> ("Drop", -9).
"""

import re

import pandas as pd

KEEP_SIG = r"\b(advise[ds]?|urge[sd]?|warn(?:s|ed|ing)?|remind(?:s|ed)?|avoid|prevent|protect|report (?:any|all|to|suspicious)|call (?:our|the|\d)|hotline|toll[- ]free|dial \d|sms (?:the word|to)|visit (?:your|our|the nearest)|register|apply|pay (?:your|before|by)|file (?:your|returns)|deadline|alert|be alert|vigilant|stay (?:safe|home|calm|healthy)|wash your|sanitize|keep (?:your|children|safe)|seek (?:medical|immediate|help)|get (?:vaccinated|tested|screened|your)|vaccinat|immuniz|(?:public|kenyans?|residents|citizens?) (?:is|are) (?:hereby )?(?:informed|advised|reminded|urged|cautioned|requested)|members of the public|ensure (?:you|that you)|make sure|remember to|please (?:note|be advised|ensure|contact|call|visit)|should|must|necessary to|are (?:encouraged|required|urged|advised|reminded|asked|expected) to|take (?:precaution|care|caution)|precautionary|safety (?:measures|tips|precautions)|will open|now open|is now available|are now available|can now|now (?:access|offers?|provides?)|wishes to inform|hereby (?:informs?|notifies)|invites? applications?|calls? for applications?|free (?:services?|screening|testing|vaccination|clinic|tuition)|available (?:at|in|from) (?:all|the nearest|your)|remain (?:open|closed)|to remain (?:open|closed)|suspended until|postponed|rescheduled|boil (?:your|water)|use (?:clean|safe|treated)|wear (?:a )?mask|maintain (?:social|physical) distance|let's|let us)\b"
IMPER_START = {'wash','avoid','report','call','visit','register','apply','pay','file','get','keep','stay','protect','seek','ensure','remember','do',"don't",'vaccinate','boil','use','check','confirm','verify','contact','download','dial','sms','text','save','carry','wear','maintain','observe','follow','stop','join','control','reduce','prevent','practice','practise','adopt','plant','harvest','store','handle','test','isolate','quarantine','enroll','enrol','attend','bring','take','give','feed','immunize','immunise','deworm'}
AUDIENCE = r"\b(kenyans?|citizens?|residents|the public|farmers?|parents?|motorists?|drivers?|taxpayers?|applicants?|candidates?|learners?|students?|pupils?|voters?|customers?|passengers?|patients?|traders?|business (?:owners?|community)|all (?:schools|parents|farmers|drivers|employers))\b"
PRESS_NEWS = r"\b(courtesy call|speaking (?:at|during)|during the (?:event|ceremony|launch|visit|occasion)|attended by|earlier today|yesterday|in a statement (?:issued|released)|read in part|media (?:invited|briefing)|photo:|pictured|has said|said the|according to the (?:cs|minister|cabinet secretary|director|ceo)|welcomed by|wins case|emphasized|reiterated|noted that|stated that|added that|said that|the (?:cs|minister|president|governor|director|ceo|chairperson) (?:said|stated|noted|announced|urged|called))\b"
ENCYC = r"\b(is a (?:disease|virus|viral|bacterial|parasitic|species|genus|family of|type of)|is caused by|are caused by|symptoms? (?:of|include)|signs? (?:of|include)|varieties include|refers to|is defined as|is an infection|is a condition|are insects|is a moth|is a weed)\b"
LEGAL3 = r"\b(pursuant to|in exercise of the powers|gazette notice|tender(?:\s+no|\s+ref|\s+document)|it is notified for the general information|expression of interest|request for (?:proposals|quotations))\b"
CONNECTIVE = r"^\s*(however|but|and|so|for example|for instance|according to|in addition|moreover|furthermore|also|then|thus|therefore)\b"

_KEEP_RE = re.compile(KEEP_SIG, re.I)
_AUDIENCE_RE = re.compile(AUDIENCE, re.I)
_PRESS_RE = re.compile(PRESS_NEWS, re.I)
_ENCYC_RE = re.compile(ENCYC, re.I)
_LEGAL_RE = re.compile(LEGAL3, re.I)
_CONNECTIVE_RE = re.compile(CONNECTIVE, re.I)
_TO_VERB_RE = re.compile(r"^\s*to\s+\w+", re.I)

LABELS = ("PSA", "PressRelease", "Legal", "Informational", "Drop")


def classify_psa(text) -> tuple[str, int]:
    """Classify one English text; return (label, score).

    label in {"PSA","PressRelease","Legal","Informational","Drop"};
    empty text -> ("Drop", -9). See module docstring for the frozen rules.
    """
    text = str(text or "").strip()
    if not text:
        return ("Drop", -9)

    score = 0
    keep_hit = bool(_KEEP_RE.search(text))
    if keep_hit:
        score += 2

    first_word = re.sub(r"[^A-Za-z']+", "", text.split()[0]).lower() \
        if text.split() else ""
    if first_word in IMPER_START:
        score += 2

    if keep_hit and _TO_VERB_RE.match(text):
        score += 1

    if _AUDIENCE_RE.search(text):
        score += 1

    if _CONNECTIVE_RE.match(text):
        score -= 1

    press_hit = bool(_PRESS_RE.search(text))
    if press_hit:
        score -= 2

    legal_hit = bool(_LEGAL_RE.search(text))
    if legal_hit:
        score -= 3

    if _ENCYC_RE.search(text) and score < 2:
        score -= 2

    if score >= 2:
        return ("PSA", score)
    if legal_hit and score < 0:
        return ("Legal", score)
    if press_hit and score <= 0:
        return ("PressRelease", score)
    return ("Informational", score)


def classify_frame(df) -> pd.DataFrame:
    """Return a copy of ``df`` with ``psa_class`` and ``psa_score`` columns
    added (classifying the ``English`` column)."""
    df = df.copy()
    results = df["English"].map(classify_psa)
    df["psa_class"] = results.map(lambda t: t[0])
    df["psa_score"] = results.map(lambda t: t[1])
    return df
