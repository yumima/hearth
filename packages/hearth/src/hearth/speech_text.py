"""Text → speakable text, for the TTS engines.

kokoro-onnx phonemizes with espeak, which reads market text badly: a decimal
point is a pause ("2.5%" → "two, five percent"), a thousands comma splits the
number ("130,000" → "one hundred thirty, zero zero zero"), "$1,234.56" comes
out as "dollar one, two hundred thirty-four, fifty-six". Expanding numbers,
money, percentages and the usual finance shorthand into words BEFORE the
phonemizer fixes all of it — and words read with natural phrasing where
digits and symbols read as a list.

Pure Python, no dependencies; `british` gives "one hundred AND thirty".
"""

from __future__ import annotations

import re

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
         "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (10**3, "thousand")]


def _below_thousand(n: int, british: bool) -> str:
    parts = []
    if n >= 100:
        parts.append(f"{_ONES[n // 100]} hundred")
        n %= 100
        if n and british:
            parts.append("and")
    if n >= 20:
        parts.append(_TENS[n // 10] + (f"-{_ONES[n % 10]}" if n % 10 else ""))
    elif n:
        parts.append(_ONES[n])
    return " ".join(parts)


def int_words(n: int, british: bool = False) -> str:
    """130000 → 'one hundred thirty thousand' (British: 'one hundred and thirty thousand')."""
    if n < 0:
        return "minus " + int_words(-n, british)
    if n == 0:
        return "zero"
    parts = []
    for value, name in _SCALES:
        if n >= value:
            parts.append(f"{_below_thousand(n // value, british)} {name}")
            n %= value
    if n:
        # British "one thousand and five": an 'and' before a final part under 100
        if british and parts and n < 100:
            parts.append("and")
        parts.append(_below_thousand(n, british))
    return " ".join(parts)


def number_words(text: str, british: bool = False, keep_trailing_zeros: bool = False) -> str:
    """'7,592.40' → 'seven thousand five hundred ninety-two point four'."""
    text = text.replace(",", "")
    whole, _, frac = text.partition(".")
    out = int_words(int(whole or "0"), british)
    if not keep_trailing_zeros:
        frac = frac.rstrip("0")
    if frac:
        out += " point " + " ".join(_ONES[int(d)] for d in frac)
    return out


def year_words(y: int, british: bool = False) -> str:
    """2026 → 'twenty twenty-six', 2005 → 'two thousand and five' / 'two thousand five'."""
    if 2000 <= y <= 2009:
        return "two thousand" + (" and " if british else " ") + _ONES[y - 2000] if y > 2000 else "two thousand"
    hi, lo = divmod(y, 100)
    if lo == 0:
        return f"{_below_thousand(hi, False)} hundred"
    return f"{_below_thousand(hi, False)} {'oh ' + _ONES[lo] if lo < 10 else _below_thousand(lo, False)}"


_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_MAG_WORD = {"k": "thousand", "thousand": "thousand", "m": "million", "mn": "million", "mm": "million",
             "million": "million", "b": "billion", "bn": "billion", "billion": "billion",
             "t": "trillion", "tn": "trillion", "trillion": "trillion"}
_CURRENCY = {"$": ("dollar", "dollars", "cent", "cents"), "US$": ("dollar", "dollars", "cent", "cents"),
             "£": ("pound", "pounds", "penny", "pence"), "€": ("euro", "euros", "cent", "cents"),
             "¥": ("yen", "yen", "", ""), "₹": ("rupee", "rupees", "", "")}

_RE_CUR_MAG = re.compile(
    r"(?P<cur>US\$|[$£€¥₹])\s?(?P<num>" + _NUM + r")\s?(?P<mag>trillion|billion|million|thousand|tn|bn|mn|mm|[kmbtKMBT])\b")
_RE_CUR = re.compile(r"(?P<cur>US\$|[$£€¥₹])\s?(?P<num>" + _NUM + r")")
# Bare magnitude: "4.2bn", "300m" — single letters only on a DECIMAL ("3M" is a company, "4.2B" is not)
_RE_MAG = re.compile(r"\b(?P<num>" + _NUM + r")\s?(?P<mag>trillion|billion|million|tn|bn|mn)\b"
                     r"|\b(?P<dnum>\d+\.\d+)(?P<dmag>[kmbtKMBT])\b")
_RE_PCT_RANGE = re.compile(r"(?P<a>" + _NUM + r")\s?(?:%\s?)?(?:-|–|—|to)\s?(?P<b>" + _NUM + r")\s?%")
_RE_PCT = re.compile(r"(?P<sign>[+\-−]?)(?P<num>" + _NUM + r")\s?%")
_RE_BPS = re.compile(r"\b(?P<num>" + _NUM + r")\s?(?:bps|bp)\b", re.I)
_RE_YEAR = re.compile(r"(?<![\d,.$£€])\b(?P<y>19\d{2}|20\d{2})\b(?![\d,.]|\s?%)")
_RE_PERIOD = re.compile(r"\b(?P<p>[QH])(?P<n>[1-4])\b")
_RE_FY = re.compile(r"\bFY\s?(?P<y>\d{2}|\d{4})\b")
_RE_SIGNED = re.compile(r"(?<![\w.])(?P<sign>[+−]|-(?=\d))(?P<num>" + _NUM + r")")
_RE_NUM = re.compile(r"(?<![\w.])(?P<num>" + _NUM + r")(?![\w])")
_RE_ORD = re.compile(r"\b(?P<n>\d+)(?P<suf>st|nd|rd|th)\b")
_RE_TIME = re.compile(r"\b(?P<h>[01]?\d|2[0-3]):(?P<m>[0-5]\d)\b")

_RE_ISO_DATE = re.compile(r"\b(?P<y>(?:19|20)\d{2})-(?P<m>0[1-9]|1[0-2])-(?P<d>0[1-9]|[12]\d|3[01])\b")
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December"]

_ABBREV = [
    (re.compile(r"\bYoY\b", re.I), "year on year"),
    (re.compile(r"\bQoQ\b", re.I), "quarter on quarter"),
    (re.compile(r"\bMoM\b"), "month on month"),
    (re.compile(r"\bYTD\b"), "year to date"),
    (re.compile(r"\bvs\.?(?=\s)", re.I), "versus"),
    (re.compile(r"\be\.g\.", re.I), "for example"),
    (re.compile(r"\bi\.e\.", re.I), "that is"),
    (re.compile(r"\betc\.", re.I), "et cetera"),
    (re.compile(r"\bapprox\.", re.I), "approximately"),
    (re.compile(r"\bJan\.(?=\s)"), "January"), (re.compile(r"\bFeb\.(?=\s)"), "February"),
    (re.compile(r"\bAug\.(?=\s)"), "August"), (re.compile(r"\bSept?\.(?=\s)"), "September"),
    (re.compile(r"\bOct\.(?=\s)"), "October"), (re.compile(r"\bNov\.(?=\s)"), "November"),
    (re.compile(r"\bDec\.(?=\s)"), "December"),
    (re.compile(r"\s&\s"), " and "),
    (re.compile(r"\bS&P\b"), "S and P"),
    (re.compile(r"\bM&A\b"), "M and A"),
    (re.compile(r"\bR&D\b"), "R and D"),
]
_ORD_WORDS = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
              "nine": "ninth", "twelve": "twelfth"}


def _ordinal(n: int, british: bool) -> str:
    w = int_words(n, british)
    head, sep, last = w.rpartition(" ")
    last_head, dash, last_tail = last.rpartition("-")
    tail = last_tail
    if tail in _ORD_WORDS:
        tail = _ORD_WORDS[tail]
    elif tail.endswith("y"):
        tail = tail[:-1] + "ieth"
    else:
        tail += "th"
    last = f"{last_head}{dash}{tail}"
    return f"{head}{sep}{last}"


def normalize(text: str, british: bool = False) -> str:
    """Speakable text. Order matters: money and percentages before bare numbers."""
    t = text

    def iso_date(m):
        day, month, year = _ordinal(int(m["d"]), british), _MONTHS[int(m["m"]) - 1], year_words(int(m["y"]), british)
        return f"the {day} of {month} {year}" if british else f"{month} {day}, {year}"

    t = _RE_ISO_DATE.sub(iso_date, t)
    for rx, rep in _ABBREV:
        t = rx.sub(rep, t)

    def cur_mag(m):
        cur = _CURRENCY[m["cur"]]
        return f"{number_words(m['num'], british)} {_MAG_WORD[m['mag'].lower()]} {cur[1]}"

    t = _RE_CUR_MAG.sub(cur_mag, t)

    def cur(m):
        one, many, sub_one, sub_many = _CURRENCY[m["cur"]]
        num = m["num"].replace(",", "")
        whole, _, frac = num.partition(".")
        w = int(whole or "0")
        if frac and len(frac) == 2 and sub_many:  # $1,234.56 → dollars and cents
            c = int(frac)
            words = f"{int_words(w, british)} {one if w == 1 else many}"
            return words + (f" and {int_words(c, british)} {sub_one if c == 1 else sub_many}" if c else "")
        return f"{number_words(num, british)} {one if num in ('1', '1.0') else many}"

    t = _RE_CUR.sub(cur, t)
    t = _RE_MAG.sub(lambda m: (f"{number_words(m['num'], british)} {_MAG_WORD[m['mag'].lower()]}" if m["num"]
                               else f"{number_words(m['dnum'], british)} {_MAG_WORD[m['dmag'].lower()]}"), t)
    t = _RE_PCT_RANGE.sub(lambda m: f"{number_words(m['a'], british)} to {number_words(m['b'], british)} percent", t)

    def pct(m):
        sign = {"+": "plus ", "-": "minus ", "−": "minus "}.get(m["sign"], "")
        return f"{sign}{number_words(m['num'], british)} percent"

    t = _RE_PCT.sub(pct, t)
    t = _RE_BPS.sub(lambda m: f"{number_words(m['num'], british)} basis points", t)
    t = _RE_FY.sub(lambda m: "fiscal " + (year_words(int(m["y"]), british) if len(m["y"]) == 4
                                          else year_words(2000 + int(m["y"]), british)), t)
    t = _RE_PERIOD.sub(lambda m: f"{m['p']} {_ONES[int(m['n'])]}", t)
    t = _RE_YEAR.sub(lambda m: year_words(int(m["y"]), british), t)
    t = _RE_ORD.sub(lambda m: _ordinal(int(m["n"]), british), t)
    t = _RE_TIME.sub(lambda m: f"{int_words(int(m['h']))}"
                               + (f" {('oh ' + _ONES[int(m['m'])]) if int(m['m']) < 10 else int_words(int(m['m']))}"
                                  if int(m["m"]) else " o'clock"), t)
    t = _RE_SIGNED.sub(lambda m: ("plus " if m["sign"] == "+" else "minus ") + number_words(m["num"], british), t)
    t = _RE_NUM.sub(lambda m: number_words(m["num"], british), t)
    return re.sub(r"[ \t]{2,}", " ", t)
