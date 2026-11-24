"""Text → speakable text. Each case is a number the read-aloud got wrong:
espeak (Kokoro's phonemizer) reads a decimal point as a pause and splits a
number at its thousands comma — "130,000" came out as "one hundred thirty,
zero zero zero"."""

from hearth.speech_text import int_words, normalize, number_words, year_words


def test_thousands_are_one_number():
    assert normalize("hired 130,000 workers") == "hired one hundred thirty thousand workers"
    assert normalize("hired 130,000 workers", british=True) == "hired one hundred and thirty thousand workers"


def test_decimals_say_point():
    assert normalize("rose 2.5%") == "rose two point five percent"
    assert normalize("closed at 7,592.40") == "closed at seven thousand five hundred ninety-two point four"
    assert normalize("down 0.35%") == "down zero point three five percent"


def test_money():
    assert normalize("$1,234.56") == "one thousand two hundred thirty-four dollars and fifty-six cents"
    assert normalize("$1.05") == "one dollar and five cents"
    assert normalize("$4.2B") == "four point two billion dollars"
    assert normalize("$3.1bn") == "three point one billion dollars"
    assert normalize("£2.5 million") == "two point five million pounds"
    assert normalize("€300m") == "three hundred million euros"


def test_bare_magnitudes_only_when_unambiguous():
    assert normalize("revenue 4.2B") == "revenue four point two billion"
    assert normalize("3M fell") == "3M fell"  # a company, not three million


def test_signs_ranges_and_bps():
    assert normalize("Gold +0.94%, oil -0.57%") == "Gold plus zero point nine four percent, oil minus zero point five seven percent"
    assert normalize("3-5% growth") == "three to five percent growth"
    assert normalize("widened 150 bps") == "widened one hundred fifty basis points"


def test_periods_years_dates_times():
    assert normalize("Q3 revenue") == "Q three revenue"
    assert normalize("In 2026") == "In twenty twenty-six"
    assert normalize("FY25 EPS") == "fiscal twenty twenty-five EPS"
    assert normalize("on 2026-11-12") == "on November twelfth, twenty twenty-six"
    assert normalize("on 2026-11-12", british=True) == "on the twelfth of November twenty twenty-six"
    assert normalize("at 10:30") == "at ten thirty"
    assert normalize("its 3rd fall") == "its third fall"


def test_shorthand():
    assert normalize("12% YoY") == "twelve percent year on year"
    assert normalize("vs. a year ago") == "versus a year ago"
    assert normalize("the S&P 500") == "the S and P five hundred"


def test_words():
    assert int_words(0) == "zero"
    assert int_words(1_000_005, british=True) == "one million and five"
    assert int_words(21) == "twenty-one"
    assert number_words("1,000") == "one thousand"
    assert year_words(2005) == "two thousand five"
    assert year_words(1905) == "nineteen oh five"
    assert year_words(2000) == "two thousand"
