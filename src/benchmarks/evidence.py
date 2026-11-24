"""Check clear question qualifiers against rows selected from the input CSV."""
from __future__ import annotations

import re


_STOP = {
    "a", "all", "an", "and", "at", "between", "by", "for", "from", "give", "how",
    "in", "is", "many", "number", "of", "or", "other", "over", "report", "reported",
    "the", "to", "total", "was", "were", "what", "with", "when",
}


def _words(text):
    words = re.findall(r"[a-z]+", str(text).casefold())
    return tuple(word[:-1] if len(word) > 3 and word.endswith("s")
                 else word[:-2] if len(word) > 5 and word.endswith("ed") else word for word in words)


def _category_scope(source, question_words, question_terms):
    """Identify exact mentions or a unique multi-word match in a text column."""
    question = " " + " ".join(question_words) + " "
    for column in source.columns:
        if " all " + " ".join(_words(column)) + " " in question:
            continue
        values = list(source[column].unique())
        if not 2 <= len(values) <= 100:
            continue
        candidates = {
            value: words for value in values if (words := _words(value))
            and len(str(value)) <= 100 and len("".join(words)) >= 3
            and not re.match(r"^\s*[+$\d-]", str(value))
        }
        if not 2 <= len(candidates) <= 100:
            continue
        exact = {value for value, words in candidates.items()
                 if " " + " ".join(words) + " " in question}
        if exact:
            yield column, exact
            continue
        scores = {value: len(question_terms.intersection(words)) for value, words in candidates.items()}
        top = max(scores.values())
        winners = {value for value, score in scores.items() if score == top}
        if top >= 2 and len(winners) == 1:
            yield column, winners


def query_scope_issues(question, source, matched, aggregate, number):
    """Report mismatched coverage when explicit qualifiers can be grounded in the CSV."""
    question_words = _words(question)
    question_terms = set(question_words) - _STOP
    expected = source
    checks = []
    years = [int(year) for year in re.findall(r"\b(?:19|20)\d{2}\b", question)]
    year_columns = [column for column in source.columns if _words(column) == ("year",)]
    if years and len(year_columns) == 1:
        year_column = year_columns[0]
        yr = number(source[year_column])
        in_scope = (yr.between(min(years), max(years)) if len(years) == 2 and "between" in question_words
                    else yr.isin(years))
        if in_scope.any():
            expected = expected.loc[in_scope]
            checks.append(f"{year_column} in requested range {min(years)}–{max(years)}")

    for column, categories in _category_scope(source, question_words, question_terms):
        in_scope = expected[column].isin(categories)
        if in_scope.any():
            expected = expected.loc[in_scope]
            checks.append(f"{column} matching {', '.join(map(str, sorted(categories)[:5]))}")

    issues = []
    if checks and set(matched.index) != set(expected.index):
        issues.append("Check row scope: " + "; ".join(checks) + ". Include all matching rows and exclude other categories or years.")

    if not aggregate:
        return issues

    measure = aggregate.get("column")
    measure_terms = set(_words(measure))
    money_terms = {"dollar", "loss", "amount", "money", "cost", "price"}
    money_columns = [column for column in source.columns
                     if "$" in column or money_terms.intersection(_words(column))]
    if money_columns and money_terms.intersection(question_terms):
        if aggregate.get("op") == "count" or measure not in money_columns:
            issues.append(f"The question asks for money. Use the monetary column {', '.join(money_columns)}.")
    elif measure and not {"rate", "percent", "per"}.intersection(question_terms):
        if "%" in measure or {"rate", "percent", "per"}.intersection(measure_terms):
            issues.append("The question asks for an absolute quantity; check whether this percentage or rate is the requested measure.")
    count_columns = [column for column in source.columns if column.strip().startswith("#")
                     and (set(_words(column)) - {"of", "the"}).intersection(question_words)]
    if count_columns and aggregate.get("op") == "count":
        issues.append(f"The rows already contain counts. Aggregate the measure {', '.join(count_columns)} instead of counting CSV rows.")

    sentence = " " + " ".join(question_words) + " "
    explicit_all = any(" all " + " ".join(words) + " " in sentence
                       for column in source.columns if (words := _words(column)))
    if explicit_all and measure in source.columns:
        valid = set(expected.index[number(expected[measure]).notna()])
        included = set(matched.index[number(matched[measure]).notna()])
        if included != valid:
            issues.append("The question asks for all categories. Include every in-scope row with a numeric value.")
    return issues
