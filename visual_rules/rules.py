from dataclasses import dataclass
import operator
import re


COMPARISONS = {"GE": operator.ge, "LE": operator.le}
CLAUSE_PATTERN = re.compile(r"(A|B|GAP)\s*(GE|LE)\s*(-?\d{1,3})", re.IGNORECASE)
VALUES_PATTERN = re.compile(r"A\s*=\s*(\d{1,3})\s*[,; ]+\s*B\s*=\s*(\d{1,3})", re.IGNORECASE)


@dataclass(frozen=True)
class Clause:
    feature: str
    comparison: str
    threshold: int

    def applies(self, values):
        first, second = values
        features = {"A": first, "B": second, "GAP": first - second}
        return COMPARISONS[self.comparison](features[self.feature], self.threshold)


@dataclass(frozen=True)
class Rule:
    clauses: tuple[Clause, ...]

    def applies(self, values):
        return all(clause.applies(values) for clause in self.clauses)

    def as_code(self):
        return " AND ".join(
            f"{clause.feature} {clause.comparison} {clause.threshold}"
            for clause in self.clauses
        )

    def as_text(self, language, style=0):
        terms = {
            "en": {"A": "A", "B": "B", "GAP": "the result of subtracting B from A"},
            "bn": {"A": "A-এর মান", "B": "B-এর মান", "GAP": "A থেকে B বিয়োগ করলে যে মান পাওয়া যায়"},
        }
        if language == "en":
            directions = {"GE": "at least", "LE": "at most"}
            clauses = [
                f"{terms[language][clause.feature]} is {directions[clause.comparison]} {clause.threshold}"
                for clause in self.clauses
            ]
            condition = " and ".join(clauses)
            templates = (
                "An entry qualifies if and only if {condition}.",
                "Mark an entry as qualifying exactly when {condition}.",
                "Meeting this condition is required and sufficient for eligibility: {condition}.",
                "Eligibility is determined solely by this condition: {condition}.",
            )
            return templates[style].format(condition=condition)
        directions = {"GE": "বা তার বেশি", "LE": "বা তার কম"}
        clauses = [
            f"{terms[language][clause.feature]} {clause.threshold} {directions[clause.comparison]}"
            for clause in self.clauses
        ]
        condition = " এবং ".join(clauses)
        templates = (
            "শুধু এই শর্ত পূরণ হলেই একটি এন্ট্রি যোগ্য হবে। {condition} হতে হবে।",
            "একটি এন্ট্রি তখনই যোগ্য হবে, যখন {condition} হবে।",
            "যোগ্য হওয়ার একমাত্র শর্ত হলো {condition} হওয়া।",
            "যোগ্যতা নির্ধারণের জন্য কেবল এই শর্তটি প্রযোজ্য: {condition}।",
        )
        return templates[style].format(condition=condition)


def parse_rule(text):
    parts = re.split(r"\s+AND\s+", text.strip().upper())
    if len(parts) not in (1, 2):
        return None
    clauses = []
    for part in parts:
        match = CLAUSE_PATTERN.fullmatch(part)
        if not match:
            return None
        feature, comparison, threshold = match.groups()
        clauses.append(Clause(feature, comparison, int(threshold)))
    return Rule(tuple(clauses))


def parse_values(text):
    match = VALUES_PATTERN.fullmatch(text.strip().rstrip("."))
    return tuple(map(int, match.groups())) if match else None


def parse_decision(text):
    answer = text.strip().upper().rstrip(".")
    return {"YES": True, "NO": False}.get(answer)
