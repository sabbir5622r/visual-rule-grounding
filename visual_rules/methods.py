from .rules import parse_decision, parse_rule, parse_values


PRIMARY_METHODS = ("direct", "read_then_apply", "execute")
GOLD_METHODS = ("gold_values", "gold_rule", "gold_both")
SHORTCUT_METHODS = ("image_only", "rule_only")
METHODS = PRIMARY_METHODS + GOLD_METHODS + SHORTCUT_METHODS


def format_values(values):
    return f"A={values[0]}, B={values[1]}"


def visual_instruction(row):
    if row.get("layout") == "photo":
        return f"Inspect the photo and count the objects. {row['measurement_note']}"
    default = "Use the chart measurements labeled A and B."
    return f"Find entry {row['target']} in the image. {row.get('measurement_note', default)}"


def run_method(method, row, respond):
    image = row["image"]
    rule_text = row["rule_text"]
    reference_rule = row["rule_code"]
    values = row["values"]
    oracle_answer = "YES" if row["answer"] else "NO"
    if method == "execute":
        extraction = respond(
            f"{visual_instruction(row)} Output only two integers in the format A=12, B=34.",
            image, 18, format_values(values),
        )
        translation = respond(
            "Represent the eligibility rule as one comparison or two comparisons joined by AND. "
            "Use only A, B, or GAP for the feature. GAP is A minus B. GE means at least and "
            "LE means at most. Use ASCII digits. Output only the comparison, like A GE 12 "
            f"or A GE 12 AND B LE 34.\nRule: {rule_text}",
            None, 36, reference_rule,
        )
        extracted_values = parse_values(extraction)
        compiled_rule = parse_rule(translation)
        answer = compiled_rule.applies(extracted_values) if compiled_rule and extracted_values else None
        expected_rule = parse_rule(reference_rule)
        return {
            "answer": answer,
            "raw": {"values": extraction, "rule": translation},
            "value_exact": extracted_values == tuple(values),
            "rule_exact": compiled_rule is not None and sorted(
                (clause.feature, clause.comparison, clause.threshold) for clause in compiled_rule.clauses
            ) == sorted(
                (clause.feature, clause.comparison, clause.threshold) for clause in expected_rule.clauses
            ),
            "value_parse_valid": extracted_values is not None,
            "rule_parse_valid": compiled_rule is not None,
        }
    use_gold_values = method in ("gold_values", "gold_both")
    use_gold_rule = method in ("gold_rule", "gold_both")
    target = "the scene" if row.get("layout") == "photo" else f"entry {row['target']}"
    measurement = (
        f"Use these measurements for {target}: {format_values(values)}."
        if use_gold_values else visual_instruction(row)
    )
    rule = (
        f"Formal rule: {reference_rule}. GE means at least. LE means at most. GAP means A minus B."
        if use_gold_rule else f"Written rule: {rule_text}"
    )
    subject = "this scene" if row.get("layout") == "photo" else "this entry"
    prompt = f"{measurement}\n{rule}\nDoes {subject} qualify under the supplied rule?"
    if method == "image_only":
        prompt = f"{visual_instruction(row)}\nDoes {subject} qualify?"
    if method == "rule_only":
        label = "Case: scene" if row.get("layout") == "photo" else f"Entry: {row['target']}"
        prompt = f"{label}\n{rule}\nDoes {subject} qualify?"
    if method == "read_then_apply":
        answer = respond(
            prompt + "\nOutput one line in the format A=12, B=34; YES. Use YES or NO after the semicolon.",
            image, 25, f"{format_values(values)}; {oracle_answer}",
        )
        predicted_values, separator, decision = answer.rpartition(";")
        parsed_values = parse_values(predicted_values)
        return {
            "answer": parse_decision(decision) if separator and parsed_values is not None else None,
            "raw": {"decision": answer},
            "value_exact": parsed_values == tuple(values),
            "value_parse_valid": parsed_values is not None,
        }
    answer = respond(
        prompt + "\nOutput only YES or NO.", None if use_gold_values or method == "rule_only" else image, 5, oracle_answer
    )
    return {"answer": parse_decision(answer), "raw": {"decision": answer}}
