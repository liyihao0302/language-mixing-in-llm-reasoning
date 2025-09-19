def extract_after_boxed(s):
    start = s.rfind(r'\boxed')  # Use rfind for the *last* occurrence
    if start == -1:
        return ""
    # Get everything after \boxed
    after = s[start + len(r'\boxed'):]
    # Trim leading characters like { or spaces
    after = after.lstrip("{ ")
    # Cut off at the last closing brace (optional: could use balancing instead)
    end = after.rfind("}")
    if end != -1:
        after = after[:end]
    return after.strip()

def evaluate_answer(gt_answer, pred):
    # Return 0 if prediction is empty
    if len(pred.strip()) == 0:
        return 0

    # Normalize LaTeX
    pred = pred.replace('\\dfrac', '\\frac')
    gt_answer = gt_answer.replace('\\dfrac', '\\frac')

    # Try to compare as numbers
    try:
        pred_num = float(pred)
        gt_num = float(gt_answer)
        return 1 if pred_num == gt_num else 0
    except ValueError:
        # Not numbers: remove $ and compare as strings
        gt_answer_clean = gt_answer.replace('$', '').strip().lower()
        return 1 if pred.strip().lower() == gt_answer_clean else -1