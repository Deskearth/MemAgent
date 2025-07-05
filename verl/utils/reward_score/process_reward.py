# Simple Process Reward Model
# Evaluate memory update quality by checking whether ground truth answer appears in the memory string.

def compute_score(solution_str: str, ground_truth, extra_info=None):
    """Compute reward for memory update steps.

    Args:
        solution_str (str): Generated memory string.
        ground_truth (str | list[str]): Correct answer(s).
        extra_info: Unused.
    Returns:
        float: Reward, 1.0 if ground truth found in memory string else 0.0.
    """
    if isinstance(ground_truth, (list, tuple)):
        ground_truths = [str(gt).lower() for gt in ground_truth]
    else:
        ground_truths = [str(ground_truth).lower()]

    solution = solution_str.lower()
    return float(any(gt in solution for gt in ground_truths))
