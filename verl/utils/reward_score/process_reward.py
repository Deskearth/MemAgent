"""LLM-based Process Reward Model.

This module implements a reward model that evaluates whether a memory update
string correctly stores the information required to answer a given question. It
relies on an LLM accessible through the OpenAI compatible API. The model is
invoked with a short yes/no prompt and returns ``1.0`` if the answer begins with
"yes" and ``0.0`` otherwise.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Iterable, Optional

try:  # Optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency might be missing
    OpenAI = None  # type: ignore

_CLIENT = None
_CONFIG: Optional[Any] = None
_LOGGER = None

# Maximum retry attempts for API call. Can be overridden by environment variable
MAX_RETRY = int(os.environ.get("VERL_PRM_RETRY", "3"))
# Seconds to wait between retries
RETRY_INTERVAL = float(os.environ.get("VERL_PRM_RETRY_INTERVAL", "1"))


_TRACKER = None


def set_config(config: Any) -> None:
    """Configure the PRM module using a trainer config."""
    global _CONFIG
    _CONFIG = config


def set_logger(logger: Any) -> None:
    """Use an existing Tracking instance for logging."""
    global _LOGGER, _TRACKER
    _LOGGER = logger
    _TRACKER = logger


def _get_tracker():
    """Return a Tracking instance for unified logging."""
    global _TRACKER
    if _TRACKER is None:
        if _LOGGER is not None:
            _TRACKER = _LOGGER
            return _TRACKER
        try:
            from verl.utils.tracking import Tracking
            if _CONFIG is not None:
                from omegaconf import OmegaConf

                _TRACKER = Tracking(
                    project_name=_CONFIG.trainer.project_name,
                    experiment_name=_CONFIG.trainer.experiment_name,
                    default_backend=_CONFIG.trainer.logger,
                    config=OmegaConf.to_container(_CONFIG, resolve=True),
                )
            else:
                _TRACKER = False  # type: ignore
        except Exception:
            _TRACKER = False  # type: ignore
    return _TRACKER


def _log_failure_to_dashboard(message: str, prompt: Optional[str] = None, attempt: Optional[int] = None) -> None:
    """Log failure info to dashboards via the unified Tracking interface.

    Older training scripts may still call this function with a single
    ``message`` argument. To remain backward compatible we accept optional
    ``prompt`` and ``attempt`` parameters and ignore them if not provided.
    """
    tracker = _get_tracker()
    if not tracker:
        return
    try:
        data = {"prm/failure": 1, "prm/error": message}
        if attempt is not None:
            data["prm/failure_attempt"] = attempt
        tracker.log(data, step=0)
        if prompt is not None:
            tracker.log({"prm/prompt": prompt[:200]}, step=0)
    except Exception:
        # Logging should never raise, so we silently ignore all errors
        pass


def _get_client() -> Any:
    """Return a cached OpenAI client instance."""

    global _CLIENT
    if _CLIENT is None:
        if OpenAI is None:
            raise RuntimeError("openai package is required for process reward")
        _CLIENT = OpenAI(
            api_key=os.environ.get("VERL_PRM_OPENAI_API_KEY", "NOT_A_REAL_KEY"),
            base_url=os.environ.get("VERL_PRM_OPENAI_BASE_URL", "http://localhost:8000/v1"),
        )
    return _CLIENT


def _single_answer(answers: Iterable[str | int | float]) -> str:
    for a in answers:
        return str(a)
    return ""


def compute_score(solution_str: str, ground_truth, extra_info: Dict[str, Any] | None = None) -> float:
    """Compute reward for memory update steps using an LLM judge.

    Args:
        solution_str: Generated memory string.
        ground_truth: The ground truth answer(s).
        extra_info: Optional dictionary containing additional information. The
            key ``"question"`` should provide the original question text.

    Returns:
        ``1.0`` if the LLM judge believes the memory string correctly stores the
        information needed to answer the question, otherwise ``0.0``.
    """

    question = None
    if isinstance(extra_info, dict):
        question = extra_info.get("question")

    if isinstance(ground_truth, (list, tuple)):
        gt_str = _single_answer(ground_truth)
    else:
        gt_str = str(ground_truth)

    prompt = (
        "Given a question, its correct answer, and a candidate memory update, "
        "determine whether the memory includes the necessary information to "
        "answer the question. Reply only 'Yes' or 'No'.\n"
        f"Question: {question}\n"
        f"Answer: {gt_str}\n"
        f"Memory: {solution_str}"
    )

    for attempt in range(1, MAX_RETRY + 1):
        try:
            client = _get_client()
            completion = client.chat.completions.create(
                model=os.environ.get("VERL_PRM_OPENAI_MODEL", "gpt-3.5-turbo"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1,
            )
            content = completion.choices[0].message.content.strip().lower()
            return 1.0 if content.startswith("yes") else 0.0
        except Exception as exc:  # pragma: no cover - network call
            print(f"Process reward attempt {attempt} failed: {exc}")
            if attempt >= MAX_RETRY:
                _log_failure_to_dashboard(str(exc), prompt, attempt)
                return 0.0
            time.sleep(RETRY_INTERVAL)
