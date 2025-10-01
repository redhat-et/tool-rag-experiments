import json
import re
from typing import Any, Dict, List

from langchain_core.language_models import BaseChatModel

from evaluator.components.data_provider import QuerySpecification
from evaluator.components.llm_provider import query_llm
from evaluator.config.schema import ModelConfig
from evaluator.interfaces.metric_collector import MetricCollector
from evaluator.utils.module_extractor import register_metric_collector
from evaluator.utils.utils import extract_final_answer_from_response, strip_think, print_verbose

"""
This collector supports FOUR judge-based metrics, each with a dedicated prompt:
  - "task_success_no_ref"     : did the final answer solve the user's request? (no reference)
  - "task_success_with_ref"   : same, but evaluated against a provided reference answer
  - "coverage"                : did the answer satisfy all constraints/requirements from the query?
  - "clarity"                 : readability/conciseness/structure (normalized 0..1 from 1..5 scale internally)

NOT INCLUDED IN THIS VERSION (planned for v2):
  - "faithfulness" / "groundedness"
  - "attribution"

Rationales are requested from the judge for every metric.
"""


# ===============================
# Metric Keys (constants)
# ===============================
TASK_SUCCESS_NO_REF = "task_success_no_ref"
TASK_SUCCESS_WITH_REF = "task_success_with_ref"
COVERAGE = "coverage"
CLARITY = "clarity"

SUPPORTED_METRICS = {TASK_SUCCESS_NO_REF, TASK_SUCCESS_WITH_REF, COVERAGE, CLARITY}

METRIC_NAME_TO_DESCRIPTION = {
    TASK_SUCCESS_NO_REF: "Average Task Success (No Ref)",
    TASK_SUCCESS_WITH_REF: "Average Task Success (With Ref)",
    COVERAGE: "Coverage of Constraints and Requirements",
    CLARITY: "Readability, Conciseness and Structure",
}


# ===============================
# Per-Metric System Prompts
# ===============================
_SYS_TASK_SUCCESS_NO_REF = """\
You are a strict evaluator. Judge whether the FINAL ANSWER successfully solves the USER QUERY.
Ignore style; focus on correctness and whether the requested outcome is achieved.
Return a JSON: {"score": 0..1, "rationale": "<=60 words>"}.
Scoring: 1=fully correct and sufficient; 0.5=partially correct or missing key parts; 0=incorrect or non-answer.
"""

_SYS_TASK_SUCCESS_WITH_REF = """\
You are a strict evaluator. Compare the FINAL ANSWER to the REFERENCE ANSWER for the same USER QUERY.
Score how well the final answer matches the reference in correctness and essential content.
Return a JSON: {"score": 0..1, "rationale": "<=60 words>"}.
Scoring: 1=substantively matches; 0.5=partially matches; 0=contradicts or misses essentials.
Do not penalize minor wording differences if content is equivalent.
"""

_SYS_COVERAGE = """\
You are a strict evaluator. Determine if the FINAL ANSWER covers ALL key requirements, constraints, and sub-questions in the USER QUERY.
Return a JSON: {"score": 0..1, "rationale": "<=60 words>"}.
Scoring: 1=all requirements satisfied; 0.5=some partially/omitted; 0=largely missing required elements.
"""

_SYS_CLARITY = """\
You are a strict evaluator of writing quality. Judge the FINAL ANSWER's clarity, conciseness, and structure.
First rate on a 1..5 scale internally (1=unclear/rambling, 5=clear/concise/well-structured) then map to 0..1: (score-1)/4.
Return a JSON: {"score": 0..1, "rationale": "<=60 words>"}.
"""


# ===============================
# User Prompt Builders (per metric)
# ===============================
def _user_task_success_no_ref(query: str, final_answer: str) -> str:
    return f"USER QUERY:\n{query or ''}\n\nFINAL ANSWER:\n{final_answer or ''}\n"


def _user_task_success_with_ref(query: str, final_answer: str, reference_answer: str) -> str:
    return (f"USER QUERY:\n{query or ''}\n\nFINAL ANSWER:\n{final_answer or ''}\n\n"
            f"REFERENCE ANSWER:\n{reference_answer or ''}\n")


def _user_coverage(query: str, final_answer: str) -> str:
    return f"USER QUERY (requirements/constraints inside):\n{query or ''}\n\nFINAL ANSWER:\n{final_answer or ''}\n"


def _user_clarity(final_answer: str) -> str:
    return f"FINAL ANSWER:\n{final_answer or ''}\n"


# ===============================
# Answer Quality Metric Collector (per-metric judges)
# ===============================
@register_metric_collector("answer_quality_metric_collector")
class AnswerQualityMetricCollector(MetricCollector):
    """
    Judge-only Answer Quality Metric Collector (per-metric prompts & per-metric judges).

    Config:
        judges (Dict[str, str]): a dictionary with metric names as keys and model IDs as values.

    You should supply configs for only the metrics you would like the judge(s) to evaluate.
    """

    def __init__(self, settings: Dict, model_config: List[ModelConfig]):
        super().__init__(settings, model_config)

        judge_models = self._settings["judges"]
        unknown = [m for m in judge_models.keys() if m not in SUPPORTED_METRICS]
        if unknown:
            raise ValueError(f"Unsupported metric keys: {unknown}. Supported: {sorted(SUPPORTED_METRICS)}")

        self.metrics = list(judge_models.keys())

        self.judges: Dict[str, BaseChatModel] = {}
        for metric, model_id in judge_models.items():
            self.judges[metric] = self._get_llm(model_id=model_id)

        self._rows: List[Dict[str, Any]] = []
        self._n = 0

    def get_collected_metrics_names(self) -> List[str]:
        return [METRIC_NAME_TO_DESCRIPTION[metric] for metric in SUPPORTED_METRICS if metric in self.metrics]

    def set_up(self) -> None:
        super().set_up()

        self._rows.clear()
        self._n = 0

    def prepare_for_measurement(self, query_spec: QuerySpecification) -> None:
        pass

    def register_measurement(self, query_spec: QuerySpecification, **kwargs) -> None:
        self._n += 1

        query = query_spec.query
        response = kwargs.get("response") or {}

        final_answer = extract_final_answer_from_response(response)
        final_answer = strip_think(final_answer)

        ref_answer = query_spec.reference_answer  # may be None

        row: Dict[str, Any] = {"query": query}

        # For each requested metric, run its own prompt+judge
        if TASK_SUCCESS_NO_REF in self.metrics:
            user_prompt = _user_task_success_no_ref(query, final_answer or "")
            print_verbose(user_prompt)
            out = self._parse_judge_score(
                query_llm(
                    model=self.judges[TASK_SUCCESS_NO_REF],
                    system_prompt=_SYS_TASK_SUCCESS_NO_REF,
                    user_prompt=user_prompt,
                )
            )
            row["task_success_no_ref_j"] = out.get("score")
            row["task_success_no_ref_j_rationale"] = out.get("rationale")

        if TASK_SUCCESS_WITH_REF in self.metrics:
            # Warn early if no reference supplied
            if ref_answer is None:
                # store None score; still keep the row
                row["task_success_with_ref_j"] = None
                row["task_success_with_ref_j_rationale"] = "reference_answer missing"
            else:
                user_prompt = _user_task_success_with_ref(query, final_answer or "", ref_answer or "")
                print_verbose(user_prompt)
                out = self._parse_judge_score(
                    query_llm(
                        model=self.judges[TASK_SUCCESS_WITH_REF],
                        system_prompt=_SYS_TASK_SUCCESS_WITH_REF,
                        user_prompt=user_prompt,
                    )
                )
                row["task_success_with_ref_j"] = out.get("score")
                row["task_success_with_ref_j_rationale"] = out.get("rationale")

        if COVERAGE in self.metrics:
            user_prompt = _user_coverage(query, final_answer or "")
            print_verbose(user_prompt)
            out = self._parse_judge_score(
                query_llm(
                    model=self.judges[COVERAGE],
                    system_prompt=_SYS_COVERAGE,
                    user_prompt=user_prompt,
                )
            )
            row["coverage_j"] = out.get("score")
            row["coverage_j_rationale"] = out.get("rationale")

        if CLARITY in self.metrics:
            user_prompt = _user_clarity(final_answer or "")
            print_verbose(user_prompt)
            out = self._parse_judge_score(
                query_llm(
                    model=self.judges[CLARITY],
                    system_prompt=_SYS_CLARITY,
                    user_prompt=user_prompt,
                )
            )
            row["clarity_j"] = out.get("score")
            row["clarity_j_rationale"] = out.get("rationale")

        self._rows.append(row)

    def tear_down(self) -> None:
        super().tear_down()

    def report_results(self) -> Dict[str, Any]:
        super().report_results()

        def vals(key): return [r[key] for r in self._rows if isinstance(r.get(key), (int, float))]

        def mean(key):
            v = vals(key)
            return round(sum(v)/len(v), 6) if v else None

        out = {}

        if TASK_SUCCESS_NO_REF in self.metrics:
            out[METRIC_NAME_TO_DESCRIPTION[TASK_SUCCESS_NO_REF]] = mean("task_success_no_ref_j")
            # out["task_success_no_ref_j_support"] = len(vals("task_success_no_ref_j"))

        if TASK_SUCCESS_WITH_REF in self.metrics:
            out[METRIC_NAME_TO_DESCRIPTION[TASK_SUCCESS_WITH_REF]] = mean("task_success_with_ref_j")
            # out["task_success_with_ref_j_support"] = len(vals("task_success_with_ref_j"))

        if COVERAGE in self.metrics:
            out[METRIC_NAME_TO_DESCRIPTION[COVERAGE]] = mean("coverage_j")
            # out["coverage_j_support"] = len(vals("coverage_j"))

        if CLARITY in self.metrics:
            out[METRIC_NAME_TO_DESCRIPTION[CLARITY]] = mean("clarity_j")
            # out["clarity_j_support"] = len(vals("clarity_j"))

        for key, value in out.items():
            print(f"{key}: {value:.2f}")
        return out

    @staticmethod
    def _parse_judge_score(text: str) -> Dict[str, Any]:
        print_verbose(f"JUDGE EVALUATION:\n{text}")

        def clip01(x: float) -> float or None:
            if x is None:
                return None
            return 0.0 if x < 0 else 1.0 if x > 1 else x

        m = re.search(r"\{.*}", text, flags=re.S)
        raw = text if m is None else m.group(0)
        try:
            obj = json.loads(raw)
        except Exception:
            score = None
            rat = "unparsed"
            m2 = re.search(r'"score"\s*:\s*([0-9.]+)', text)
            if m2:
                try:
                    score = float(m2.group(1))
                except Exception:
                    pass
            m3 = re.search(r'"rationale"\s*:\s*"([^"]+)"', text)
            if m3:
                rat = m3.group(1)
            return {"score": clip01(score),
                    "rationale": rat or ""}
        score = obj.get("score", None)
        rat = obj.get("rationale", "")
        return {
            "score": clip01(float(score)),
            "rationale": str(rat),
        }
