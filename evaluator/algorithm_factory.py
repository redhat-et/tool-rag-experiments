import importlib
import pkgutil
from typing import Type, Dict, Tuple, Any, Sequence, List
from difflib import get_close_matches

from evaluator.tool_rag_algorithm import ToolRagAlgorithm

_ALGO_REGISTRY: Dict[str, Type[ToolRagAlgorithm]] = {}
_PACKAGE_TO_SCAN = "evaluator.algorithms"


def _normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def register_tool_rag_algorithm(name: str):
    """Decorator to register a ToolRagAlgorithm subclass under `name`."""
    norm = _normalize(name)

    def _decorator(cls: Type[ToolRagAlgorithm]):
        if not issubclass(cls, ToolRagAlgorithm):
            raise TypeError(f"{cls.__name__} must subclass ToolRagAlgorithm")
        _ALGO_REGISTRY[norm] = cls
        setattr(cls, "__algo_name__", name)
        return cls
    return _decorator


def _resolve(name: str) -> Type[ToolRagAlgorithm]:
    norm = _normalize(name)
    if norm not in _ALGO_REGISTRY:
        sugg = get_close_matches(norm, _ALGO_REGISTRY.keys(), n=3, cutoff=0.6)
        hint = f" Did you mean: {', '.join(getattr(_ALGO_REGISTRY[s],'__algo_name__', s) for s in sugg)}?" if sugg else ""
        available = ", ".join(sorted(getattr(c, "__algo_name__", k) for k, c in _ALGO_REGISTRY.items()))
        raise ValueError(f"Unknown algorithm '{name}'. Available: {available}.{hint}")
    return _ALGO_REGISTRY[norm]


def _import_all_algorithms() -> None:
    package = importlib.import_module(_PACKAGE_TO_SCAN)
    for _, module_name, _ in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        importlib.import_module(module_name)


# ---- Factory with lazy auto-import ----
Spec = Tuple[str, Dict[str, Any]]


def create_algorithms(specs: Sequence[Spec]) -> List[ToolRagAlgorithm]:
    """
    specs: [("algorithm_a", {...}), ("algorithm_b", {...}), ...]
    returns: [AlgorithmA(...), AlgorithmB(...), ...]
    """
    if not _ALGO_REGISTRY:
        _import_all_algorithms()

    instances: List[ToolRagAlgorithm] = []
    for name, settings in specs:
        cls = _resolve(name)
        instances.append(cls(dict(settings)))
    return instances
