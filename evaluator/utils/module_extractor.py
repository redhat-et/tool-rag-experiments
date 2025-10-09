import importlib
import pkgutil
import traceback
from typing import Type, Dict, Any, List

from evaluator.config.schema import MetricCollectorConfig, AlgorithmConfig, ModelConfig
from evaluator.interfaces.metric_collector import MetricCollector
from evaluator.interfaces.algorithm import Algorithm
from evaluator.utils.utils import log

_ALGO_REGISTRY: Dict[str, Type[Algorithm]] = {}
_ALGO_PACKAGE = "evaluator.algorithms"

_METRIC_REGISTRY: Dict[str, Type[MetricCollector]] = {}
_METRIC_PACKAGE = "evaluator.metric_collectors"


def _normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _resolve(name: str, registry: Dict[str, Any]) -> Any:
    norm = _normalize(name)
    if norm not in registry:
        raise ValueError(f"Unknown module '{name}'")
    return registry[norm]


def _import_all_algorithms(package_path: str) -> None:
    package = importlib.import_module(package_path)
    for _, module_name, _ in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        try:
            importlib.import_module(module_name)
        except Exception as e:
            # if some modules couldn't be imported, you still have to proceed
            log(f"ERROR importing module {module_name}: {e}")
            traceback.print_exc()


# Algorithm factory
def register_algorithm(name: str):
    """Decorator to register an Algorithm subclass under `name`."""
    norm = _normalize(name)

    def _decorator(cls: Type[Algorithm]):
        if not issubclass(cls, Algorithm):
            raise TypeError(f"{cls.__name__} must subclass Algorithm")
        _ALGO_REGISTRY[norm] = cls
        setattr(cls, "__algo_name__", name)
        return cls
    return _decorator


def create_algorithms(specs: List[AlgorithmConfig], model_config: List[ModelConfig]) -> List[Algorithm]:
    if not _ALGO_REGISTRY:
        _import_all_algorithms(_ALGO_PACKAGE)

    instances: List[Algorithm] = []
    for spec in specs:
        cls = _resolve(spec.module_name, _ALGO_REGISTRY)
        instances.append(cls(dict(spec.settings), model_config, spec.label))
    return instances


# Metric collector factory
def register_metric_collector(name: str):
    """Decorator to register a MetricCollector subclass under `name`."""
    norm = _normalize(name)

    def _decorator(cls: Type[MetricCollector]):
        if not issubclass(cls, MetricCollector):
            raise TypeError(f"{cls.__name__} must subclass MetricCollector")
        _METRIC_REGISTRY[norm] = cls
        setattr(cls, "__collector_name__", name)
        return cls
    return _decorator


def create_metric_collectors(specs: List[MetricCollectorConfig], model_config: List[ModelConfig]) -> List[MetricCollector]:
    if not _METRIC_REGISTRY:
        _import_all_algorithms(_METRIC_PACKAGE)

    instances: List[MetricCollector] = []
    for spec in specs:
        cls = _resolve(spec.module_name, _METRIC_REGISTRY)
        instances.append(cls(dict(spec.settings), model_config))
    return instances
