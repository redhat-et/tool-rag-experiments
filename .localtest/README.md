# Archive Folder

This folder contains previous versions of evaluators and other files that have been superseded by improved implementations.

## Files

### `toolbench_pass_rate_evaluator.py`
**Original ToolBench evaluator implementation**

**Why archived:**
- Replaced by `simplified_toolbench_evaluator.py` which has superior functionality
- Lacks multi-threading, backoff retry logic, and comprehensive StableToolBench prompts
- Kept for backward compatibility and reference

**Key differences from current version:**
- Basic prompt template (shorter, less comprehensive)
- No multi-threading support
- No backoff retry logic
- No progress bars
- Basic error handling

**To restore:** Copy back to main directory and update imports in `enhanced_maxtool_experiment.py`

## Current Production Files

The current production-ready evaluator is:
- `simplified_toolbench_evaluator.py` - Complete StableToolBench integration with all features

## Migration Notes

If you need to revert to the original evaluator:
1. Copy `toolbench_pass_rate_evaluator.py` back to main directory
2. Update import in `enhanced_maxtool_experiment.py`:
   ```python
   from toolbench_pass_rate_evaluator import ToolBenchPassRateEvaluator, PassRateResult
   ```
3. Update evaluator initialization:
   ```python
   pass_rate_evaluator = ToolBenchPassRateEvaluator(evaluate_times=4)
   ```
