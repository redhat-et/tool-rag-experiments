import csv
from pathlib import Path
from typing import Any, Iterable, List, Dict, Optional

from evaluator.interfaces.metric_collector import MetricCollector


class CSVLogger:
    """ Encapsulates CSV logging operations. """
    def __init__(
        self,
        components: List[MetricCollector],
        csv_path: Path,
        metadata_columns: Optional[List[str]] = None,
        newline: str = "",
        encoding: str = "utf-8",
    ) -> None:
        self.components = components
        self.csv_path = csv_path
        self.metadata_columns = metadata_columns or []
        self.newline = newline
        self.encoding = encoding

        self._header = []
        self._writer = None
        self._file_handle = None

    def open(self) -> None:
        """
        Compute header and open the CSV for writing. Call once before the loop.
        """
        component_columns = self._collect_all_component_columns()
        # Ensure no collisions between metadata and component columns
        collisions = set(self.metadata_columns) & set(component_columns)
        if collisions:
            raise ValueError(f"Metadata columns collide with component columns: {sorted(collisions)}")

        self._header = list(self.metadata_columns) + component_columns
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        self._file_handle = self.csv_path.open("w", newline=self.newline, encoding=self.encoding)
        self._writer = csv.DictWriter(self._file_handle, fieldnames=self._header, extrasaction="ignore")
        self._writer.writeheader()

    def close(self) -> None:
        """Close the CSV handle. Call once after the loop (or use context manager)."""
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
            self._writer = None

    # Context manager convenience
    def __enter__(self) -> "CSVLogger":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def log_experiment(self, meta_values: Optional[Dict[str, Any]] = None) -> None:
        """
        Compose an aligned row from all components (plus optional `extra`)
        and write it to the CSV. Call at the end of each iteration.
        """
        if self._writer is None:
            raise RuntimeError("CSVLogger is not open. Call .open() before logging.")

        row = self._compose_row(self._header, self.components, extra=meta_values)
        self._writer.writerow(row)

    def _collect_all_component_columns(self) -> List[str]:
        """
        Collect columns from components in a deterministic order.
        Raises on duplicates to preserve one-to-one mapping.
        """
        ordered_cols: List[str] = []
        seen = set()
        for comp in self.components:
            cols = comp.get_collected_metrics_names()
            for col in cols:
                if col in seen:
                    raise ValueError(
                        f"Duplicate column detected: '{col}' from {comp.__class__.__name__}. "
                        "Consider namespacing columns per component (e.g., with a prefix)."
                    )
                seen.add(col)
                ordered_cols.append(col)
        return ordered_cols

    @staticmethod
    def _compose_row(
            columns: List[str],
            components: Iterable[MetricCollector],
            extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build a row dict aligned to `columns`.
        - Initializes all columns to empty string to keep CSV shape stable.
        - Overwrites with results from each component.
        - Optionally merges in `extra` fields if present (must exist in `columns`).
        """
        row = {c: "" for c in columns}

        for comp in components:
            results = comp.report_results()
            unknown = [k for k in results.keys() if k not in columns]
            if unknown:
                raise KeyError(
                    f"Component {comp.__class__.__name__} returned unknown keys not in header: {unknown}"
                )
            row.update(results)

        if extra:
            unknown = [k for k in extra.keys() if k not in columns]
            if unknown:
                raise KeyError(f"'extra' contains unknown keys not in header: {unknown}")
            row.update(extra)

        return row
