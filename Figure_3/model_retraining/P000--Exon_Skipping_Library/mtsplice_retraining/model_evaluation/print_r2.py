#!/usr/bin/env python3
"""
print_r2_table.py

Small utility to print r^2 (or any scalar metric) for a list of cell types across
multiple named datasets. Column widths are computed automatically from the
longest dataset name, cell type label, and formatted values so the table is
nicely aligned no matter what keys you pass in.

Usage:
    python print_r2_table.py

Or import `print_table` from your code and call with your dict.
"""
from typing import List, Dict, Optional, Iterable
import math
import sys


def _format_value(v: Optional[float], precision: int) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "nan"
    return f"{v:.{precision}f}"


def print_table(
    cell_types: List[str],
    data: Dict[str, List[Optional[float]]],
    *,
    dataset_order: Optional[Iterable[str]] = None,
    precision: int = 3,
    title: Optional[str] = None,
    name_col_label: str = "",
    mean_label: str = "Mean",
    print_string: bool = True,
) -> None:
    """Print a table where rows are datasets (keys of `data`) and columns are
    `cell_types`. `data` must map a dataset name -> list of values in the same
    order as `cell_types`.

    Args:
        cell_types: list of column labels
        data: mapping of dataset name -> list of numeric values (or None)
        dataset_order: optional order for iterating rows (defaults to data.keys())
        precision: number of decimal places when formatting numbers
        title: optional title printed above the table
        name_col_label: label for the left-most column (defaults to "Dataset")
        mean_label: label for the mean column
    """
    final_string = []
    if dataset_order is None:
        dataset_order = list(data.keys())

    # Validate lengths
    for k in dataset_order:
        vals = data.get(k)
        if vals is None:
            raise ValueError(f"Dataset '{k}' not found in data")
        if len(vals) != len(cell_types):
            raise ValueError(
                f"Dataset '{k}' length ({len(vals)}) != number of cell types ({len(cell_types)})"
            )

    # Pre-format values and compute column widths
    formatted: Dict[str, List[str]] = {}
    for k in dataset_order:
        formatted[k] = [_format_value(v, precision) for v in data[k]]

    # Mean per dataset (ignore None/nan)
    means: Dict[str, str] = {}
    for k in dataset_order:
        vals = [v for v in data[k] if v is not None and not (isinstance(v, float) and math.isnan(v))]
        means[k] = _format_value(sum(vals) / len(vals) if vals else float('nan'), precision)

    # Column widths: left-most name column, then one col per cell type, then mean col
    name_col_width = max(len(name_col_label), max(len(k) for k in dataset_order)) + 2
    col_widths = []
    for ci, ct in enumerate(cell_types):
        max_val_len = max(len(ct), *(len(formatted[k][ci]) for k in dataset_order))
        col_widths.append(max_val_len + 2)
    mean_col_width = max(len(mean_label), max(len(m) for m in means.values())) + 2

    # Optional title (centered relative to table width)
    table_width = name_col_width + sum(col_widths) + mean_col_width
    if title:
        final_string.append(title.center(table_width))

    # Header
    header = f"{name_col_label:<{name_col_width}}"
    for ct, w in zip(cell_types, col_widths):
        header += f"{ct:>{w}}"
    header += f"{mean_label:>{mean_col_width}}"
    final_string.append(header)
    final_string.append("-" * table_width)

    # Rows
    for k in dataset_order:
        row = f"{k:<{name_col_width}}"
        for s, w in zip(formatted[k], col_widths):
            row += f"{s:>{w}}"
        row += f"{means[k]:>{mean_col_width}}"
        final_string.append(row)
    if print_string:
        print("\n".join(final_string))
    return final_string

# --- Example usage / CLI ---
if __name__ == "__main__":
    # Example cell types and three datasets with different labels
    cell_types = ["HEK293", "HeLa", "K562", "MCF7", "HMC3"]

    r2_vals = {
        "train dLogit": [0.812, 0.721, 0.654, 0.790, 0.701],
        "validation dLogit": [0.803, 0.709, 0.640, 0.771, 0.695],
        "test set dLogit": [0.799, 0.715, 0.648, 0.765, 0.692],
    }

    title = ""
    print_table(cell_types, r2_vals, title=title)

