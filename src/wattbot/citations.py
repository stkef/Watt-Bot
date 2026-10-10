"""Map the model's S-numbers back to document ids (FR-8) and build submission fields."""
from __future__ import annotations

import re

from .data import format_ref_list


def map_sources(sources: list, chunk_by_label: dict[str, "object"]) -> list[str]:
    """['S2', 'S1', 'S2'] -> unique doc ids in first-use order. Unknown labels are dropped."""
    seen: list[str] = []
    for s in sources or []:
        label = re.sub(r"[^A-Za-z0-9]", "", str(s)).upper()
        ch = chunk_by_label.get(label)
        if ch is not None and ch.doc_id not in seen:
            seen.append(ch.doc_id)
    return seen


def ref_fields(doc_ids: list[str], url_by_id: dict[str, str]) -> tuple[str, str]:
    """-> (ref_id, ref_url) strings in train_QA.csv's list format."""
    return format_ref_list(doc_ids), format_ref_list([url_by_id[d] for d in doc_ids if d in url_by_id])
