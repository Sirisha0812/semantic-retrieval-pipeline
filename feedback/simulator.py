from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from retrieval.vector_store import Document


@dataclass 
class UserFeedback:
    query_id: str
    clicked_doc_ids: list[str]
    dwell_times: dict[str, float]
    follow_up_query: Optional[str]

    def to_reward(self) -> float:
        # click score
        click_score = min(1.0, len(self.clicked_doc_ids) / 2.0)

        # dwell score
        if not self.dwell_times:
            dwell_score = 0.0
        else:
            avg_dwell = sum(self.dwell_times.values()) / len(self.dwell_times)
            dwell_score = max(0.0, min(1.0, (avg_dwell - 5) / (30 - 5)))

        # no-refinement score
        no_refinement_score = 0.0 if self.follow_up_query is not None else 1.0

        return (
            0.4 * click_score
            + 0.4 * dwell_score
            + 0.2 * no_refinement_score
        )


class FeedbackSimulator:
    _REFINEMENT_PREFIXES = [
        "more details about",
        "specifically",
        "explain",
        "define",
    ]

    def simulate(
        self,
        query_id: str,
        documents: list[Document],
        relevance_scores: list[float],
    ) -> UserFeedback:
        clicked_doc_ids: list[str] = []
        dwell_times: dict[str, float] = {}
        follow_up_query: Optional[str] = None

        for doc, score in zip(documents, relevance_scores):
            if score > 0.8:
                clicked_doc_ids.append(doc.id)
                dwell_times[doc.id] = random.uniform(30, 120)

            elif score > 0.5:
                if random.random() > 0.4:
                    clicked_doc_ids.append(doc.id)
                    dwell_times[doc.id] = random.uniform(8, 35)
                if random.random() < 0.3:
                    follow_up_query = self._make_follow_up(query_id)

            else:
                if random.random() < 0.15:
                    clicked_doc_ids.append(doc.id)
                    dwell_times[doc.id] = random.uniform(2, 8)
                follow_up_query = self._make_follow_up(query_id)

        return UserFeedback(
            query_id=query_id,
            clicked_doc_ids=clicked_doc_ids,
            dwell_times=dwell_times,
            follow_up_query=follow_up_query,
        )

    def _make_follow_up(self, query_id: str) -> str:
        prefix = random.choice(self._REFINEMENT_PREFIXES)
        return f"{prefix} {query_id}"
