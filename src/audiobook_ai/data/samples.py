"""Built-in sample data for offline demos and CPU-only tests (no downloads)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

# (written, spoken) pairs spanning semiotic classes — used by tests & the demo.
SAMPLE_TN_PAIRS: List[Tuple[str, str]] = [
    ("He paid $5.2M for the company in 1998.",
     "He paid five point two million dollars for the company in nineteen ninety eight."),
    ("The meeting is at 3:30 PM on the 21st.",
     "The meeting is at three thirty p m on the twenty first."),
    ("Dr. Reeves lives at 42 Baker St. downtown.",
     "Doctor Reeves lives at forty two Baker Street downtown."),
    ("About 1984 people attended the concert.",
     "About one thousand nine hundred eighty four people attended the concert."),
    ("Chapter IV covers the 1990s in detail.",
     "Chapter four covers the nineteen nineties in detail."),
    ("The box weighs 5kg and is 80% full.",
     "The box weighs five kilograms and is eighty percent full."),
]

# A short, multi-chapter "book" rich in semiotic tokens — drives the agent demo.
SAMPLE_BOOK = """The Clockmaker's Ledger

Chapter 1: The Inheritance

On March 3, 1921, Mr. Harrow received a letter. It said the estate was worth $1.2M,
held in an account at 1492 Elm St. He had until the 15th to reply.
"This cannot be right," he whispered, reading it twice.

The ledger listed 1,204 entries. Each was dated and signed. Only 3/4 of them made sense.

Chapter 2: The Visitor

At 9:45 AM a stranger arrived. She introduced herself as Dr. Vance and lived on Oak Dr.
"I represent the firm," she said. "You owe approximately 250kg of silver, or 42% of the estate."
He called 555-203-1984 to confirm, then wrote to info@harrow.org for the records.

King Henry VIII once owned the clock, she claimed. It had not run since the 1880s.
"""


def write_sample_book(out_dir: str | Path) -> Path:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    dest = p / "sample_book.txt"
    dest.write_text(SAMPLE_BOOK, encoding="utf-8")
    return dest


__all__ = ["SAMPLE_TN_PAIRS", "SAMPLE_BOOK", "write_sample_book"]
