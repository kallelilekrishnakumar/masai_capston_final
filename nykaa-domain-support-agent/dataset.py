"""Seeded synthetic order data generator and validator."""
from __future__ import annotations
import random
from collections import Counter
from dataclasses import asdict, dataclass

CATEGORIES = ["Apparel", "Electronics", "Home", "Footwear", "Beauty"]
STATUSES = ["Placed", "Shipped", "Delivered", "Returned", "Refunded"]
SEED = 20261011

@dataclass(frozen=True)
class Order:
    record_id: str
    category: str
    status: str
    order_value_inr: int
    days_since_created: int
    delayed_shipment: bool

def _generate_with_seed(seed: int, n: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    # Guarantee category and status coverage through the seeded sampling plan, not hand edits.
    for i in range(n):
        category = CATEGORIES[i % len(CATEGORIES)] if i < len(CATEGORIES) * 3 else rng.choice(CATEGORIES)
        status = STATUSES[i] if i < len(STATUSES) else rng.choice(STATUSES)
        rows.append(asdict(Order(
            record_id=f"NYK-{i+1:04d}",
            category=category,
            status=status,
            order_value_inr=rng.randint(299, 24999),
            days_since_created=rng.randint(0, 30),
            delayed_shipment=(rng.random() < 0.20),
        )))
    return rows

def generate_orders(seed: int = SEED, n: int = 50) -> list[dict]:
    if n < 40:
        raise ValueError("n must be at least 40")
    # Reseed with deterministic offsets until the delayed-shipment rate lands in the required band.
    attempt = 0
    while True:
        rows = _generate_with_seed(seed + attempt, n)
        rate = sum(x["delayed_shipment"] for x in rows) / len(rows)
        if 0.10 <= rate <= 0.30:
            break
        attempt += 1
        if attempt > 100:
            raise RuntimeError("Could not generate delay rate in required band")
    validate_orders(rows)
    return rows

def validate_orders(rows: list[dict]) -> None:
    assert len(rows) >= 40
    assert all(r["category"] in CATEGORIES for r in rows)
    assert all(r["status"] in STATUSES for r in rows)
    assert all(isinstance(r["days_since_created"], int) and 0 <= r["days_since_created"] <= 30 for r in rows)
    assert all(isinstance(r["delayed_shipment"], bool) for r in rows)
    categories, statuses = Counter(r["category"] for r in rows), Counter(r["status"] for r in rows)
    assert all(categories[c] >= 3 for c in CATEGORIES), categories
    assert all(statuses[s] >= 1 for s in STATUSES), statuses
    pct = 100 * sum(r["delayed_shipment"] for r in rows) / len(rows)
    assert 10 <= pct <= 30, pct

def escalation_score(row: dict) -> float:
    """Designed escalation score in [0, 1] combining BOTH signals (so it is not a
    bare boolean of delayed_shipment): a normalized recency term
    (days_since_created / 30) plus the delayed_shipment flag, gated by order status
    so already-resolved orders (Delivered/Returned/Refunded) de-escalate.

        recency = days_since_created / 30
        status_factor = 1.0 if status in {Placed, Shipped} else 0.5
        score = status_factor * (0.4 * delayed_shipment + 0.6 * recency)
    """
    recency = min(1.0, row["days_since_created"] / 30.0)
    status_factor = 1.0 if row["status"] in ("Placed", "Shipped") else 0.5
    return round(min(1.0, max(0.0, status_factor * (0.4 * int(row["delayed_shipment"]) + 0.6 * recency))), 3)

def escalation_threshold(orders: list[dict] | None = None, percentile: float = 80) -> float:
    """Escalation threshold as a stated percentile of the score distribution."""
    import math
    scores = sorted(escalation_score(o) for o in (orders if orders is not None else ORDERS))
    if not scores:
        return 0.0
    k = (len(scores) - 1) * percentile / 100.0
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return scores[int(k)]
    return round(scores[f] + (scores[c] - scores[f]) * (k - f), 3)

ORDERS = generate_orders()
ESCALATION_THRESHOLD = escalation_threshold()  # 80th percentile of the score distribution

def distribution(rows: list[dict] = ORDERS) -> dict:
    ages = sorted(r["days_since_created"] for r in rows)
    return {
        "count": len(rows),
        "category_counts": dict(Counter(r["category"] for r in rows)),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "delayed_count": sum(r["delayed_shipment"] for r in rows),
        "delayed_percentage": round(100 * sum(r["delayed_shipment"] for r in rows) / len(rows), 2),
        "days_since_created_sorted": ages,
        "p80_days_since_created": ages[min(len(ages)-1, int(0.80 * len(ages)))],
    }

if __name__ == "__main__":
    import json
    print(json.dumps(distribution(), indent=2))
