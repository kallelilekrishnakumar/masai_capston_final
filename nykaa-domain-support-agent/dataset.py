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

def generate_orders(seed: int = SEED, n: int = 50) -> list[dict]:
    if n < 40:
        raise ValueError("n must be at least 40")
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
    rate = sum(x["delayed_shipment"] for x in rows) / len(rows)
    # If the seeded sample is outside the required band, regenerate with deterministic seed offsets.
    attempt = 0
    while not 0.10 <= rate <= 0.30:
        attempt += 1
        if attempt > 100:
            raise RuntimeError("Could not generate delay rate in required band")
        return generate_orders(seed + attempt, n)
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

ORDERS = generate_orders()

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
