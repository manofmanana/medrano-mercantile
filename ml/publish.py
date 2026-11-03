#!/usr/bin/env python3
"""Build the public dashboard data: aggregates only, with no item, consignor, buyer, profit or cost fields."""
import json
import os
import sys
from datetime import datetime, timezone

import psycopg

OUTPUT_DIR = os.environ.get("ML_OUTPUT_DIR", "/opt/mercantile-ml/output")
MIN_CATEGORY_SIZE = 4
CURVE_MAX_POINTS = 60

TOTALS = """
select count(*) filter (where is_sold), count(*), coalesce(sum(sale_price), 0), coalesce(avg(sale_price), 0), max(sold_date)
from marts.dim_items
"""
MONTHLY = """
select to_char(sold_date, 'YYYY-MM'), count(*), sum(sale_price)
from marts.dim_items where is_sold group by 1 order by 1
"""
CATEGORIES = """
with sized as (select *, count(*) over (partition by category) as size from marts.dim_items),
grouped as (select case when size >= %s then category else 'Other' end as grp, is_sold, sale_price from sized)
select grp, count(*), count(*) filter (where is_sold), coalesce(sum(sale_price), 0)
from grouped group by 1 order by 4 desc
"""


def scorecard(report):
    rows = []
    for question, result in (("Sale price from category and name", report["list_price"]), ("Sale price given asking price", report["sale_given_asking"])):
        rows.append({
            "question": question, "baseline": result["best_baseline"], "baseline_score": round(result["results"][result["best_baseline"]]["mae"], 2),
            "model": result["best_model"], "model_score": round(result["results"][result["best_model"]]["mae"], 2),
            "metric": "average error, $", "passes": result["passes_gate"],
        })
    speed = report["time_to_sell"]
    rows.append({
        "question": "Time to sell", "baseline": "shuffled labels", "baseline_score": round(speed["null_c_index"], 2),
        "model": "Cox: category + asking price", "model_score": round(speed["cv_c_index"], 2),
        "metric": "C-index, higher is better", "passes": speed["passes_gate"],
    })
    return rows


def thin(curve):
    step = max(1, len(curve) // CURVE_MAX_POINTS)
    points = curve[::step]
    if points[-1] != curve[-1]:
        points.append(curve[-1])
    return [[round(day, 1), round(share, 3)] for day, share in points]


def build():
    with open(os.path.join(OUTPUT_DIR, "report.json"), encoding="utf-8") as handle:
        report = json.load(handle)
    with psycopg.connect(
        host=os.environ["ML_PG_HOST"], dbname=os.environ["ML_PG_DB"],
        user=os.environ["ML_PG_USER"], password=os.environ["ML_PG_PASSWORD"],
        options="-c default_transaction_read_only=on",
    ) as conn, conn.cursor() as cur:
        sold, listed, revenue, average, last_sale = cur.execute(TOTALS).fetchone()
        monthly = cur.execute(MONTHLY).fetchall()
        categories = cur.execute(CATEGORIES, (MIN_CATEGORY_SIZE,)).fetchall()
    speed = report["time_to_sell"]
    return {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "last_sale_month": last_sale.strftime("%Y-%m") if last_sale else None,
        "totals": {"items_listed": listed, "items_sold": sold, "revenue": float(revenue), "average_sale": round(float(average), 2), "sell_through": round(sold / listed, 3), "median_days_to_sell": speed["median_days"]},
        "monthly": [{"month": m, "items": n, "revenue": float(r)} for m, n, r in monthly],
        "categories": [{"category": g, "items": n, "sold": s, "revenue": float(r)} for g, n, s, r in categories],
        "time_to_sell": {
            "curve": thin(speed["curve"]), "sold_by_30d": round(speed["sold_by_30d"], 3), "sold_by_90d": round(speed["sold_by_90d"], 3),
            "by_category": [{"category": name, "items": c["items"], "sold": c["sold"], "median_days": c["median_days"]} for name, c in sorted(speed["by_category"].items(), key=lambda kv: (kv[1]["median_days"] is None, kv[1]["median_days"] or 0))],
        },
        "scorecard": scorecard(report),
    }


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUTPUT_DIR, "data.json")
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(build(), handle, indent=1)
    print("wrote", target)
