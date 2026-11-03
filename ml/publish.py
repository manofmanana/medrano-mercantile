#!/usr/bin/env python3
"""Build the data behind the public read-only mirror of the Metabase dashboard.

Consignors, payouts, buyers, notes and SKUs are never selected, so they can't reach the page.
"""
import json
import os
import sys
from datetime import date, datetime, timezone
from decimal import Decimal

import psycopg

OUTPUT_DIR = os.environ.get("ML_OUTPUT_DIR", "/opt/mercantile-ml/output")

SALES = """
select item_name as item, category, platform, sold_date as sold, sale_price as price, net_profit, days_to_sell_from_listed as days
from marts.fct_sales order by sold_date
"""
INVENTORY = """
select item_name as item, category, status, asking_price as asking, potential_profit, days_in_inventory as days, aging_bucket as aging, aging_bucket_order as aging_order, is_stale
from marts.fct_inventory order by days_in_inventory desc
"""
CATEGORY_ITEMS = """
select category, count(*) as items, count(*) filter (where is_sold) as sold from marts.dim_items group by 1
"""
REFRESHED = "select refreshed_at from marts.rpt_kpis limit 1"
SCORECARD = """
select question, best_baseline as baseline, round(baseline_score, 2) as baseline_score, best_model as model, round(model_score, 2) as model_score, metric, passes_gate as passes
from ml.model_scorecard order by question
"""
SPEED = "select category, median_days from ml.time_to_sell_by_category where median_days is not null order by median_days"
PREDICTIONS = """
select item_name as item, category, asking_price as asking, days_listed as days, chance_sells_next_30d as chance, estimate_basis as basis
from ml.listed_item_predictions order by chance_sells_next_30d desc nulls last, days_listed desc
"""
TRAINED = "select max(trained_at) from ml.model_scorecard"


def plain(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def rows(cur, sql):
    cur.execute(sql)
    names = [c.name for c in cur.description]
    return [{n: plain(v) for n, v in zip(names, row)} for row in cur.fetchall()]


def build():
    with psycopg.connect(
        host=os.environ["ML_PG_HOST"], dbname=os.environ["ML_PG_DB"],
        user=os.environ["ML_PG_USER"], password=os.environ["ML_PG_PASSWORD"],
        options="-c default_transaction_read_only=on",
    ) as conn, conn.cursor() as cur:
        refreshed = cur.execute(REFRESHED).fetchone()[0]
        trained = cur.execute(TRAINED).fetchone()[0]
        return {
            "published": datetime.now(timezone.utc).isoformat(timespec="minutes"),
            "refreshed": plain(refreshed), "trained": plain(trained),
            "sales": rows(cur, SALES), "inventory": rows(cur, INVENTORY), "category_items": rows(cur, CATEGORY_ITEMS),
            "scorecard": rows(cur, SCORECARD), "median_days": rows(cur, SPEED), "predictions": rows(cur, PREDICTIONS),
        }


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUTPUT_DIR, "data.json")
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(build(), handle, indent=1)
    print("wrote", target)
