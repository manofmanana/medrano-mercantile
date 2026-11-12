#!/usr/bin/env python3
"""Record one pipeline step's outcome in ml.pipeline_runs. Grafana alerts read this table.

Usage: record_run.py STEP ok|failed [DETAIL]
"""
import os
import sys

import psycopg


def main():
    step, status = sys.argv[1], sys.argv[2]
    detail = sys.argv[3] if len(sys.argv) > 3 else None
    with psycopg.connect(
        host=os.environ["ML_PG_HOST"], dbname=os.environ["ML_PG_DB"],
        user=os.environ["ML_PG_WRITE_USER"], password=os.environ["ML_PG_WRITE_PASSWORD"],
    ) as conn:
        conn.execute("insert into ml.pipeline_runs (step, ok, detail) values (%s, %s, %s)",
                     (step, status == "ok", detail))


if __name__ == "__main__":
    main()
