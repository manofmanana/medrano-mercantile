"""Weekly business email: last week's sales, the year so far with hours and profit per hour,
and this week's reprice list.

Set on the Airflow host, never in this repo:
  connection mercantile_ro  Postgres, the read-only role on the mercantile database
  connection gmail_smtp     host smtp.gmail.com, port 587, login = Gmail address, password = Gmail app password
  variable weekly_report_to where to send it
"""
import html
import smtplib
from datetime import datetime, timedelta
from email.message import EmailMessage

from airflow import DAG
from airflow.hooks.base import BaseHook
from airflow.models import Variable
from airflow.operators.python import PythonOperator

WEEK = """
select count(*) filter (where sold_date > current_date - 7),
       coalesce(sum(sale_price) filter (where sold_date > current_date - 7), 0),
       coalesce(sum(net_profit) filter (where sold_date > current_date - 7), 0),
       coalesce(sum(sale_price) filter (where sold_date > current_date - 14 and sold_date <= current_date - 7), 0)
from marts.fct_sales
"""
SOLD = """
select item_name, sale_price, platform, sold_date from marts.fct_sales
where sold_date > current_date - 7 order by sold_date, sale_price desc
"""
YEAR = """
select coalesce(sum(revenue), 0), coalesce(sum(net_profit), 0), coalesce(sum(total_hours), 0),
       coalesce(sum(logged_hours), 0)
from marts.rpt_hours_monthly where month_start >= date_trunc('year', current_date)
"""
STOCK = "select open_items, open_inventory_value, stale_items, refreshed_at from marts.rpt_kpis"
REPRICE = """
select action, sku, item_name, asking_price, suggested_price, reason
from marts.rpt_reprice_suggestions where action <> 'Hold'
order by action_order, days_in_inventory desc nulls last
"""
HAULS = """
select haul, items_sold, items, revenue, total_hours, profit_per_hour
from marts.rpt_haul_performance where haul <> '(no haul)' order by first_received_date desc limit 5
"""
LAST_TRAIN = "select max(finished_at) from ml.pipeline_runs where step = 'train' and ok"


def money(value):
    return "-" if value is None else f"${value:,.0f}"


def table(headers, rows):
    head = "".join(f'<th style="text-align:left;padding:4px 10px 4px 0;border-bottom:1px solid #ddd">{h}</th>'
                   for h in headers)
    body = "".join("<tr>" + "".join(f'<td style="padding:4px 10px 4px 0;border-bottom:1px solid #eee">'
                                    f"{html.escape(str(c))}</td>" for c in row) + "</tr>" for row in rows)
    return f'<table style="border-collapse:collapse;font-size:14px">{head and "<tr>" + head + "</tr>"}{body}</table>'


def build_and_send():
    import psycopg2

    c = BaseHook.get_connection("mercantile_ro")
    with psycopg2.connect(host=c.host, port=c.port or 5432, dbname=c.schema, user=c.login, password=c.password,
                          options="-c default_transaction_read_only=on") as conn, conn.cursor() as cur:
        def one(sql):
            cur.execute(sql)
            return cur.fetchone()

        def all_rows(sql):
            cur.execute(sql)
            return cur.fetchall()

        sold_n, week_rev, week_profit, prev_rev = one(WEEK)
        sold = all_rows(SOLD)
        year_rev, year_profit, year_hours, logged_hours = one(YEAR)
        open_items, open_value, stale, refreshed = one(STOCK)
        reprice = all_rows(REPRICE)
        hauls = all_rows(HAULS)
        cur.execute(LAST_TRAIN)
        last_train = cur.fetchone()[0]

    per_hour = year_profit / year_hours if year_hours else None
    hours_note = ("all logged" if logged_hours and logged_hours == year_hours
                  else "estimated" if not logged_hours else f"{logged_hours:.0f} logged, the rest estimated")
    change = "" if not prev_rev else f" ({(week_rev - prev_rev) / prev_rev:+.0%} vs the week before)"

    parts = [
        '<div style="font-family:system-ui,sans-serif;color:#2e353b;max-width:760px">',
        "<h2 style='margin:0 0 8px'>Medrano Mercantile, week ending "
        f"{datetime.now():%b %d}</h2>",
        f"<p><b>Last 7 days:</b> {sold_n} sold for {money(week_rev)}{change}, {money(week_profit)} profit.</p>",
        f"<p><b>This year:</b> {money(year_rev)} revenue, {money(year_profit)} profit, {year_hours:.0f} hours "
        f"({hours_note}), so <b>{money(per_hour)} per hour</b>.</p>",
        f"<p><b>On hand:</b> {open_items} items listed at {money(open_value)}, {stale} of them 60+ days old.</p>",
    ]
    if sold:
        parts += ["<h3>Sold this week</h3>",
                  table(["Item", "Price", "Platform", "Date"],
                        [(n, money(p), pl, f"{d:%a %b %d}") for n, p, pl, d in sold])]
    if reprice:
        parts += [f"<h3>Do this week ({len(reprice)} items)</h3>",
                  table(["Action", "SKU", "Item", "Asking", "New price", "Why"],
                        [(a, s, n, money(ap), money(sp) if sp is not None else "", r)
                         for a, s, n, ap, sp, r in reprice])]
    if hauls:
        parts += ["<h3>Recent hauls</h3>",
                  table(["Haul", "Sold", "Revenue", "Hours", "Per hour"],
                        [(h, f"{s}/{i}", money(r), f"{hr:.1f}", money(ph)) for h, s, i, r, hr, ph in hauls])]
    parts += [f"<p style='color:#696e7b;font-size:12px'>Data refreshed {refreshed:%b %d %H:%M} UTC. "
              f"Models last trained {last_train:%b %d %H:%M} UTC.</p>" if last_train else
              f"<p style='color:#696e7b;font-size:12px'>Data refreshed {refreshed:%b %d %H:%M} UTC.</p>",
              "</div>"]

    message = EmailMessage()
    smtp = BaseHook.get_connection("gmail_smtp")
    message["From"] = f"Medrano Mercantile <{smtp.login}>"
    message["To"] = Variable.get("weekly_report_to")
    message["Subject"] = f"Weekly: {money(week_rev)} sold, {len(reprice)} items to reprice"
    message.set_content(f"{sold_n} sold for {money(week_rev)} in the last 7 days. "
                        f"This year {money(year_profit)} profit over {year_hours:.0f} hours. "
                        "Open the HTML version for the reprice list.")
    message.add_alternative("".join(parts), subtype="html")
    with smtplib.SMTP(smtp.host, smtp.port or 587, timeout=30) as server:
        server.starttls()
        server.login(smtp.login, smtp.password)
        server.send_message(message)


with DAG(
    dag_id="mercantile_weekly_report",
    default_args={"owner": "alejandro", "retries": 2, "retry_delay": timedelta(minutes=10)},
    schedule="0 15 * * 0",  # Sunday 15:00 UTC, early morning Pacific
    start_date=datetime(2026, 11, 1),
    catchup=False,
    tags=["mercantile", "report"],
) as dag:
    PythonOperator(task_id="send_weekly_email", python_callable=build_and_send)
