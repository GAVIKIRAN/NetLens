"""Show what the IDS recorded for one source: run this after a controlled test.

    python scripts/show_detections.py --source 192.168.1.44
    python scripts/show_detections.py --source 192.168.1.44 --minutes 30

Prints only what is stored in the database, so a test either shows a detection or it
doesn't.  Times are UTC.  It only reads rows, but opening the database applies any
pending schema upgrade, as the dashboard does.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

from _bootstrap import bootstrap


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--source", required=True, help="source IP address of the test machine")
    parser.add_argument("--minutes", type=float, default=60.0, help="how far back to look")
    args = parser.parse_args()
    settings = bootstrap(args.config)

    from src.database.database import get_database

    database = get_database(settings)
    since = (datetime.now(timezone.utc) - timedelta(minutes=args.minutes)).strftime("%Y-%m-%dT%H:%M:%S")

    events = database.query(
        "SELECT timestamp, decision, kind, dst_ip, dst_port, risk_score, risk_level, repeats, reason "
        "FROM behaviour_events WHERE src_ip = ? AND timestamp >= ? ORDER BY id",
        (args.source, since),
    )
    print(f"\nBehavioural detections from {args.source}, last {args.minutes:.0f} min: {len(events)}")
    for row in events.itertuples():
        print(f"  {row.timestamp}  {row.decision:<10} risk {row.risk_score:>5.1f} {row.risk_level:<8} {row.reason}")

    flows = database.query(
        "SELECT p.prediction, p.final_classification AS final, COUNT(*) AS flows, "
        "COUNT(DISTINCT f.dst_port) AS ports "
        "FROM flows f JOIN predictions p ON p.flow_id = f.id "
        "WHERE f.src_ip = ? AND p.timestamp >= ? GROUP BY 1, 2 ORDER BY flows DESC",
        (args.source, since),
    )
    print(f"\nScored flows from {args.source} (a lone SYN never becomes a flow):")
    if flows.empty:
        print("  none")
    for row in flows.itertuples():
        print(f"  model: {row.prediction:<20} final: {str(row.final):<11} flows: {row.flows:>5}  ports: {row.ports}")

    alerts = database.query(
        "SELECT COUNT(*) AS n FROM alerts WHERE src_ip = ? AND timestamp >= ?",
        (args.source, since),
    )
    print(f"\nAlerts: {int(alerts.iloc[0]['n'])}")
    if events.empty:
        print(
            "\nNo behavioural detection. Check that LIVE mode was running on the right "
            "interface during the test, and that the scan reached this machine."
        )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
