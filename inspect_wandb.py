"""
Diagnostic: dump the raw protobuf structure of records in a .wandb file
so we can see exactly what fields are populated and where the metric
key/value data actually lives.
"""

import argparse
from collections import Counter
from pathlib import Path

from google.protobuf import text_format
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--max-history-dumps", type=int, default=3,
                        help="How many history records to fully dump")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    wandb_file = next(run_dir.glob("run-*.wandb"))
    print(f"[info] inspecting {wandb_file}\n")

    ds = datastore.DataStore()
    ds.open_for_scan(str(wandb_file))

    type_counts = Counter()
    history_dumps = 0
    summary_dumped = False
    stats_dumped = False
    config_dumped = False
    first_record_dumped = False

    while True:
        data = ds.scan_data()
        if data is None:
            break

        rec = wandb_internal_pb2.Record()
        rec.ParseFromString(data)
        rtype = rec.WhichOneof("record_type")
        type_counts[rtype] += 1

        # Dump the very first record regardless of type
        if not first_record_dumped:
            print("=" * 70)
            print(f"FIRST RECORD (type={rtype})")
            print("=" * 70)
            print(text_format.MessageToString(rec, as_utf8=True)[:3000])
            print()
            first_record_dumped = True

        if rtype == "history" and history_dumps < args.max_history_dumps:
            print("=" * 70)
            print(f"HISTORY RECORD #{history_dumps + 1}")
            print("=" * 70)
            # Show available fields on the history sub-message
            h = rec.history
            print(f"  ListFields on rec.history: "
                  f"{[f.name for f, _ in h.ListFields()]}")
            print(f"  step: {h.step}")
            print(f"  len(item): {len(h.item)}")
            if len(h.item) > 0:
                first_item = h.item[0]
                print(f"  ListFields on first item: "
                      f"{[f.name for f, _ in first_item.ListFields()]}")
                # Show first few items in detail
                for i, it in enumerate(h.item[:10]):
                    print(f"    item[{i}]: key={it.key!r}  "
                          f"nested_key={list(it.nested_key)!r}  "
                          f"value_json={it.value_json[:120]!r}")
            print()
            print("Full text dump (truncated):")
            print(text_format.MessageToString(rec, as_utf8=True)[:2000])
            print()
            history_dumps += 1

        elif rtype == "summary" and not summary_dumped:
            print("=" * 70)
            print("SUMMARY RECORD")
            print("=" * 70)
            print(text_format.MessageToString(rec, as_utf8=True)[:3000])
            print()
            summary_dumped = True

        elif rtype == "stats" and not stats_dumped:
            print("=" * 70)
            print("STATS RECORD")
            print("=" * 70)
            print(text_format.MessageToString(rec, as_utf8=True)[:1500])
            print()
            stats_dumped = True

        elif rtype == "config" and not config_dumped:
            print("=" * 70)
            print("CONFIG RECORD (truncated)")
            print("=" * 70)
            print(text_format.MessageToString(rec, as_utf8=True)[:1500])
            print()
            config_dumped = True

    print("=" * 70)
    print("RECORD TYPE COUNTS")
    print("=" * 70)
    for k, v in type_counts.most_common():
        print(f"  {k!r}: {v}")


if __name__ == "__main__":
    main()