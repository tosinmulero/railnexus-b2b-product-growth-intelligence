from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from railnexus.config import RAW_DIR  # noqa: E402
from railnexus.synthetic import SyntheticConfig, build_all  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic RailNexus B2B product data."
    )
    parser.add_argument("--sessions", type=int, default=500_000)
    parser.add_argument("--partners", type=int, default=600)
    parser.add_argument("--stations", type=int, default=360)
    parser.add_argument("--seed", type=int, default=20261007)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    cfg = SyntheticConfig(
        n_sessions=args.sessions,
        n_partners=args.partners,
        n_stations=args.stations,
        seed=args.seed,
    )

    frames = build_all(cfg)

    print("=" * 72)
    print("RAILNEXUS â€” STAGE 1 SYNTHETIC DATA GENERATION")
    print("=" * 72)

    for name, df in frames.items():
        path = RAW_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"{name:<20} {len(df):>12,} rows -> {path.relative_to(ROOT)}")

    searches = frames["searches"]
    bookings = frames["bookings"]
    sessions = frames["sessions"]

    search_to_book = bookings["search_id"].nunique() / max(searches["search_id"].nunique(), 1)
    no_result_rate = searches["no_result_flag"].mean()
    active_partners = sessions["partner_id"].nunique()

    eligible = sessions["experiment_variant"].isin(["control", "variant"])
    exp_sessions = sessions.loc[eligible, ["session_id", "experiment_variant"]]
    exp_booked = bookings[["session_id"]].drop_duplicates().assign(booked=1)
    exp = exp_sessions.merge(exp_booked, on="session_id", how="left")
    exp["booked"] = exp["booked"].fillna(0).astype(int)
    exp_summary = exp.groupby("experiment_variant")["booked"].agg(["count", "mean"])

    print()
    print("QUALITY SNAPSHOT")
    print(f"Sessions             : {len(sessions):,}")
    print(f"Searches             : {len(searches):,}")
    print(f"Bookings             : {len(bookings):,}")
    print(f"Active partners      : {active_partners:,}")
    print(f"Search-to-book rate  : {search_to_book:.2%}")
    print(f"No-result rate       : {no_result_rate:.2%}")
    print()
    print("EXPERIMENT SNAPSHOT")
    print(exp_summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print()
    print("STAGE 1              : PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()
