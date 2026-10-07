from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

EUROPEAN_HUBS: Final[list[tuple[str, str, float, float]]] = [
    ("London", "GB", 51.5074, -0.1278),
    ("Paris", "FR", 48.8566, 2.3522),
    ("Amsterdam", "NL", 52.3676, 4.9041),
    ("Brussels", "BE", 50.8503, 4.3517),
    ("Berlin", "DE", 52.5200, 13.4050),
    ("Frankfurt", "DE", 50.1109, 8.6821),
    ("Madrid", "ES", 40.4168, -3.7038),
    ("Barcelona", "ES", 41.3874, 2.1686),
    ("Milan", "IT", 45.4642, 9.1900),
    ("Rome", "IT", 41.9028, 12.4964),
    ("Edinburgh", "GB", 55.9533, -3.1883),
    ("Manchester", "GB", 53.4808, -2.2426),
]


@dataclass(frozen=True)
class SyntheticConfig:
    n_sessions: int = 500_000
    n_partners: int = 600
    n_stations: int = 360
    start_date: str = "2026-01-01"
    end_date: str = "2026-06-30"
    seed: int = 20261007


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_partners(cfg: SyntheticConfig) -> pd.DataFrame:
    rng = _rng(cfg.seed + 1)

    partner_types = np.array(
        ["Corporate Travel Platform", "Online Travel Retailer", "Rail Carrier"]
    )
    integration_types = np.array(["API", "Embedded Widget", "White Label"])
    tiers = np.array(["Growth", "Scale", "Enterprise"])
    countries = np.array(["GB", "FR", "DE", "IT", "ES", "NL", "BE"])

    ids = np.array([f"PTR_{i:04d}" for i in range(1, cfg.n_partners + 1)])
    launch_start = pd.Timestamp(cfg.start_date) - pd.Timedelta(days=540)
    launch_end = pd.Timestamp(cfg.start_date) + pd.Timedelta(days=90)
    launch_days = rng.integers(0, (launch_end - launch_start).days + 1, cfg.n_partners)

    df = pd.DataFrame(
        {
            "partner_id": ids,
            "partner_type": rng.choice(partner_types, cfg.n_partners, p=[0.42, 0.38, 0.20]),
            "country": rng.choice(
                countries, cfg.n_partners, p=[0.34, 0.16, 0.14, 0.10, 0.10, 0.09, 0.07]
            ),
            "integration_type": rng.choice(integration_types, cfg.n_partners, p=[0.56, 0.29, 0.15]),
            "contract_tier": rng.choice(tiers, cfg.n_partners, p=[0.48, 0.36, 0.16]),
            "launch_date": launch_start + pd.to_timedelta(launch_days, unit="D"),
            "account_manager_region": rng.choice(
                ["UK&I", "Western Europe", "Central Europe", "Southern Europe"],
                cfg.n_partners,
                p=[0.38, 0.24, 0.20, 0.18],
            ),
        }
    )

    tier_factor = df["contract_tier"].map({"Growth": 0.75, "Scale": 1.15, "Enterprise": 1.65})
    type_factor = df["partner_type"].map(
        {"Corporate Travel Platform": 1.00, "Online Travel Retailer": 1.25, "Rail Carrier": 0.90}
    )
    df["traffic_weight"] = (
        tier_factor * type_factor * rng.lognormal(0, 0.35, cfg.n_partners)
    ).clip(0.15, 4.5)
    return df


def make_stations(cfg: SyntheticConfig) -> pd.DataFrame:
    rng = _rng(cfg.seed + 2)
    hub_idx = rng.integers(0, len(EUROPEAN_HUBS), cfg.n_stations)

    rows = []
    for i, hidx in enumerate(hub_idx, start=1):
        city, country, lat, lon = EUROPEAN_HUBS[hidx]
        rows.append(
            {
                "station_id": f"STN_{i:04d}",
                "station_name": f"RailNexus {city} Station {i:03d}",
                "city": city,
                "country": country,
                "latitude": lat + rng.normal(0, 0.18),
                "longitude": lon + rng.normal(0, 0.24),
                "hub_weight": float(rng.lognormal(mean=0.0, sigma=0.75)),
            }
        )
    return pd.DataFrame(rows)


def make_sessions(cfg: SyntheticConfig, partners: pd.DataFrame) -> pd.DataFrame:
    rng = _rng(cfg.seed + 3)
    start = pd.Timestamp(cfg.start_date)
    end = pd.Timestamp(cfg.end_date) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    seconds = int((end - start).total_seconds())

    partner_probs = partners["traffic_weight"].to_numpy(dtype=float)
    partner_probs = partner_probs / partner_probs.sum()

    partner_ids = rng.choice(partners["partner_id"], cfg.n_sessions, p=partner_probs)
    timestamps = pd.Series(
        start
        + pd.to_timedelta(
            rng.integers(0, seconds + 1, cfg.n_sessions),
            unit="s",
        ),
        dtype="datetime64[ns]",
    )

    launch_lookup = pd.to_datetime(partners.set_index("partner_id")["launch_date"])
    partner_launch = pd.Series(partner_ids).map(launch_lookup)
    prelaunch = timestamps < partner_launch

    if prelaunch.any():
        remaining_seconds = (
            (end - partner_launch.loc[prelaunch]).dt.total_seconds().astype(np.int64).clip(lower=0)
        )
        offsets = (rng.random(int(prelaunch.sum())) * (remaining_seconds.to_numpy() + 1)).astype(
            np.int64
        )
        timestamps.loc[prelaunch] = (
            partner_launch.loc[prelaunch] + pd.to_timedelta(offsets, unit="s")
        ).to_numpy()
    device = rng.choice(
        ["Desktop Web", "Mobile Web", "iOS App", "Android App"],
        cfg.n_sessions,
        p=[0.31, 0.27, 0.23, 0.19],
    )
    channel = rng.choice(
        ["Direct", "Partner Referral", "Paid Search", "Organic Search", "CRM"],
        cfg.n_sessions,
        p=[0.28, 0.34, 0.12, 0.18, 0.08],
    )

    df = pd.DataFrame(
        {
            "session_id": [f"SES_{i:09d}" for i in range(1, cfg.n_sessions + 1)],
            "partner_id": partner_ids,
            "session_ts": timestamps.to_numpy(),
            "device_type": device,
            "acquisition_channel": channel,
            "is_logged_in": rng.random(cfg.n_sessions) < 0.64,
            "traveller_type": rng.choice(["Business", "Leisure"], cfg.n_sessions, p=[0.44, 0.56]),
        }
    )

    # Experiment eligibility and deterministic assignment.
    exp_start = pd.Timestamp("2026-04-01")
    exp_end = pd.Timestamp("2026-05-15 23:59:59")
    eligible = (df["session_ts"] >= exp_start) & (df["session_ts"] <= exp_end)
    assignment_draw = rng.random(cfg.n_sessions)
    df["experiment_name"] = np.where(eligible, "smart_alternatives_v1", "not_eligible")
    df["experiment_variant"] = np.where(
        eligible,
        np.where(assignment_draw < 0.5, "control", "variant"),
        "not_eligible",
    )
    return df


def make_searches(
    cfg: SyntheticConfig,
    sessions: pd.DataFrame,
    stations: pd.DataFrame,
    partners: pd.DataFrame,
) -> pd.DataFrame:
    rng = _rng(cfg.seed + 4)

    # 1–3 searches per session.
    search_count = rng.choice([1, 2, 3], cfg.n_sessions, p=[0.70, 0.24, 0.06])
    session_idx = np.repeat(np.arange(cfg.n_sessions), search_count)
    n = len(session_idx)

    s = sessions.iloc[session_idx].reset_index(drop=True)
    station_probs = stations["hub_weight"].to_numpy()
    station_probs = station_probs / station_probs.sum()

    origin = rng.choice(stations["station_id"], n, p=station_probs)
    destination = rng.choice(stations["station_id"], n, p=station_probs)
    same = origin == destination
    while same.any():
        destination[same] = rng.choice(stations["station_id"], same.sum(), p=station_probs)
        same = origin == destination

    base_no_result = 0.065
    mobile_penalty = np.where(s["device_type"].isin(["Mobile Web", "Android App"]), 0.012, 0.0)
    variant_bonus = np.where(s["experiment_variant"].eq("variant"), -0.018, 0.0)
    no_result_prob = np.clip(base_no_result + mobile_penalty + variant_bonus, 0.01, 0.18)
    no_result = rng.random(n) < no_result_prob

    result_count = np.where(
        no_result,
        0,
        np.clip(rng.poisson(lam=7.2, size=n) + 1, 1, 25),
    )

    latency = (
        rng.lognormal(mean=np.log(520), sigma=0.42, size=n)
        + np.where(s["device_type"].eq("Mobile Web"), 95, 0)
        + np.where(s["experiment_variant"].eq("variant"), 18, 0)
    ).round(0)

    selection_logit = (
        -0.72
        + 0.13 * np.log1p(result_count)
        + 0.30 * s["is_logged_in"].astype(int).to_numpy()
        - 0.00045 * np.maximum(latency - 500, 0)
        - 2.6 * no_result.astype(int)
        + 0.08 * (s["experiment_variant"].eq("variant")).astype(int).to_numpy()
    )
    selection_prob = 1 / (1 + np.exp(-selection_logit))
    selected = (rng.random(n) < selection_prob) & (~no_result)

    df = pd.DataFrame(
        {
            "search_id": [f"SRC_{i:010d}" for i in range(1, n + 1)],
            "session_id": s["session_id"].to_numpy(),
            "partner_id": s["partner_id"].to_numpy(),
            "search_ts": s["session_ts"].to_numpy()
            + pd.to_timedelta(rng.integers(0, 1200, n), unit="s"),
            "origin_station_id": origin,
            "destination_station_id": destination,
            "journey_type": rng.choice(["Single", "Return"], n, p=[0.61, 0.39]),
            "passengers": rng.choice([1, 2, 3, 4], n, p=[0.63, 0.26, 0.08, 0.03]),
            "days_before_travel": np.clip(
                rng.gamma(shape=2.1, scale=6.2, size=n).round(), 0, 90
            ).astype(int),
            "result_count": result_count.astype(int),
            "no_result_flag": no_result.astype(int),
            "search_latency_ms": latency.astype(int),
            "alternatives_shown": (
                (s["experiment_variant"].eq("variant")).to_numpy()
                & (result_count <= 4)
                & (~no_result)
            ).astype(int),
            "selected_flag": selected.astype(int),
            "experiment_variant": s["experiment_variant"].to_numpy(),
        }
    )
    return df


def make_bookings(
    cfg: SyntheticConfig,
    searches: pd.DataFrame,
    sessions: pd.DataFrame,
    partners: pd.DataFrame,
) -> pd.DataFrame:
    rng = _rng(cfg.seed + 5)

    session_lookup = sessions.set_index("session_id")[
        ["device_type", "is_logged_in", "traveller_type"]
    ]
    x = searches.join(session_lookup, on="session_id")

    logit = (
        -1.18
        + 1.00 * x["selected_flag"].to_numpy()
        + 0.32 * x["is_logged_in"].astype(int).to_numpy()
        + 0.16 * x["traveller_type"].eq("Business").astype(int).to_numpy()
        - 2.8 * x["no_result_flag"].to_numpy()
        - 0.00035 * np.maximum(x["search_latency_ms"].to_numpy() - 600, 0)
        + 0.10 * x["experiment_variant"].eq("variant").astype(int).to_numpy()
    )
    p = 1 / (1 + np.exp(-logit))
    booked = (
        (rng.random(len(x)) < p)
        & x["selected_flag"].eq(1).to_numpy()
        & x["no_result_flag"].eq(0).to_numpy()
    )
    b = x.loc[booked].copy().reset_index(drop=True)
    n = len(b)

    base_fare = (
        22
        + 13 * b["passengers"].to_numpy()
        + 1.15 * np.sqrt(np.maximum(b["days_before_travel"].to_numpy(), 1))
        + rng.gamma(shape=2.6, scale=15.0, size=n)
    )
    booking_value = np.round(np.clip(base_fare, 12, 420), 2)

    cancellation_prob = (
        0.032
        + 0.012 * b["journey_type"].eq("Return").astype(int).to_numpy()
        + 0.008 * (booking_value > 140)
    )
    cancelled = rng.random(n) < cancellation_prob
    refunded = cancelled & (rng.random(n) < 0.82)

    df = pd.DataFrame(
        {
            "booking_id": [f"BKG_{i:09d}" for i in range(1, n + 1)],
            "search_id": b["search_id"].to_numpy(),
            "session_id": b["session_id"].to_numpy(),
            "partner_id": b["partner_id"].to_numpy(),
            "booking_ts": b["search_ts"].to_numpy()
            + pd.to_timedelta(rng.integers(60, 1800, n), unit="s"),
            "booking_value": booking_value,
            "platform_revenue": np.round(booking_value * rng.uniform(0.045, 0.085, n), 2),
            "cancelled_flag": cancelled.astype(int),
            "refund_flag": refunded.astype(int),
            "experiment_variant": b["experiment_variant"].to_numpy(),
        }
    )
    return df


def make_partner_events(
    cfg: SyntheticConfig,
    partners: pd.DataFrame,
    searches: pd.DataFrame,
    bookings: pd.DataFrame,
) -> pd.DataFrame:
    rng = _rng(cfg.seed + 6)
    rows: list[dict[str, object]] = []

    first_search = searches.groupby("partner_id")["search_ts"].min()
    first_booking = bookings.groupby("partner_id")["booking_ts"].min()

    for row in partners.itertuples(index=False):
        pid = row.partner_id
        launch = pd.Timestamp(row.launch_date)
        milestones = [
            ("contract_signed", launch - pd.Timedelta(days=int(rng.integers(7, 35)))),
            ("api_key_created", launch),
            ("integration_test", launch + pd.Timedelta(days=int(rng.integers(1, 5)))),
            ("webhook_configured", launch + pd.Timedelta(days=int(rng.integers(2, 8)))),
        ]

        if pid in first_search.index:
            milestones.append(("first_search", pd.Timestamp(first_search.loc[pid])))
        if pid in first_booking.index:
            milestones.append(("first_booking", pd.Timestamp(first_booking.loc[pid])))

        for event_type, event_ts in milestones:
            rows.append(
                {
                    "partner_id": pid,
                    "event_ts": event_ts,
                    "event_type": event_type,
                    "integration_type": row.integration_type,
                    "contract_tier": row.contract_tier,
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(["partner_id", "event_ts", "event_type"])
        .reset_index(drop=True)
    )


def build_all(cfg: SyntheticConfig) -> dict[str, pd.DataFrame]:
    partners = make_partners(cfg)
    stations = make_stations(cfg)
    sessions = make_sessions(cfg, partners)
    searches = make_searches(cfg, sessions, stations, partners)
    bookings = make_bookings(cfg, searches, sessions, partners)
    partner_events = make_partner_events(cfg, partners, searches, bookings)
    return {
        "partners": partners,
        "stations": stations,
        "sessions": sessions,
        "searches": searches,
        "bookings": bookings,
        "partner_events": partner_events,
    }
