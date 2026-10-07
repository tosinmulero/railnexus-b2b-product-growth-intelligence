import pandas as pd

from railnexus.synthetic import SyntheticConfig, build_all


def build_small(seed: int = 123):
    return build_all(
        SyntheticConfig(
            n_sessions=2_500,
            n_partners=80,
            n_stations=70,
            seed=seed,
        )
    )


def test_expected_tables() -> None:
    frames = build_small()
    assert set(frames) == {
        "partners",
        "stations",
        "sessions",
        "searches",
        "bookings",
        "partner_events",
    }


def test_unique_primary_keys() -> None:
    frames = build_small(321)
    assert frames["partners"]["partner_id"].is_unique
    assert frames["stations"]["station_id"].is_unique
    assert frames["sessions"]["session_id"].is_unique
    assert frames["searches"]["search_id"].is_unique
    assert frames["bookings"]["booking_id"].is_unique


def test_referential_integrity() -> None:
    frames = build_small(456)
    assert set(frames["sessions"]["partner_id"]).issubset(set(frames["partners"]["partner_id"]))
    assert set(frames["searches"]["session_id"]).issubset(set(frames["sessions"]["session_id"]))
    assert set(frames["bookings"]["search_id"]).issubset(set(frames["searches"]["search_id"]))


def test_zero_results_cannot_be_selected() -> None:
    frames = build_small(777)
    searches = frames["searches"]
    assert searches.query("no_result_flag == 1 and selected_flag == 1").empty


def test_bookings_require_selection() -> None:
    frames = build_small(888)
    searches = frames["searches"]
    bookings = frames["bookings"]
    booked = searches.loc[searches["search_id"].isin(bookings["search_id"])]
    assert (booked["selected_flag"] == 1).all()
    assert (booked["no_result_flag"] == 0).all()


def test_sessions_do_not_precede_partner_launch() -> None:
    frames = build_small(999)
    sessions = frames["sessions"].merge(
        frames["partners"][["partner_id", "launch_date"]],
        on="partner_id",
        how="left",
        validate="many_to_one",
    )
    sessions["session_ts"] = pd.to_datetime(sessions["session_ts"])
    sessions["launch_date"] = pd.to_datetime(sessions["launch_date"])
    assert (sessions["session_ts"] >= sessions["launch_date"]).all()
