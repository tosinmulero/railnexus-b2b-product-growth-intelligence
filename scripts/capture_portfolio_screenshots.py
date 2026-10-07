from __future__ import annotations

from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "screenshots"
BASE_URL = "http://127.0.0.1:8501"

PAGES = [
    ("Executive Overview", "01_executive_overview.png"),
    ("Partner Growth", "02_partner_growth.png"),
    ("Experimentation", "03_experimentation.png"),
    ("Network Intelligence", "04_network_intelligence.png"),
    ("Predictive Modelling", "05_predictive_modelling.png"),
]


def settle(page: Page) -> None:
    page.wait_for_timeout(2400)
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(500)


def capture(page: Page, label: str, filename: str) -> None:
    if label != "Executive Overview":
        radio = page.get_by_role("radio", name=label)
        radio.check(force=True)
    settle(page)
    page.screenshot(
        path=str(OUTPUT / filename),
        full_page=False,
        animations="disabled",
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=["--disable-dev-shm-usage", "--no-sandbox"],
        )
        context = browser.new_context(
            viewport={"width": 1600, "height": 1000},
            device_scale_factor=1,
            color_scheme="dark",
        )
        page = context.new_page()
        page.goto(BASE_URL, wait_until="domcontentloaded", timeout=120_000)
        page.get_by_text("RailNexus", exact=True).first.wait_for(timeout=120_000)
        settle(page)

        for label, filename in PAGES:
            capture(page, label, filename)
            print(f"Captured {label}: {filename}")

        browser.close()

    images = sorted(OUTPUT.glob("0*_*.png"))
    if len(images) != len(PAGES):
        raise RuntimeError(f"Expected {len(PAGES)} screenshots, found {len(images)}.")
    for image in images:
        if image.stat().st_size < 40_000:
            raise RuntimeError(f"Screenshot looks too small: {image}")


if __name__ == "__main__":
    main()
