"""Inspect a running local Gradio app at desktop and mobile widths."""

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:7867")
    parser.add_argument("--output-dir", default="test-results")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        for name, width, height in (("desktop", 1280, 900), ("mobile", 390, 844)):
            page.set_viewport_size({"width": width, "height": height})
            page.goto(args.url, wait_until="domcontentloaded")
            page.get_by_role("heading", name="Twi–French", exact=True).wait_for()
            assert page.get_by_text("Awaiting checkpoint", exact=True).count() or page.locator("textarea").count()
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
            page.screenshot(path=str(output / f"gradio-{name}.png"), full_page=True)
        browser.close()
    print("Desktop/mobile layout checks passed.")


if __name__ == "__main__":
    main()
