"""Browser acceptance tests: real button clicks, two independently created visitors.

Uses a local FastAPI process and Playwright Chromium; no production users,
payments or external APIs. Screenshots are retained by GitHub Actions.
"""
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import pytest
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:18877"
SCREENSHOTS = HERE / "artifacts" / "screenshots"
SCREENSHOTS.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="module")
def live_app():
    env = os.environ.copy()
    for variable in ("RENDER", "TALK_PUBLIC_URL", "ADVICE_BOOKINGS_ENABLED",
                     "ADVICE_HOST_AVAILABLE", "PAYSTACK_SECRET_KEY", "DATABASE_URL"):
        env.pop(variable, None)
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1",
         "--port", "18877", "--log-level", "warning"], cwd=HERE,
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    try:
        for _ in range(80):
            try:
                with urlopen(BASE + "/healthz", timeout=1) as response:
                    if response.status == 200:
                        break
            except Exception:
                time.sleep(.15)
        else:
            raise AssertionError("Local service never became available")
        yield BASE
    finally:
        server.terminate()
        try:server.wait(timeout=4)
        except subprocess.TimeoutExpired:server.kill()


@pytest.fixture(scope="module")
def chromium():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=["--no-sandbox"])
        try:yield browser
        finally:browser.close()


def enter_queue(page, origin):
    page.goto(origin, wait_until="networkidle")
    page.locator("#adult-confirmed").check()
    page.locator("#rules-accepted").check()
    page.locator("#find-stranger").click()
    page.locator("#chat-screen").wait_for(state="visible")
    page.locator("#presence-text").wait_for(state="visible")


def test_strangers_exchange_text_report_and_leave(chromium, live_app, tmp_path):
    context_a = chromium.new_context(viewport={"width": 390, "height": 844})
    context_b = chromium.new_context(viewport={"width": 390, "height": 844})
    try:
        a = context_a.new_page()
        b = context_b.new_page()
        enter_queue(a, live_app)
        assert a.locator("#report-stranger").is_disabled()
        assert a.locator("#next-stranger").is_disabled()
        enter_queue(b, live_app)
        a.locator("#send").wait_for(state="visible")
        a.locator("#send").evaluate("(el) => { if(el.disabled) throw Error('send still disabled'); }")
        b.locator("#send").evaluate("(el) => { if(el.disabled) throw Error('send still disabled'); }")
        a.locator("#message").fill("Hello from an actual test visitor")
        a.locator("#send").click()
        b.get_by_text("Hello from an actual test visitor").wait_for()
        b.locator("#message").fill("Hello right back")
        b.locator("#send").click()
        a.get_by_text("Hello right back").wait_for()
        a.screenshot(path=str(SCREENSHOTS / "chat-mobile.png"), full_page=True)

        a.locator("#report-stranger").click()
        a.locator("#report-panel").wait_for(state="visible")
        a.locator("#report-reason").select_option("spam")
        a.get_by_role("button", name="Report and skip").click()
        a.get_by_text("Reference: R-", exact=False).wait_for()
        b.locator("#send").evaluate("(el) => { if(!el.disabled) throw Error('reported partner still active'); }")

        # The leave button uses the server ACK and returns to the homepage.
        a.locator("#leave-room").click()
        a.wait_for_url(live_app + "/", timeout=10000)
        assert a.locator("#hero").is_visible()
        b.locator("#leave-room").click()
        b.wait_for_url(live_app + "/", timeout=10000)
    finally:
        context_a.close()
        context_b.close()


def test_review_button_and_clean_no-payment_state(chromium, live_app, tmp_path):
    context = chromium.new_context(viewport={"width": 390, "height": 844})
    try:
        page = context.new_page()
        page.goto(live_app, wait_until="networkidle")
        page.locator("#booking-form").wait_for(state="hidden")
        assert page.locator("#advice-interest").is_visible()
        page.locator("#open-review").click()
        page.locator("#review-panel").wait_for(state="visible")
        page.locator("#review-rating").select_option("4")
        page.locator("#review-issue").select_option("layout")
        page.locator("#submit-review").click()
        page.get_by_text("Review received", exact=False).wait_for()
        page.screenshot(path=str(SCREENSHOTS / "review-mobile.png"), full_page=True)
        page.locator("#close-review").click()
        page.locator("#review-panel").wait_for(state="hidden")
        page.locator("#open-review-top").click()
        page.locator("#review-panel").wait_for(state="visible")
        page.keyboard.press("Escape")
        page.locator("#review-panel").wait_for(state="hidden")
    finally:
        context.close()


def test_desktop_first_impression(chromium, live_app, tmp_path):
    page = chromium.new_page(viewport={"width": 1360, "height": 900})
    try:
        page.goto(live_app, wait_until="networkidle")
        assert page.get_by_role("button", name="Meet a stranger").is_visible()
        assert page.locator("#home-availability").is_visible()
        page.screenshot(path=str(SCREENSHOTS / "home-desktop.png"), full_page=True)
    finally:
        page.close()
