"""Keep a public Streamlit Community Cloud app awake with a real browser session.

A plain HTTP GET returns 200 (the static shell) but does NOT reset Streamlit's
inactivity timer, which tracks real viewer sessions (the WebSocket the JS opens).
This loads the app in headless Chromium, searches every frame for the rendered
app (Community Cloud may embed it in an iframe), wakes it if a button (the sleep
prompt) is showing, and stays on the page so the session registers. Exits
non-zero with diagnostics if the app never rendered.
"""
import os
import sys
import time

from playwright.sync_api import sync_playwright

URL = os.environ.get("APP_URL")
if not URL:
    sys.exit("APP_URL not set")

DEADLINE_S = 180
APP = '[data-testid="stApp"]'


def find_app_frame(page):
    for fr in page.frames:
        try:
            if fr.query_selector(APP):
                return fr
        except Exception:
            pass
    return None


def try_wake(page):
    for fr in page.frames:
        try:
            btn = fr.query_selector("button")
            if btn:
                btn.click()
                return True
        except Exception:
            pass
    return False


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        print(f"Visiting {URL}", flush=True)
        page.goto(URL, wait_until="load", timeout=90000)

        clicked = False
        loaded = False
        last_frames = None
        start = time.time()
        while time.time() - start < DEADLINE_S:
            if find_app_frame(page):
                loaded = True
                break
            frames = [f.url for f in page.frames]
            if frames != last_frames:
                print(f"frames: {frames}", flush=True)
                last_frames = frames
            print(f"t={int(time.time() - start)}s no app yet", flush=True)
            if not clicked and try_wake(page):
                clicked = True
                print("  clicked a button (assumed wake control)", flush=True)
            time.sleep(6)

        if loaded:
            print("Streamlit app rendered — session established", flush=True)
            time.sleep(20)  # keep the session open so it counts as activity
            browser.close()
            return 0

        print(f"FAILED within {DEADLINE_S}s. frames={[f.url for f in page.frames]}",
              flush=True)
        try:
            print("top html snippet:", page.content()[:800], flush=True)
        except Exception:
            pass
        browser.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())
