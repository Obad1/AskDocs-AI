#!/usr/bin/env python3
"""Produce honest beta demo footage from the REAL locally running product UI.

No human chats, payment flow, fake success screens or AI-generated personas.
The generated video is REVIEW ONLY and cannot be auto-published.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

from playwright.sync_api import sync_playwright, expect

HERE=Path(__file__).resolve().parents[1]
OUT=HERE/"artifacts"/"review-demo"
RAW=OUT/"raw"
OUT.mkdir(parents=True,exist_ok=True)
RAW.mkdir(parents=True,exist_ok=True)
BASE="http://127.0.0.1:18882"
events=[]
started=time.monotonic()


def mark(name):
    events.append({"elapsed_sec":round(time.monotonic()-started,2),"scene":name})


def enter(page):
    page.goto(BASE,wait_until="domcontentloaded")
    page.locator("#adult-confirmed").check()
    page.locator("#rules-accepted").check()
    page.locator("#find-stranger").click()
    expect(page.locator("#chat-screen")).to_be_visible()


def wait_for_server(proc):
    for i in range(80):
        if proc.poll() is not None:
            raise RuntimeError("Local app terminated before becoming ready")
        try:
            with urlopen(BASE+"/healthz", timeout=1) as response:
                if response.status==200:
                    return
        except Exception:
            time.sleep(.15)
    raise RuntimeError("Local app not ready")


def produce():
    env=os.environ.copy()
    for name in ("RENDER","TALK_PUBLIC_URL","TALK_PUBLIC_MATCHING_ENABLED",
                 "REPORT_DATABASE_URL","TALK_MODERATOR_SECRET","ADVICE_BOOKINGS_ENABLED",
                 "ADVICE_HOST_AVAILABLE","PAYSTACK_SECRET_KEY","DATABASE_URL"):
        env.pop(name,None)
    process=subprocess.Popen([sys.executable,"-m","uvicorn","app:app","--host","127.0.0.1",
                              "--port","18882","--log-level","warning"],
                             cwd=HERE,env=env,stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    try:
        wait_for_server(process)
        with sync_playwright() as play:
            browser=play.chromium.launch(headless=True,args=["--no-sandbox"])
            viewer=browser.new_context(viewport={"width":405,"height":720},
                                       record_video_dir=str(RAW),
                                       record_video_size={"width":540,"height":960})
            second=browser.new_context(viewport={"width":405,"height":720})
            third=browser.new_context(viewport={"width":405,"height":720})
            fourth=browser.new_context(viewport={"width":405,"height":720})
            first_page=viewer.new_page()
            second_page=second.new_page()
            third_page=third.new_page()
            fourth_page=fourth.new_page()
            first_page.goto(BASE,wait_until="domcontentloaded")
            mark("Authentic product home screen from isolated local app")
            time.sleep(3)
            first_page.locator("#adult-confirmed").check()
            first_page.locator("#rules-accepted").check()
            time.sleep(1)
            first_page.locator("#find-stranger").click()
            expect(first_page.locator("#chat-screen")).to_be_visible()
            mark("Actual queue; one visitor cannot send without a peer")
            expect(first_page.locator("#send")).to_be_disabled()
            time.sleep(2)
            enter(second_page)
            expect(first_page.locator("#send")).to_be_enabled()
            mark("Two real isolated browser sessions matched, no generated bots")
            time.sleep(1.5)
            second_page.locator("#message").fill("Hello from an invented test participant!")
            second_page.locator("#send").click()
            expect(first_page.get_by_text("Hello from an invented test participant!",exact=True)).to_be_visible()
            mark("Real incoming WebSocket message, sample text only")
            time.sleep(3)
            first_page.locator("#message").fill("Hi! What makes a good conversation?")
            first_page.locator("#send").click()
            expect(second_page.get_by_text("Hi! What makes a good conversation?",exact=True)).to_be_visible()
            mark("Actual message delivery to peer")
            time.sleep(3)
            first_page.locator("#next-stranger").click()
            expect(first_page.locator("#send")).to_be_disabled()
            mark("Real Next button disconnects and prevents instant repeated match")
            time.sleep(2)
            enter(third_page)
            expect(second_page.locator("#send")).to_be_enabled()
            expect(third_page.locator("#send")).to_be_enabled()
            expect(first_page.locator("#send")).to_be_disabled()
            enter(fourth_page)
            expect(first_page.locator("#send")).to_be_enabled()
            mark("A different synthetic test visitor matched; no immediate rematch")
            time.sleep(2)
            first_page.locator("#report-stranger").click()
            expect(first_page.locator("#report-form")).to_be_visible()
            first_page.locator("#report-reason").select_option("spam")
            mark("Report form interaction, synthetic incident")
            time.sleep(1)
            first_page.get_by_role("button",name="Report and skip").click()
            first_page.get_by_text("Reference: R-",exact=False).wait_for()
            mark("Actual report response in isolated test; not durable-hosted proof")
            time.sleep(2.5)
            first_page.locator("#leave-room").click()
            first_page.wait_for_url(BASE+"/",timeout=10000)
            mark("Leave returns to home")
            time.sleep(3)
            video=first_page.video
            first_page.close()
            second.close()
            third.close()
            fourth.close()
            viewer.close()
            raw_path=Path(video.path())
            browser.close()
        raw_output=OUT/"raw_capture.webm"
        if raw_path != raw_output:
            shutil.copyfile(raw_path,raw_output)
        mp4=OUT/"talk-with-friend-beta-review.mp4"
        subprocess.run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i",str(raw_output),
                        "-vf","scale=1080:1920:flags=lanczos,fps=30",
                        "-an","-c:v","libx264","-preset","medium","-crf","19",
                        "-movflags","+faststart",str(mp4)],check=True,timeout=200)
        manifest={"status":"REVIEW_ONLY","product":"Talk with a Friend",
                  "source":"Actual local FastAPI app; 3 isolated Playwright browser contexts",
                  "claims_not_verified":["Public matching with live real strangers",
                    "Age assurance","Staffed abuse review","Durable hosted reports",
                    "Payment or therapy availability"],
                  "approval":"HOLD","target_dimensions":"1080x1920",
                  "video_sha256":hashlib.sha256(mp4.read_bytes()).hexdigest(),
                  "duration_approx_sec":round(time.monotonic()-started,1),
                  "scenes":events}
        (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        print(json.dumps({"status":manifest["status"],"sha":manifest["video_sha256"],
                          "scenes":len(events),"output":str(mp4)},indent=2))
    finally:
        process.terminate()
        try: process.wait(timeout=4)
        except subprocess.TimeoutExpired: process.kill()


if __name__=="__main__":
    produce()
