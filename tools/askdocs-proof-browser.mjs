// Isolated proof capture of the REAL AskDocs React app (no production writes).
// Runs on a review-only branch. Never substitute mocked UI or fabricated answers.
import { chromium } from "playwright";
import fs from "node:fs";
import path from "node:path";

const OUT = "proof";
const FRAMES = path.join(OUT, "frames");
fs.mkdirSync(FRAMES, { recursive: true });
const qa = {
  product: "AskDocs AI",
  kind: "real React source UI in local Vite; NOT hosted-site proof",
  url: "http://127.0.0.1:5173/",
  syntheticDataOnly: true,
  runAtUTC: new Date().toISOString(),
  source: "Obad1/AskDocs-AI (current workflow branch)",
  gates: {
    appRendered: false,
    sampleCTAVisible: false,
    modalOpened: false,
    ingestionSucceeded: false,
    questionSubmitted: false,
    answerShown: false,
    retrievedPassagesVisible: false,
    mobileAppRendered: false,
    mobileSampleCTAVisible: false
  },
  steps: [], errors: [], runtimeErrors: [], networkErrors: [], frames: 0
};
let browser;
let frameCount = 0;
let cdp;
let screenActive = false;
const pause = (ms) => new Promise((r) => setTimeout(r, ms));

async function shot(page, name) {
  try {
    await page.screenshot({ path: path.join(OUT, name + ".png"), fullPage: false, timeout: 15000 });
    qa.steps.push(name);
  } catch (e) {
    qa.errors.push("screenshot " + name + ": " + String(e).slice(0, 220));
  }
}

try {
  browser = await chromium.launch({
    headless: true,
    executablePath: process.env.CHROME_PATH || "/usr/bin/google-chrome",
    args: ["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu-sandbox"]
  });
  const desktop = await browser.newContext({
    viewport: { width: 1360, height: 840 },
    deviceScaleFactor: 1
  });
  await desktop.addInitScript(() => {
    try { localStorage.setItem("askdocs.onboarded", "1"); } catch {}
  });
  const page = await desktop.newPage();
  page.on("pageerror", (e) => qa.runtimeErrors.push("desktop: " + String(e).slice(0, 500)));
  page.on("requestfailed", (r) => qa.networkErrors.push((r.method() + " " + r.url()).slice(0, 300)));
  const url = qa.url;
  await page.goto(url, { waitUntil: "domcontentloaded", timeout: 40000 });
  await pause(3200);
  qa.gates.appRendered = (await page.locator("#root").innerText()).length > 50;
  qa.desktopTitle = await page.title();
  await shot(page, "01-desktop-start");

  // CDP screencast saves actual Chromium compositor frames. No video emulation.
  try {
    cdp = await desktop.newCDPSession(page);
    cdp.on("Page.screencastFrame", (ev) => {
      try {
        frameCount += 1;
        if (frameCount <= 320) {
          fs.writeFileSync(path.join(FRAMES, String(frameCount).padStart(5, "0") + ".jpg"), Buffer.from(ev.data, "base64"));
        }
      } catch (e) {
        qa.errors.push("frame writing: " + String(e).slice(0, 100));
      }
      void cdp.send("Page.screencastFrameAck", { sessionId: ev.sessionId }).catch(() => {});
    });
    await cdp.send("Page.startScreencast", { format: "jpeg", quality: 78, everyNthFrame: 2 });
    screenActive = true;
  } catch (e) {
    qa.errors.push("screen recording unavailable: " + String(e).slice(0, 200));
  }

  const firstUse = page.getByRole("button", { name: /Try sample documents/i }).first();
  qa.gates.sampleCTAVisible = await firstUse.isVisible().catch(() => false);
  if (qa.gates.sampleCTAVisible) {
    await firstUse.click({ timeout: 12000 });
    qa.gates.modalOpened = await page.getByRole("dialog", { name: /Interactive demo/i }).isVisible().catch(() => false);
    await pause(750);
    await shot(page, "02-real-demo-modal");
    if (qa.gates.modalOpened) {
      await page.getByRole("button", { name: "Load sample dataset", exact: true }).click({ timeout: 12000 });
      await pause(3300);
      await shot(page, "03-real-indexing");
      const status = page.locator('[role="dialog"]');
      let outcome = "timeout";
      try {
        await Promise.race([
          status.getByText(/Demo workspace ready/i).waitFor({ state: "visible", timeout: 105000 }).then(() => { outcome = "ready"; }),
          status.getByText(/Demo load failed/i).waitFor({ state: "visible", timeout: 105000 }).then(() => { outcome = "failed"; })
        ]);
      } catch (e) {
        qa.errors.push("index outcome timeout: " + String(e).slice(0, 140));
      }
      qa.ingestionOutcome = outcome;
      qa.gates.ingestionSucceeded = outcome === "ready";
      qa.modalMessage = (await status.innerText().catch(() => "")).slice(0, 1400);
      await shot(page, "04-real-index-result");
      if (qa.gates.ingestionSucceeded) {
        const close = status.getByRole("button", { name: "Close", exact: true }).last();
        await close.click();
        await pause(700);
        await shot(page, "05-real-workspace");
        const prompt = page.getByRole("textbox", { name: "AskDocs prompt" }).first();
        if (await prompt.isEnabled().catch(() => false)) {
          await prompt.fill("Where does the Calvin cycle take place?");
          await page.getByRole("button", { name: "Run selected task" }).first().click({ timeout: 12000 });
          qa.gates.questionSubmitted = true;
          await pause(13000);
          await shot(page, "06-real-question");
          const messages = page.locator('[data-tour="chat"] .msg-ai');
          qa.answerText = (await messages.last().innerText().catch(() => "")).slice(0, 1100);
          qa.gates.answerShown = !!qa.answerText && !/^(Thinking…|Thinking\.\.\.)/.test(qa.answerText);
          qa.gates.retrievedPassagesVisible = (await page.locator('[data-tour="citations"]').count()) > 0;
          await shot(page, "07-real-answer-evidence");
        } else {
          qa.errors.push("sample indexed, but chat prompt remained disabled");
        }
      }
    }
  } else {
    qa.errors.push("New user could not see Try sample documents CTA on desktop");
  }
  qa.desktopOverflowPx = await page.evaluate(() => Math.max(0, document.documentElement.scrollWidth - innerWidth));
  if (screenActive) {
    try { await cdp.send("Page.stopScreencast"); } catch {}
  }
  await desktop.close();

  const mobile = await browser.newContext({
    viewport: { width: 390, height: 844 }, deviceScaleFactor: 1, isMobile: true, hasTouch: true
  });
  await mobile.addInitScript(() => {
    try { localStorage.setItem("askdocs.onboarded", "1"); } catch {}
  });
  const mpage = await mobile.newPage();
  mpage.on("pageerror", (e) => qa.runtimeErrors.push("mobile: " + String(e).slice(0, 500)));
  await mpage.goto(url, { waitUntil: "domcontentloaded", timeout: 40000 });
  await pause(3300);
  qa.gates.mobileAppRendered = (await mpage.locator("#root").innerText()).length > 50;
  qa.gates.mobileSampleCTAVisible = await mpage.getByRole("button", { name: /Try sample documents/i }).isVisible().catch(() => false);
  qa.mobileOverflowPx = await mpage.evaluate(() => Math.max(0, document.documentElement.scrollWidth - innerWidth));
  await shot(mpage, "08-mobile-start");
  if (qa.gates.mobileSampleCTAVisible) {
    try {
      await mpage.getByRole("button", { name: /Try sample documents/i }).click({ timeout: 12000 });
      await shot(mpage, "09-mobile-demo-modal");
      qa.mobileDemoModalOpened = await mpage.getByRole("dialog", { name: /Interactive demo/i }).isVisible().catch(() => false);
    } catch (e) { qa.errors.push("mobile CTA click: " + String(e).slice(0, 250)); }
  }
  await mobile.close();
} catch (e) {
  qa.errors.push("fatal: " + String(e).slice(0, 1500));
} finally {
  if (browser) await browser.close().catch(() => {});
  qa.frames = Math.min(320, frameCount);
  qa.fullPathProven = Object.values(qa.gates).every(Boolean);
  qa.status = qa.fullPathProven ? "LOCAL_SOURCE_UI_PASS_NOT_HOSTED" : "HOLD_INCOMPLETE_OR_FAILED";
  fs.writeFileSync(path.join(OUT, "qa.json"), JSON.stringify(qa, null, 2));
  console.log(JSON.stringify({status:qa.status,gates:qa.gates,errors:qa.errors,mobileOverflowPx:qa.mobileOverflowPx,frames:qa.frames}));
}
