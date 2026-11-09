# Talk with a Friend — free private-invite chat

An independent AVAMALE lab application for **two people who already know each other**. This is a new implementation: no existing repository, deployment or Wix site with that name was found during the 9 Nov 2026 inventory. Its source is isolated in this directory.

## What works in code

- Accessible responsive 1:1 web interface with an invitation link, no registration or payment.
- Two-person WebSocket rooms authenticated by distinct unguessable seat tokens; maximum two connected participants.
- Ephemeral messages relayed live, **not saved to SQLite or any database**, and not logged as content.
- 12-hour room expiry, 5 room creations/minute/IP, 20 messages/10 seconds/socket, 1,000-character message cap.
- Message DOM insertion uses `textContent` (no untrusted HTML rendering), secret invitation token stays in URL fragment rather than HTTP query logs.
- Optional anonymous nickname, presence, typing indicators, reconnect UX, and host-controlled room closure.
- Security headers, strict Origin match when `TALK_PUBLIC_URL` is configured, TLS on host.

## Local smoke test

```bash
cd talk-with-a-friend
python -m pip install -r requirements.txt httpx pytest
uvicorn app:app --host 127.0.0.1 --port 8000
```

Visit `http://127.0.0.1:8000`, start a conversation, copy the invite into another browser profile, then exchange messages. The app does not need a paid API, database, email or AI subscription.

```bash
python -m pytest -q tests
node --check static/app.js
python -m py_compile app.py
```

## Free-tier deployment recommendation

- Use **Render Free** on a separate service attached to the feature branch, build `cd talk-with-a-friend && pip install -r requirements.txt`, start `cd talk-with-a-friend && uvicorn app:app --host 0.0.0.0 --port $PORT`.
- Set `TALK_PUBLIC_URL=https://<actual-provider-hostname>` after a hostname is assigned. Never assume a URL before provisioning.
- User visits automatically wake free service; active WebSocket sessions require the instance to stay awake. Free-tier sleep/restarts kill active rooms and there are no delivery or uptime guarantees.
- If usage grows, upgrade reliability using an allowed free-tier managed database or coordination store **only after** checking real terms and abuse/privacy safeguards. Do not charge customers or add paid resources without authorization.

## Limits and safety

- **NOT end-to-end encrypted.** TLS protects transit, but the server processes messages to relay them.
- Anyone holding the invitation can take the guest seat. Share links privately; avoid sending secrets, passwords, banking details or sensitive records.
- No moderation, abuse-reporting, account recovery or guaranteed availability. For use among acquaintances, not anonymous public matching or crisis counseling.
- No durable message history, file transfers, voice/video calls, notifications, friend directory, search, or shared user accounts.
- Rooms and chat contents are lost on server restart; client holds only text visible during the open session.
- Before public broad release add abuse reporting and blocking, spam protections across multiple instances, suitable privacy/terms notices, accessibility QA, load tests and host monitoring.

## Acceptance

1. CI tests pass for room creation, wrong token, two-person attendance, chat relay, rate cap, room end, and no persistence.
2. On two separate devices, join via a real invite link; exchange at least 10 messages both ways; verify identity separation, disconnect/reconnect, and room expiration.
3. Test portrait/landscape mobile layouts, keyboard navigation, screen-reader announcement order and long unbroken messages.
4. Confirm no chargeable plan or billable add-on was created. Treat quotas/idle-sleep as real limits.
