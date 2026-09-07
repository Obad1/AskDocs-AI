"""Local passphrase-based auth for Avam Search. No email, no cloud, pure privacy."""
import json
import secrets
from pathlib import Path
from datetime import datetime
from functools import wraps
from flask import session, redirect, url_for, request, jsonify

try:
    from werkzeug.security import generate_password_hash, check_password_hash
except ImportError:
    def generate_password_hash(pw):
        import hashlib
        return hashlib.sha256(pw.encode()).hexdigest()
    def check_password_hash(hash, pw):
        import hashlib
        return hash == hashlib.sha256(pw.encode()).hexdigest()

USERS_FILE = Path("data/sessions/users.json")

def _load_users():
    if USERS_FILE.exists():
        try:
            return json.loads(USERS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_users(users):
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(json.dumps(users, indent=2, ensure_ascii=False), encoding="utf-8")

def user_exists():
    return len(_load_users()) > 0

def register_passphrase(passphrase):
    users = _load_users()
    if users:
        return False, "Passphrase already set. Log in instead."
    salt = secrets.token_hex(8)
    users[salt] = {
        "hash": generate_password_hash(passphrase + salt),
        "created": datetime.now().isoformat(),
        "prefs": {}
    }
    _save_users(users)
    return True, "Passphrase created."

def verify_passphrase(passphrase):
    users = _load_users()
    for salt, data in users.items():
        if check_password_hash(data["hash"], passphrase + salt):
            session["authenticated"] = True
            session["user_id"] = salt
            session["login_time"] = datetime.now().isoformat()
            return True
    return False

def logout():
    session.clear()

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("authenticated"):
            if request.is_json:
                return jsonify({"error": "Authentication required"}), 401
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated
