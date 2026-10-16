import re
import os
import sys
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.database import engine
from main import app

def get_all_routes(router, prefix=""):
    routes = []
    for r in router.routes:
        p = prefix + getattr(r, "path", "")
        if hasattr(r, "routes"):
            routes.extend(get_all_routes(r, p))
        elif hasattr(r, "router"):
            routes.extend(get_all_routes(r.router, p))
        else:
            methods = ",".join(getattr(r, "methods", []))
            routes.append((methods, p))
    return routes

all_registered = get_all_routes(app)
registered_routes = set()
for methods, p in all_registered:
    registered_routes.add(p)
    print(f"[{methods:16}] {p}")

print("\n=== 2. AUDITING FRONTEND FETCH ENDPOINTS IN index.html ===")
with open("frontend/index.html", "r", encoding="utf-8") as f:
    html_content = f.read()

# Extract fetch URLs
fetch_matches = re.findall(r"fetch\((?:'|\"|`)([^'\"`]+)(?:'|\"|`)", html_content)
print(f"Total fetch calls in frontend: {len(fetch_matches)}")
all_match = True
for raw_url in sorted(set(fetch_matches)):
    url_base = raw_url.split("?")[0]
    # Replace JS string interpolation ${...} with dummy or check prefix
    clean_url = re.sub(r"\$\{[^}]+\}", "1", url_base)
    # Check if this matches a route pattern
    matched = False
    for r in registered_routes:
        # turn /api/customers/{id} into a regex pattern
        pattern = "^" + re.sub(r"\{[^}]+\}", r"[^/]+", r) + "$"
        if re.match(pattern, clean_url) or re.match(pattern, url_base):
            matched = True
            break
    if matched:
        print(f"  [VALID MATCH] {raw_url} -> matches backend route")
    else:
        print(f"  [POSSIBLE MISMATCH] {raw_url}")
        all_match = False

if all_match:
    print("PASS: 100% of frontend fetch URLs map to registered backend FastAPI routes!")

print("\n=== 3. SEARCHING FOR FAKE OR HARDCODED CUSTOMER DATA ===")
fake_patterns = [
    r"John Doe", r"Jane Doe", r"Demo Customer", r"Sample Customer",
    r"fake", r"mock_customer", r"dummy_data", r"mock_outbound"
]
found_fakes = []
for p in fake_patterns:
    matches = re.findall(p, html_content, re.IGNORECASE)
    if matches:
        found_fakes.append((p, len(matches)))
if found_fakes:
    print("Found patterns in frontend:", found_fakes)
else:
    print("PASS: Zero fake customer names, dummy CRM records, or mock data found in frontend/index.html")

print("\n=== 4. CURRENT REAL MYSQL DATABASE AUDIT ===")
with engine.connect() as conn:
    for t in ["system_users", "customers", "calls", "messages", "requirements", "human_assistance_requests"]:
        cnt = conn.execute(text(f"SELECT COUNT(*) FROM {t};")).scalar()
        print(f"Table {t:28}: {cnt} rows")

    print("\n--- SAMPLE REAL CUSTOMERS FROM MYSQL ---")
    cust_rows = conn.execute(text("SELECT id, external_customer_id, name, phone FROM customers LIMIT 5;")).fetchall()
    for row in cust_rows:
        print(f"ID: {row[0]:<4} ExtID: {row[1]:<25} Name: {str(row[2]):<20} Phone: {str(row[3])}")

    print("\n--- SAMPLE REAL CALLS WITH TRANSCRIPT MESSAGES ---")
    call_rows = conn.execute(text("SELECT id, customer_id, status, language, started_at FROM calls ORDER BY id DESC LIMIT 3;")).fetchall()
    for r in call_rows:
        msg_count = conn.execute(text(f"SELECT COUNT(*) FROM messages WHERE call_id = {r[0]};")).scalar()
        print(f"Call #{r[0]}: Customer #{r[1]}, Status: {r[2]}, Language: {r[3]}, Messages in DB: {msg_count}")
        if msg_count > 0:
            msgs = conn.execute(text(f"SELECT speaker, text FROM messages WHERE call_id = {r[0]} ORDER BY id ASC LIMIT 3;")).fetchall()
            for m in msgs:
                print(f"   [{m[0]}]: {m[1][:80]}...")

    print("\n--- SAMPLE REAL HUMAN ESCALATIONS FROM MYSQL ---")
    esc_rows = conn.execute(text("SELECT id, call_id, customer_name, language_selected, escalation_reason, status FROM human_assistance_requests ORDER BY id DESC LIMIT 3;")).fetchall()
    for e in esc_rows:
        print(f"Escalation #{e[0]}: Call #{e[1]}, Customer: {e[2]}, Lang: {e[3]}, Status: {e[5]}")
        print(f"   Reason: {e[4]}")
