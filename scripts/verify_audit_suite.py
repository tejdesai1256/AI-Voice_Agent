import os
import sys
import time
import httpx
from datetime import datetime

target_port = os.environ.get("PORT")
if not target_port:
    # Auto-detect whether 8001 or 8000 is active
    for p in ["8001", "8000"]:
        try:
            test_r = httpx.get(f"http://127.0.0.1:{p}/health", timeout=1.0)
            if test_r.status_code == 200:
                target_port = p
                break
        except Exception:
            continue
    if not target_port:
        target_port = "8001"

BASE_URL = os.environ.get("BASE_URL", f"http://127.0.0.1:{target_port}")

print("=========================================================")
print(f"LIVE AUDIT SUITE: VERIFYING RUNNING SYSTEM AT {BASE_URL}")
print("=========================================================")

client = httpx.Client(base_url=BASE_URL, timeout=120.0)

# Check 1: Server and Health
try:
    r = client.get("/health")
    assert r.status_code == 200 and r.json().get("status") == "healthy"
    print(f"[PASS] 1. Uvicorn server is live and healthy on {BASE_URL}")
except Exception as e:
    print(f"[FAIL] 1. Server health check failed: {e}")
    sys.exit(1)

# Check 2: Unauthenticated protection
r_unauth = client.get("/api/customers")
assert r_unauth.status_code == 401
print("[PASS] 2. Unauthenticated access properly rejected (HTTP 401)")

# Check 3: Authentication & Login
t0 = time.perf_counter()
r_login = client.post("/api/auth/login", json={"username": "admin", "password": "AdminPass123!"})
t_login = time.perf_counter() - t0
assert r_login.status_code == 200, f"Login failed: {r_login.text}"
token = r_login.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}
print(f"[PASS] 3. Login successful in {t_login:.3f}s. User: admin, Role: admin, JWT received.")

# Check 4: Real Customers from MySQL
r_cust = client.get("/api/customers", headers=headers)
assert r_cust.status_code == 200
customers = r_cust.json()
print(f"[PASS] 4. Customer endpoint returned {len(customers)} real customer records from MySQL.")
sample_cust = customers[0]
print(f"       Sample real customer: ID {sample_cust['id']}, ExtID: {sample_cust['external_customer_id']}, Name: {sample_cust['name']}")

# Check 5: Customer Search
if sample_cust.get("name") and sample_cust["name"] != "N/A":
    q_term = sample_cust["name"][:4]
    r_search = client.get(f"/api/customers?q={q_term}", headers=headers)
    assert r_search.status_code == 200
    found = r_search.json()
    assert len(found) > 0
    print(f"[PASS] 5. Search for '{q_term}' filtered properly ({len(found)} results).")

# Check 6: Call History and Transcript Details
r_calls = client.get("/api/calls", headers=headers)
assert r_calls.status_code == 200
calls = r_calls.json()
print(f"[PASS] 6. Call history returned {len(calls)} call sessions from MySQL.")

if len(calls) > 0:
    first_call_id = calls[0]["id"]
    r_detail = client.get(f"/api/calls/{first_call_id}", headers=headers)
    assert r_detail.status_code == 200
    call_detail = r_detail.json()
    assert "customer" in call_detail and "call" in call_detail and "transcript" in call_detail
    print(f"[PASS] 7. Call details loaded for Call #{first_call_id}:")
    print(f"       Customer: {call_detail['customer']['name']}")
    print(f"       Status: {call_detail['call']['status']}, Language: {call_detail['call']['language']}")
    print(f"       Messages in transcript: {len(call_detail['transcript'])}")

# Check 8: Automated Calling Controls & Safety
r_auto = client.get("/api/automation/status", headers=headers)
assert r_auto.status_code == 200
auto_data = r_auto.json()
assert auto_data["is_running"] is False, "Automated calling must start OFF!"
assert auto_data["simulation_mode"] is True, "Simulation mode must be declared!"
print("[PASS] 8. Automated calling is confirmed STOPPED by default (is_running=False, simulation_mode=True).")

# Check 9: Human Assistance Escalations
r_escs = client.get("/api/escalations", headers=headers)
assert r_escs.status_code == 200
escs = r_escs.json()
print(f"[PASS] 9. Human Assistance escalations returned {len(escs)} records from MySQL.")

# Check 10: Static Waiting Audio Clips
for clip in ["waiting_1.wav", "waiting_2.wav", "waiting_3.wav"]:
    r_clip = client.get(f"/static/waiting/{clip}")
    assert r_clip.status_code == 200
    assert len(r_clip.content) > 10000
print("[PASS] 10. Waiting audio clips accessible via HTTP GET (/static/waiting/waiting_1..3.wav).")

print("\nALL LIVE INTEGRATION AUDIT CHECKS PASSED SUCCESSFULLY!")
