import os
import sys
from fastapi.testclient import TestClient

# Ensure root on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from scripts.create_admin import create_or_update_admin

try:
    _live = httpx.get("http://127.0.0.1:8001/health", timeout=1.0)
    if _live.status_code == 200:
        client = httpx.Client(base_url="http://127.0.0.1:8001", timeout=30.0)
    else:
        from main import app
        client = TestClient(app)
except Exception:
    from main import app
    client = TestClient(app)


def test_complete_backend():
    print("=== STARTING BACKEND SUITE ===")

    # 1. Health check
    res = client.get("/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    print("PASS: Health check /health")

    # 2. Ensure admin exists
    create_or_update_admin("test_admin_suite", "SuiteAdminPass123!", "admin")
    print("PASS: create_or_update_admin")

    # 3. Test Unauthorized Access
    res = client.get("/api/customers")
    assert res.status_code == 401, f"Expected 401 Unauthorized, got {res.status_code}"
    print("PASS: Protected route rejected unauthorized request with 401")

    # 4. Test Login
    login_res = client.post(
        "/api/auth/login",
        json={"username": "test_admin_suite", "password": "SuiteAdminPass123!"},
    )
    assert login_res.status_code == 200, f"Login failed: {login_res.text}"
    login_data = login_res.json()
    token = login_data["access_token"]
    assert token, "Token not returned"
    print(f"PASS: Login successful. User: {login_data['user']['username']} Role: {login_data['user']['role']}")

    headers = {"Authorization": f"Bearer {token}"}

    # 5. Test /api/auth/me
    me_res = client.get("/api/auth/me", headers=headers)
    assert me_res.status_code == 200, f"/me failed: {me_res.text}"
    print(f"PASS: /api/auth/me verified for {me_res.json()['username']}")

    # 6. Test Customer List & Persistence
    cust_res = client.get("/api/customers", headers=headers)
    assert cust_res.status_code == 200, f"Customer list failed: {cust_res.text}"
    customers = cust_res.json()
    assert isinstance(customers, list), "Customers must be a list"
    print(f"PASS: /api/customers returned {len(customers)} real customer records from MySQL")

    # 7. Test Customer Search
    if customers:
        target_cust = next((c for c in customers if c["name"] and c["name"] != "N/A"), customers[0])
        search_term = target_cust["name"] if target_cust["name"] != "N/A" else target_cust["external_customer_id"][:4]
        search_res = client.get(f"/api/customers?q={search_term}", headers=headers)
        assert search_res.status_code == 200
        found = search_res.json()
        assert len(found) > 0, f"Search for '{search_term}' did not return results"
        print(f"PASS: Customer search for '{search_term}' returned {len(found)} results")

    # 8. Test Call History
    calls_res = client.get("/api/calls", headers=headers)
    assert calls_res.status_code == 200, f"Calls list failed: {calls_res.text}"
    calls = calls_res.json()
    assert isinstance(calls, list), "Calls must be a list"
    print(f"PASS: /api/calls returned {len(calls)} call history sessions")

    # 9. Test Call Details with Transcript & Requirements
    if calls:
        call_id = calls[0]["id"]
        detail_res = client.get(f"/api/calls/{call_id}", headers=headers)
        assert detail_res.status_code == 200, f"Call details failed: {detail_res.text}"
        detail = detail_res.json()
        assert "customer" in detail, "Missing customer in call details"
        assert "call" in detail, "Missing call in call details"
        assert "transcript" in detail, "Missing transcript in call details"
        assert "requirements" in detail, "Missing requirements in call details"
        print(f"PASS: /api/calls/{call_id} details loaded (Transcript: {len(detail['transcript'])} messages, Requirements present: {detail['requirements'] is not None})")

    # 10. Test Human Assistance Escalations
    esc_res = client.get("/api/escalations", headers=headers)
    assert esc_res.status_code == 200, f"Escalations list failed: {esc_res.text}"
    escalations = esc_res.json()
    assert isinstance(escalations, list), "Escalations must be a list"
    print(f"PASS: /api/escalations returned {len(escalations)} escalation records")

    # 11. Test Automated Calling Status
    auto_res = client.get("/api/automation/status", headers=headers)
    assert auto_res.status_code == 200, f"Automation status failed: {auto_res.text}"
    auto_status = auto_res.json()
    assert auto_status["is_running"] is False, "CRITICAL ERROR: Automated calling must NOT be running on init!"
    assert auto_status["simulation_mode"] is True, "Simulation mode flag missing"
    print("PASS: Automated calling is safely STOPPED by default (is_running=False, simulation_mode=True)")

    # 12. Test Static Waiting Audio
    for clip in ["waiting_1.wav", "waiting_2.wav", "waiting_3.wav"]:
        clip_path = os.path.join("static", "waiting", clip)
        assert os.path.exists(clip_path), f"Waiting audio clip {clip} missing"
        assert os.path.getsize(clip_path) > 10000, f"Waiting audio clip {clip} is too small"
    print("PASS: Waiting audio clips verified in static/waiting/")

    print("\nALL 12 BACKEND INTEGRATION TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_complete_backend()
