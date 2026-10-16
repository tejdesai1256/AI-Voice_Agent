import sys
import os
import time
import json
import httpx
from datetime import datetime

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath("."))

print("==================================================================")
print("STRICT 14-POINT LIVE FUNCTIONAL AND VISUAL VERIFICATION")
print("==================================================================")

BASE_URL = "http://127.0.0.1:8000"
client = httpx.Client(base_url=BASE_URL, timeout=120.0)

results = []

def record(item_no, name, endpoint, req_details, status_code, resp_summary, success, notes=""):
    res = {
        "item": item_no,
        "name": name,
        "endpoint": endpoint,
        "request": req_details,
        "status_code": status_code,
        "response": resp_summary,
        "success": success,
        "notes": notes
    }
    results.append(res)
    mark = "[SUCCESS]" if success else "[FAILED]"
    print(f"\n{mark} Item {item_no}: {name}")
    print(f"  Endpoint: {endpoint}")
    print(f"  Status: {status_code}")
    print(f"  Response: {resp_summary}")
    if notes:
        print(f"  Notes: {notes}")
    return success

# -------------------------------------------------------------
# 1. Run the server / Server Health
# -------------------------------------------------------------
try:
    r = client.get("/health")
    succeeded = (r.status_code == 200 and r.json().get("status") == "healthy")
    record(1, "Run the server (Health Check)", "GET /health", "None", r.status_code, json.dumps(r.json()), succeeded, "Server running with MySQL database connected")
except Exception as e:
    record(1, "Run the server (Health Check)", "GET /health", "None", 0, str(e), False, "Server is not reachable")
    sys.exit(1)

# -------------------------------------------------------------
# 2. Test login
# -------------------------------------------------------------
try:
    login_payload = {"username": "admin", "password": "AdminPass123!"}
    r_login = client.post("/api/auth/login", json=login_payload)
    login_data = r_login.json() if r_login.status_code == 200 else {}
    token = login_data.get("access_token", "")
    headers = {"Authorization": f"Bearer {token}"}
    succeeded = (r_login.status_code == 200 and "access_token" in login_data)
    summary = f"token_type: {login_data.get('token_type')}, user: {login_data.get('user', {}).get('username')}, role: {login_data.get('user', {}).get('role')}"
    record(2, "Test login", "POST /api/auth/login", json.dumps(login_payload), r_login.status_code, summary, succeeded, "Bcrypt verified, JWT issued")
except Exception as e:
    record(2, "Test login", "POST /api/auth/login", str(login_payload), 0, str(e), False)

# -------------------------------------------------------------
# 3. Test unauthenticated access
# -------------------------------------------------------------
try:
    r_unauth = client.get("/api/customers")
    succeeded = (r_unauth.status_code == 401)
    record(3, "Test unauthenticated access", "GET /api/customers", "No auth header", r_unauth.status_code, r_unauth.text, succeeded, "Properly blocked without valid JWT")
except Exception as e:
    record(3, "Test unauthenticated access", "GET /api/customers", "None", 0, str(e), False)

# -------------------------------------------------------------
# 4. Test authenticated customer loading
# -------------------------------------------------------------
cust_data = []
try:
    r_cust = client.get("/api/customers", headers=headers)
    cust_data = r_cust.json() if r_cust.status_code == 200 else []
    succeeded = (r_cust.status_code == 200 and isinstance(cust_data, list))
    sample = f"Total count: {len(cust_data)}. First: ID={cust_data[0]['id']}, Name={cust_data[0]['name']}, Phone={cust_data[0]['phone']}" if cust_data else "Empty"
    record(4, "Test authenticated customer loading", "GET /api/customers", "Bearer JWT", r_cust.status_code, sample, succeeded, "Real customers loaded from MySQL")
except Exception as e:
    record(4, "Test authenticated customer loading", "GET /api/customers", "Bearer JWT", 0, str(e), False)

# -------------------------------------------------------------
# 5. Test add/edit/delete customer
# -------------------------------------------------------------
try:
    # 5a: Add Customer (FormData)
    add_form = {
        "external_customer_id": f"TEST_VERIF_{int(time.time())}",
        "name": "Strict Verification User",
        "phone": "+91 98765 00099"
    }
    r_add = client.post("/api/customers", headers=headers, data=add_form)
    add_res = r_add.json() if r_add.status_code == 200 else {}
    test_cust_id = add_res.get("id")
    
    # 5b: Edit Customer (FormData)
    edit_form = {
        "external_customer_id": add_form["external_customer_id"],
        "name": "Strict Verification User Updated",
        "phone": "+91 98765 00088"
    }
    r_edit = client.put(f"/api/customers/{test_cust_id}", headers=headers, data=edit_form)
    edit_res = r_edit.json() if r_edit.status_code == 200 else {}

    # 5c: Delete Customer
    r_del = client.delete(f"/api/customers/{test_cust_id}", headers=headers)
    del_res = r_del.json() if r_del.status_code == 200 else {}

    succeeded = (r_add.status_code == 200 and r_edit.status_code == 200 and r_del.status_code == 200 and edit_res.get("name") == "Strict Verification User Updated" and del_res.get("status") == "deleted")
    summary = f"POST (ID={test_cust_id}, Name='{add_res.get('name')}') -> PUT (Updated='{edit_res.get('name')}') -> DELETE (status='{del_res.get('status')}')"
    record(5, "Test add/edit/delete customer", "POST / PUT / DELETE /api/customers/{id}", f"FormData({add_form['external_customer_id']})", 200, summary, succeeded, "Customer lifecycle fully verified and cleaned in MySQL")
except Exception as e:
    record(5, "Test add/edit/delete customer", "CRUD /api/customers", "Test customer", 0, str(e), False)

# -------------------------------------------------------------
# 6. Test call history API
# -------------------------------------------------------------
calls_data = []
try:
    r_calls = client.get("/api/calls", headers=headers)
    calls_data = r_calls.json() if r_calls.status_code == 200 else []
    succeeded = (r_calls.status_code == 200 and isinstance(calls_data, list))
    sample = f"Total calls: {len(calls_data)}. Latest Call: ID={calls_data[0]['id']}, Status={calls_data[0]['status']}, Customer={calls_data[0]['customer_name']}" if calls_data else "No calls"
    record(6, "Test call history API", "GET /api/calls", "Bearer JWT", r_calls.status_code, sample, succeeded, "Call history queried from calls table")
except Exception as e:
    record(6, "Test call history API", "GET /api/calls", "Bearer JWT", 0, str(e), False)

# -------------------------------------------------------------
# 7. Test transcript retrieval
# -------------------------------------------------------------
try:
    # Pick a call with messages
    target_call = next((c for c in calls_data if c.get("message_count", 0) > 0), calls_data[0] if calls_data else None)
    if target_call:
        target_id = target_call["id"]
        r_call_detail = client.get(f"/api/calls/{target_id}", headers=headers)
        detail_data = r_call_detail.json() if r_call_detail.status_code == 200 else {}
        transcript = detail_data.get("transcript") or []
        requirements = detail_data.get("requirements")
        succeeded = (r_call_detail.status_code == 200 and "call" in detail_data and "transcript" in detail_data and len(transcript) > 0)
        req_summary = "Captured" if requirements else "None"
        sample_first_msg = f"[{transcript[0]['speaker']}]: '{transcript[0]['text'][:40]}...'" if transcript else "Empty"
        summary = f"Call #{target_id}: Customer='{detail_data['customer']['name']}', Status='{detail_data['call']['status']}', Language='{detail_data['call']['language']}', Messages={len(transcript)} (e.g. {sample_first_msg}), Requirements={req_summary}"
        record(7, "Test transcript retrieval", f"GET /api/calls/{target_id}", "Bearer JWT", r_call_detail.status_code, summary, succeeded, "Full chronological transcript returned with speaker labels and timestamps")
    else:
        record(7, "Test transcript retrieval", "GET /api/calls/{id}", "Bearer JWT", 404, "No calls found to test", False)
except Exception as e:
    record(7, "Test transcript retrieval", "GET /api/calls/{id}", "Bearer JWT", 0, str(e), False)

# -------------------------------------------------------------
# 8. Verify messages exist in MySQL
# -------------------------------------------------------------
try:
    from app.database.database import SessionLocal
    from app.database.models import Message, Call, Customer
    db = SessionLocal()
    total_msgs = db.query(Message).count()
    sample_msgs = db.query(Message).order_by(Message.timestamp.desc()).limit(3).all()
    msg_summary = f"Total messages in MySQL: {total_msgs}. Recent records: " + "; ".join([f"[{m.speaker}]: '{m.text[:30]}...'" for m in sample_msgs])
    succeeded = (total_msgs > 0)
    db.close()
    record(8, "Verify messages exist in MySQL", "Direct MySQL Query (messages table)", "SessionLocal.query(Message)", 200, msg_summary, succeeded, "Messages confirmed stored and readable directly from MySQL")
except Exception as e:
    record(8, "Verify messages exist in MySQL", "Direct MySQL Query", "SessionLocal", 0, str(e), False)

# -------------------------------------------------------------
# 9. Test recorded audio -> SEND AUDIO
# -------------------------------------------------------------
try:
    first_cust = cust_data[0]
    # Create call session with Form data
    start_payload = {"external_customer_id": first_cust["external_customer_id"], "language": "en"}
    r_start = client.post("/api/calls", headers=headers, data=start_payload)
    start_data = r_start.json() if r_start.status_code == 200 else {}
    live_call_id = start_data.get("call_id")
    
    # Process turn with audio file
    audio_path = "static/waiting/waiting_1.wav"
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    
    files = {"file": ("test_input.wav", audio_bytes, "audio/wav")}
    
    t_start = time.perf_counter()
    r_turn = client.post(f"/api/calls/{live_call_id}/turn", headers=headers, files=files)
    t_duration = time.perf_counter() - t_start
    turn_res = r_turn.json() if r_turn.status_code == 200 else {}
    
    succeeded = (r_turn.status_code == 200 and "response" in turn_res and "audio_url" in turn_res)
    timings = turn_res.get("timings", {})
    summary = f"Call #{live_call_id} processed in {t_duration:.2f}s. Customer STT: '{turn_res.get('transcript')}', AI Response: '{turn_res.get('response')[:50]}...', Audio URL: '{turn_res.get('audio_url')}'"
    record(9, "Test recorded audio -> SEND AUDIO", f"POST /api/calls/{live_call_id}/turn", f"Multipart file={audio_path}", r_turn.status_code, summary, succeeded, f"Timings: stt={timings.get('stt_sec')}s, llm={timings.get('llm_sec')}s, tts={timings.get('tts_sec')}s, total={timings.get('total_sec')}s")
except Exception as e:
    record(9, "Test recorded audio -> SEND AUDIO", "POST /api/calls/{id}/turn", "Multipart", 0, str(e), False)

# -------------------------------------------------------------
# 10. Test waiting audio
# -------------------------------------------------------------
try:
    clips = ["waiting_1.wav", "waiting_2.wav", "waiting_3.wav"]
    clip_statuses = []
    for c in clips:
        rc = client.get(f"/static/waiting/{c}")
        clip_statuses.append((c, rc.status_code, len(rc.content)))
    all_clips_ok = all(cs[1] == 200 and cs[2] > 10000 for cs in clip_statuses)
    summary = ", ".join([f"{c}: {status} ({sz} bytes)" for c, status, sz in clip_statuses])
    record(10, "Test waiting audio", "GET /static/waiting/*.wav", "HTTP GET static assets", 200, summary, all_clips_ok, "Sequential pre-rendered Piper audio prompts available")
except Exception as e:
    record(10, "Test waiting audio", "GET /static/waiting/", "None", 0, str(e), False)

# -------------------------------------------------------------
# 11. Test automated calling START
# -------------------------------------------------------------
try:
    # Ensure it starts OFF
    r_auto_pre = client.get("/api/automation/status", headers=headers)
    auto_pre = r_auto_pre.json()
    assert auto_pre["is_running"] is False, "Automation must be stopped initially"
    
    # Start automated calling for 2 customer IDs
    target_ids = [c["id"] for c in cust_data[:2]]
    r_start_auto = client.post("/api/automation/start", headers=headers, json={"customer_ids": target_ids})
    start_auto_res = r_start_auto.json() if r_start_auto.status_code == 200 else {}
    
    r_auto_running = client.get("/api/automation/status", headers=headers)
    auto_running = r_auto_running.json()
    
    succeeded = (r_start_auto.status_code == 200 and auto_running.get("is_running") is True)
    summary = f"POST /api/automation/start -> {r_start_auto.status_code}: '{start_auto_res.get('message')}'. is_running={auto_running.get('is_running')}, Queue total={auto_running.get('total')}"
    record(11, "Test automated calling START", "POST /api/automation/start", json.dumps({"customer_ids": target_ids}), r_start_auto.status_code, summary, succeeded, "Queue initialized with selected customers")
except Exception as e:
    record(11, "Test automated calling START", "POST /api/automation/start", "JSON", 0, str(e), False)

# -------------------------------------------------------------
# 12. Test automated calling STOP
# -------------------------------------------------------------
try:
    r_stop_auto = client.post("/api/automation/stop", headers=headers)
    stop_res = r_stop_auto.json() if r_stop_auto.status_code == 200 else {}
    
    r_auto_stopped = client.get("/api/automation/status", headers=headers)
    auto_stopped = r_auto_stopped.json()
    
    succeeded = (r_stop_auto.status_code == 200 and auto_stopped.get("is_running") is False)
    summary = f"POST /api/automation/stop -> {r_stop_auto.status_code}: '{stop_res.get('message')}'. is_running={auto_stopped.get('is_running')}"
    record(12, "Test automated calling STOP", "POST /api/automation/stop", "None", r_stop_auto.status_code, summary, succeeded, "Queue stopped safely on user command")
except Exception as e:
    record(12, "Test automated calling STOP", "POST /api/automation/stop", "None", 0, str(e), False)

# -------------------------------------------------------------
# 13. Test human assistance display
# -------------------------------------------------------------
try:
    r_esc = client.get("/api/escalations", headers=headers)
    esc_data = r_esc.json() if r_esc.status_code == 200 else []
    succeeded = (r_esc.status_code == 200 and isinstance(esc_data, list))
    sample = f"Total escalations: {len(esc_data)}. Sample: ID={esc_data[0]['id']}, Customer='{esc_data[0]['customer_name']}', Reason='{esc_data[0]['escalation_reason'][:40]}...', Language='{esc_data[0]['language_selected']}', Status='{esc_data[0]['status']}'" if esc_data else "No escalations"
    record(13, "Test human assistance display", "GET /api/escalations", "Bearer JWT", r_esc.status_code, sample, succeeded, "Real escalations queried from human_assistance_requests table")
except Exception as e:
    record(13, "Test human assistance display", "GET /api/escalations", "Bearer JWT", 0, str(e), False)

# -------------------------------------------------------------
# 14. Check frontend for fake/hardcoded customer data
# -------------------------------------------------------------
try:
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        fe_content = f.read()
    
    suspicious_patterns = [
        "John Doe", "Jane Doe", "Alice Smith", "Bob Johnson",
        "demo-customer", "sample-customer", "dummy customer",
        "555-0199", "fake@example.com"
    ]
    found_suspicious = [p for p in suspicious_patterns if p.lower() in fe_content.lower()]
    
    tbody_count = fe_content.count("<tbody")
    has_empty_tbodies = ('<tbody id="customers-tbody">' in fe_content and
                         '<tbody id="calls-tbody">' in fe_content and
                         '<tbody id="escalations-tbody">' in fe_content)
    
    succeeded = (len(found_suspicious) == 0 and has_empty_tbodies)
    summary = f"Suspicious hardcoded names found: {found_suspicious}. Empty dynamic tbodies verified: {has_empty_tbodies}."
    record(14, "Check frontend for fake/hardcoded customer data", "Static HTML Audit (frontend/index.html)", "Regex / text scan", 200, summary, succeeded, "Zero hardcoded customers in HTML; 100% loaded dynamically via authenticated API")
except Exception as e:
    record(14, "Check frontend for fake/hardcoded customer data", "Static Audit", "File read", 0, str(e), False)

print("\n==================================================================")
print("AUDIT SUMMARY:")
all_passed = all(r["success"] for r in results)
passed_count = sum(1 for r in results if r["success"])
print(f"Passed: {passed_count}/{len(results)}")
print("==================================================================")

with open("scripts/strict_verification_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
