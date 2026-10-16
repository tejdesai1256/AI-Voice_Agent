import sys, os
sys.path.insert(0, os.path.abspath("."))
from app.database.database import SessionLocal
from app.services.escalation_service import EscalationService

with SessionLocal() as db:
    escs = EscalationService.get_escalations(db)
    print(f"Total escalations: {len(escs)}")
    if escs:
        first = escs[0]
        esc_id = first["id"]
        print(f"Testing on escalation #{esc_id}, current status: {first['status']}, callback_status: {first.get('callback_status')}")
        init_res = EscalationService.initiate_callback(db, esc_id, "Operator test callback")
        print("Initiate callback result:", init_res)
        assert init_res["status"] == "callback_initiated"
        assert init_res["callback_status"] == "Callback Initiated"
        assert init_res["telephony_active"] is False, "Expected integration-ready status when no live carrier is configured"
        assert "pending configuration" in init_res["message"]

        comp_res = EscalationService.complete_callback(db, esc_id, "Customer query discussed")
        print("Complete callback result:", comp_res)
        assert comp_res["status"] == "callback_completed"

        # Verify get_escalations status mapping
        updated = EscalationService.get_escalations(db, status_filter="callback_completed")
        assert any(e["id"] == esc_id for e in updated)

        print("CALLBACK WORKFLOW BACKEND VERIFICATION SUCCESSFUL!")
