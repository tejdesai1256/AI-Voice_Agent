import asyncio
import os
from datetime import datetime, timezone, timedelta
from typing import Optional, List
from sqlalchemy.orm import Session
from app.database.database import SessionLocal
from app.database.models import Customer, Call
from app.services.conversation_service import ConversationService
from app.services.call_manager_service import CallManagerService


class AutomationScheduler:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self.is_running = False
        self.outbound_queue: List[int] = []  # List of customer IDs
        self.inbound_queue: List[int] = []   # List of customer IDs (priority)

        # State
        self.current_call_id: Optional[int] = None
        self.current_customer_id: Optional[int] = None
        self.current_customer_name: Optional[str] = None
        self.current_call_type: Optional[str] = None

        self.completed_calls: int = 0
        self.completed_history: List[dict] = []
        self.failed_history: List[dict] = []

        self.next_call_at: Optional[datetime] = None
        self.last_call_at: Optional[datetime] = None
        self.last_error: Optional[str] = None

        self._loop_task: Optional[asyncio.Task] = None
        self._active_call_task: Optional[asyncio.Task] = None

        self.conversation_service = ConversationService()
        self.call_manager = CallManagerService()

    def _get_delays(self):
        # Default 5s delay between automated simulation calls for responsive testing
        outbound_delay = int(os.environ.get("AUTOMATION_DELAY_SECONDS", 5))
        inbound_delay = int(os.environ.get("AUTOMATION_INCOMING_DELAY_SECONDS", 2))
        return outbound_delay, inbound_delay

    def get_status(self):
        status_str = "Stopped"
        if self.is_running:
            if self.current_call_id is not None:
                status_str = "In Call"
            elif self.outbound_queue or self.inbound_queue:
                status_str = "Waiting"
            else:
                status_str = "Queue Finished"

        return {
            "is_running": self.is_running,
            "automation_status": status_str,
            "current_call_id": self.current_call_id,
            "current_customer_id": self.current_customer_id,
            "current_customer_name": self.current_customer_name,
            "current_call_type": self.current_call_type,
            "outbound_queue": list(self.outbound_queue),
            "inbound_queue": list(self.inbound_queue),
            "completed_calls": self.completed_calls,
            "completed_history": self.completed_history[-20:],
            "failed_history": self.failed_history[-20:],
            "next_call_at": self.next_call_at.isoformat() if self.next_call_at else None,
            "last_call_at": self.last_call_at.isoformat() if self.last_call_at else None,
            "last_error": self.last_error,
            "simulation_mode": True,
            "disclaimer": "SIMULATION MODE ONLY: Automated calling uses local AI voice simulation for testing. No actual telephone calls are dialed.",
        }

    def start(self, customer_ids: Optional[List[int]] = None):
        """Starts automated calling for explicitly selected customer IDs."""
        if self.is_running:
            return

        self.is_running = True
        self.last_error = None

        with SessionLocal() as db:
            if customer_ids and len(customer_ids) > 0:
                # Filter to ensure IDs actually exist
                valid_ids = [c.id for c in db.query(Customer).filter(Customer.id.in_(customer_ids)).all()]
                self.outbound_queue = valid_ids
            else:
                customers = db.query(Customer).order_by(Customer.id.asc()).all()
                self.outbound_queue = [c.id for c in customers]

        self.next_call_at = datetime.now(timezone.utc)
        self._loop_task = asyncio.create_task(self._main_loop())

    async def stop(self):
        """Safely stops the automated calling scheduler and cancels future calls."""
        self.is_running = False
        if self._active_call_task and not self._active_call_task.done():
            await self.cancel_active_call()
        if self._loop_task:
            self._loop_task.cancel()
            self._loop_task = None
        self.outbound_queue.clear()
        self.inbound_queue.clear()
        self.next_call_at = None

    async def cancel_active_call(self):
        """Forcefully cancels the currently active simulation call."""
        if self._active_call_task and not self._active_call_task.done():
            self._active_call_task.cancel()
            try:
                await self._active_call_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"Error cancelling active call: {e}")

            if self.current_customer_id:
                self.failed_history.append({
                    "customer_id": self.current_customer_id,
                    "customer_name": self.current_customer_name,
                    "reason": "Cancelled by operator",
                    "time": datetime.now(timezone.utc).isoformat(),
                })

            self._active_call_task = None
            self.current_call_id = None
            self.current_customer_id = None
            self.current_customer_name = None
            self.current_call_type = None
            self.last_call_at = datetime.now(timezone.utc)

    def trigger_incoming(self, customer_id: int):
        self.inbound_queue.append(customer_id)
        if self.is_running and self.current_call_id is None and self.next_call_at:
            _, inbound_delay = self._get_delays()
            expedited_time = datetime.now(timezone.utc) + timedelta(seconds=inbound_delay)
            if expedited_time < self.next_call_at:
                self.next_call_at = expedited_time

    async def _main_loop(self):
        try:
            while self.is_running:
                now = datetime.now(timezone.utc)

                # Check active call task
                if self._active_call_task is not None:
                    if self._active_call_task.done():
                        c_id = self.current_customer_id
                        c_name = self.current_customer_name
                        call_id = self.current_call_id
                        try:
                            self._active_call_task.result()
                            self.completed_calls += 1
                            self.completed_history.append({
                                "customer_id": c_id,
                                "customer_name": c_name,
                                "call_id": call_id,
                                "time": datetime.now(timezone.utc).isoformat(),
                            })
                        except Exception as e:
                            self.last_error = str(e)
                            print(f"Automated call task failed: {e}")
                            self.failed_history.append({
                                "customer_id": c_id,
                                "customer_name": c_name,
                                "call_id": call_id,
                                "reason": str(e),
                                "time": datetime.now(timezone.utc).isoformat(),
                            })

                        self.last_call_at = datetime.now(timezone.utc)
                        self.current_call_id = None
                        self.current_customer_id = None
                        self.current_customer_name = None
                        self.current_call_type = None
                        self._active_call_task = None

                        # Schedule next call
                        outbound_delay, inbound_delay = self._get_delays()
                        if self.inbound_queue:
                            self.next_call_at = datetime.now(timezone.utc) + timedelta(seconds=inbound_delay)
                        elif self.outbound_queue:
                            self.next_call_at = datetime.now(timezone.utc) + timedelta(seconds=outbound_delay)
                        else:
                            self.next_call_at = None

                # Ready to start next call
                elif self.next_call_at and now >= self.next_call_at:
                    if self.inbound_queue:
                        next_cust_id = self.inbound_queue.pop(0)
                        call_type = "INBOUND"
                    elif self.outbound_queue:
                        next_cust_id = self.outbound_queue.pop(0)
                        call_type = "OUTBOUND"
                    else:
                        next_cust_id = None
                        self.next_call_at = None

                    if next_cust_id:
                        try:
                            with SessionLocal() as db:
                                customer = db.query(Customer).filter(Customer.id == next_cust_id).first()
                                if customer:
                                    self.current_customer_id = next_cust_id
                                    self.current_customer_name = customer.name
                                    self.current_call_type = call_type

                                    call = self.conversation_service.create_call(db, customer.id)
                                    call.call_type = call_type
                                    call.status = "IN_PROGRESS"
                                    db.commit()
                                    self.current_call_id = call.id

                                    self.next_call_at = None
                                    self._active_call_task = asyncio.create_task(
                                        self.call_manager.run_call_simulation(call.id)
                                    )
                        except Exception as e:
                            self.last_error = str(e)
                            print(f"Failed to initiate automated call: {e}")
                            self.next_call_at = datetime.now(timezone.utc) + timedelta(seconds=5)

                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Scheduler loop error: {e}")
            self.last_error = str(e)
            self.is_running = False
