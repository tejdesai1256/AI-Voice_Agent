from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy.orm import Session, joinedload
from app.database.models import Customer, Call, Message


class ConversationService:
    def get_or_create_customer(
        self,
        db: Session,
        external_customer_id: str,
        name: Optional[str] = None,
        phone: Optional[str] = None,
    ) -> Customer:
        """
        Retrieves an existing Customer by external_customer_id or creates a new one.
        """
        try:
            customer = (
                db.query(Customer)
                .filter(Customer.external_customer_id == external_customer_id)
                .first()
            )
            if customer:
                return customer

            customer = Customer(
                external_customer_id=external_customer_id,
                name=name,
                phone=phone,
            )
            db.add(customer)
            db.commit()
            db.refresh(customer)
            return customer
        except Exception:
            db.rollback()
            raise

    def create_call(self, db: Session, customer_id: int) -> Call:
        """
        Creates a new Call session for the given customer_id.
        """
        try:
            call = Call(customer_id=customer_id)
            db.add(call)
            db.commit()
            db.refresh(call)
            return call
        except Exception:
            db.rollback()
            raise

    def end_call(self, db: Session, call_id: int, status: str = "RESOLVED") -> Call:
        """
        Marks a call as ended by setting ended_at to current UTC datetime.
        """
        try:
            call = db.query(Call).filter(Call.id == call_id).first()
            if not call:
                raise ValueError(f"Call with id {call_id} not found.")

            call.ended_at = datetime.now(timezone.utc)
            call.status = status
            db.commit()
            db.refresh(call)
            return call
        except Exception:
            db.rollback()
            raise

    def update_call_status(self, db: Session, call_id: int, status: str) -> Call:
        """
        Updates the status of a call.
        """
        try:
            call = db.query(Call).filter(Call.id == call_id).first()
            if not call:
                raise ValueError(f"Call with id {call_id} not found.")

            call.status = status
            db.commit()
            db.refresh(call)
            return call
        except Exception:
            db.rollback()
            raise


    def get_call(self, db: Session, call_id: int) -> Call:
        """
        Retrieve the Call by ID, eagerly loading its messages.
        """
        try:
            call = (
                db.query(Call)
                .options(joinedload(Call.messages))
                .filter(Call.id == call_id)
                .first()
            )
            if not call:
                raise ValueError(f"Call with id {call_id} not found.")
            return call
        except Exception:
            db.rollback()
            raise

    def is_call_active(self, db: Session, call_id: int) -> bool:
        """
        Return True if the call has not ended (ended_at is None).
        """
        call = self.get_call(db, call_id)
        return call.ended_at is None

    def add_message(
        self,
        db: Session,
        call_id: int,
        speaker: str,
        text: str,
    ) -> Message:
        """
        Adds a single conversation message (from customer or assistant) to a call session.
        """
        try:
            message = Message(
                call_id=call_id,
                speaker=speaker,
                text=text,
            )
            db.add(message)
            db.commit()
            db.refresh(message)
            return message
        except Exception:
            db.rollback()
            raise

    def get_call_history(self, db: Session, customer_id: int) -> List[Call]:
        """
        Returns all calls for a given customer_id ordered by started_at ascending,
        with messages eagerly loaded.
        """
        try:
            calls = (
                db.query(Call)
                .options(joinedload(Call.messages))
                .filter(Call.customer_id == customer_id)
                .order_by(Call.started_at.asc())
                .all()
            )
            return calls
        except Exception:
            db.rollback()
            raise
