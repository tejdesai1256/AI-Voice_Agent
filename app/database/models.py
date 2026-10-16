from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship, synonym
from app.database.database import Base


def utc_now():
    return datetime.now(timezone.utc)


class SystemUser(Base):
    __tablename__ = "system_users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), default="system_user", nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)


class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, index=True)
    external_customer_id = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=True)
    phone = Column(String(50), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    # Relationships
    calls = relationship("Call", back_populates="customer", cascade="all, delete-orphan")
    escalations = relationship("HumanAssistanceRequest", back_populates="customer", cascade="all, delete-orphan")


class Call(Base):
    __tablename__ = "calls"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False, index=True)
    status = Column(String(50), default="IN_PROGRESS", nullable=False)
    call_type = Column(String(50), default="OUTBOUND", nullable=False)
    recording_url = Column(String(255), nullable=True)
    language = Column(String(50), default="en", nullable=True)
    started_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    ended_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    customer = relationship("Customer", back_populates="calls")
    messages = relationship("Message", back_populates="call", cascade="all, delete-orphan")
    requirement = relationship("Requirement", back_populates="call", uselist=False, cascade="all, delete-orphan")
    escalations = relationship("HumanAssistanceRequest", back_populates="call", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    call_id = Column(Integer, ForeignKey("calls.id"), nullable=False, index=True)
    speaker = Column(String(50), nullable=False)  # 'customer' or 'assistant'
    text = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    # Relationships
    call = relationship("Call", back_populates="messages")


class Requirement(Base):
    __tablename__ = "requirements"

    id = Column(Integer, primary_key=True, index=True)
    call_id = Column(Integer, ForeignKey("calls.id"), unique=True, nullable=False, index=True)
    status = Column(String(50), default="COLLECTED", nullable=False)
    structured_requirements = Column(JSON, nullable=False)
    confirmed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    # Relationships
    call = relationship("Call", back_populates="requirement")


class HumanAssistanceRequest(Base):
    __tablename__ = "human_assistance_requests"

    id = Column(Integer, primary_key=True, index=True)
    call_id = Column(Integer, ForeignKey("calls.id"), nullable=False, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False, index=True)
    customer_name = Column(String(255), nullable=True)
    customer_phone = Column(String(50), nullable=True)
    language_selected = Column(String(50), default="en", nullable=False)
    escalation_reason = Column(String(500), nullable=False)
    status = Column(String(50), default="waiting", nullable=False)  # waiting, contacted, resolved
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    # Aliases / Synonyms for seamless access as language and reason
    language = synonym("language_selected")
    reason = synonym("escalation_reason")

    # Relationships
    call = relationship("Call", back_populates="escalations")
    customer = relationship("Customer", back_populates="escalations")

