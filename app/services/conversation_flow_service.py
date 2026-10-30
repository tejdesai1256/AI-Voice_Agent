import re
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from app.config.conversation_questions import get_configured_questions
from app.database.models import Call, Customer, Message, Requirement, ConversationState
from app.services.escalation_service import EscalationService
from app.services.llm_service import generate_response


class ConversationFlowService:
    def __init__(self):
        pass

    @staticmethod
    def is_plausible_name(name_str: Optional[str]) -> bool:
        """
        Validates whether a given string is a plausible person's name,
        protecting database and conversation context against storing entire requirement sentences.
        """
        if not name_str or not isinstance(name_str, str):
            return False
        clean = name_str.strip()
        if not clean or len(clean) < 2 or len(clean) > 40:
            return False
        if clean.lower() in ("customer", "unknown", "n/a", "none", "caller", "user", "admin", "null"):
            return False
        # Reject sentence punctuation, newlines, digits, currency symbols
        if any(ch in clean for ch in [",", ";", ":", "!", "?", "\n", "\r", "\t", "$", "%", "@", "#", "/", "\\", "(", ")", "[", "]", "{", "}"]):
            return False
        if any(ch.isdigit() for ch in clean):
            return False
        words = clean.split()
        if not (1 <= len(words) <= 4):
            return False

        stop_words = {
            "hello", "hi", "hey", "good", "morning", "afternoon", "evening",
            "need", "want", "require", "looking", "interested", "help", "please",
            "budget", "cost", "price", "pricing", "quote", "rate", "rupees", "dollars", "inr", "usd",
            "website", "app", "application", "software", "development", "laptops", "computers", "services",
            "company", "business", "work", "calling", "call", "agent", "human", "assistant",
            "yes", "no", "okay", "sure", "thanks", "thank", "months", "weeks", "days", "years",
        }
        lowered_words = [w.lower().strip(".'\"") for w in words]
        if any(w in stop_words for w in lowered_words):
            return False
        return True

    def get_or_create_state(self, db: Session, call_id: int) -> ConversationState:
        """
        Retrieves existing ConversationState for a call or initializes a new one.
        Pre-populates known customer details (e.g. existing name).
        """
        state = db.query(ConversationState).filter(ConversationState.call_id == call_id).first()
        if state:
            return state

        call = db.query(Call).filter(Call.id == call_id).first()
        if not call:
            raise ValueError(f"Call with id {call_id} not found.")

        questions = get_configured_questions()
        initial_answers = {}

        # If customer already has a name recorded in database, note it only if plausible
        if call.customer and call.customer.name:
            c_name = call.customer.name.strip()
            if self.is_plausible_name(c_name):
                initial_answers["name"] = c_name

        # Calculate missing info
        missing = [q["id"] for q in questions if q["key"] not in initial_answers]
        current_q_id = missing[0] if missing else (questions[0]["id"] if questions else None)

        state = ConversationState(
            call_id=call_id,
            current_question_id=current_q_id,
            current_stage="ASKING_QUESTION",
            questions_asked=[current_q_id] if current_q_id else [],
            answers=initial_answers,
            missing_info=missing,
            conversation_status="AWAITING_RESPONSE",
            is_processing=0,
        )
        db.add(state)
        db.commit()
        db.refresh(state)
        return state

    def get_initial_greeting_and_question(self, db: Session, call: Call) -> str:
        """
        Returns the initial greeting combined with the first unanswered question.
        Ensures the phrase 'Hello! I am your AI voice assistant' is included for backward compatibility.
        """
        state = self.get_or_create_state(db, call.id)
        questions = get_configured_questions()
        q_map = {q["id"]: q for q in questions}

        curr_q = q_map.get(state.current_question_id) if state.current_question_id else (questions[0] if questions else None)
        question_text = curr_q["question"] if curr_q else "How may I help you today?"

        customer_name = state.answers.get("name")
        if not customer_name and call.customer and call.customer.name:
            c_name = call.customer.name.strip()
            if self.is_plausible_name(c_name):
                customer_name = c_name

        if customer_name and self.is_plausible_name(customer_name):
            return f"Hello {customer_name}! I am your AI voice assistant. {question_text}"
        return f"Hello! I am your AI voice assistant. {question_text}"

    def get_question_by_id(self, q_id: str) -> Optional[Dict[str, Any]]:
        questions = get_configured_questions()
        for q in questions:
            if q["id"] == q_id:
                return q
        return None

    def get_next_unanswered_question(self, answers: Dict[str, Any], current_q_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Returns the next question whose key has not been answered.
        """
        questions = get_configured_questions()
        for q in questions:
            key = q["key"]
            if key not in answers:
                return q
        return None

    def classify_intent(self, transcript: str, current_question: Optional[Dict[str, Any]]) -> str:
        """
        Classifies customer transcript into conversation intents.
        """
        text = transcript.strip().lower()

        # 1. Empty / Inaudible Speech
        if not text or text in (
            "[no clear speech detected]",
            "[inaudible]",
            "[inaudible speech]",
            "...",
        ) or len(text) < 2:
            return "EMPTY_SPEECH"

        # 2. Human Assistance Escalation
        should_esc, _ = EscalationService.should_escalate(transcript)
        if should_esc:
            return "HUMAN_ESCALATION"

        # 3. Customer Not Interested
        not_interested_phrases = [
            "not interested",
            "no interest",
            "don't want this",
            "do not want this",
            "stop calling",
            "not for me",
            "no thank you, not interested",
            "no thanks, not interested",
            "not really interested",
            "take me off",
            "don't call again",
            "do not call again",
            "remove my number",
        ]
        if any(p in text for p in not_interested_phrases):
            return "NOT_INTERESTED"

        # 4. Goodbye / Closing
        goodbye_phrases = [
            "goodbye",
            "bye bye",
            "bye",
            "have a good day",
            "have a nice day",
            "take care",
            "that's all, thank you",
            "that is all, thank you",
            "that's all, thanks",
            "that is all, thanks",
            "talk to you later",
        ]
        if any(p in text for p in goodbye_phrases):
            return "GOODBYE"

        # 5. Refusal to Answer
        refusal_phrases = [
            "i'd rather not say",
            "rather not say",
            "prefer not to answer",
            "prefer not to say",
            "skip this",
            "skip question",
            "skip",
            "i don't want to tell you",
            "do not want to tell you",
            "none of your business",
            "won't answer",
            "no comment",
            "pass on this",
            "let's skip",
        ]
        if any(p in text for p in refusal_phrases):
            return "REFUSAL"

        # 6. Unclear / Ambiguous
        unclear_phrases = [
            "i don't know what you mean",
            "what did you say",
            "say again",
            "huh",
            "pardon",
            "excuse me",
            "come again",
            "can't understand",
            "cannot understand",
            "didn't catch",
            "did not catch",
        ]
        if any(p in text for p in unclear_phrases):
            return "UNCLEAR"

        # 7. Customer Correction
        correction_patterns = [
            r"\b(?:actually|instead|correction|make that|change\s+to|update\s+my|change\s+the\s+budget|change\s+my\s+name|not\s+a\s+website)\b",
            r"\b(?:not\s+[\$₹]?\d+[\w\s,]*but\s+[\$₹]?\d+)\b",
            r"\b(?:instead\s+of\s+\d+[\w\s,]*make\s+it\s+\d+)\b",
        ]
        if any(re.search(pat, text) for pat in correction_patterns):
            # Guard: If currently at q1_name and user says "My name is actually X", treat as name unless correcting
            if current_question and current_question.get("key") == "name" and not any(k in text for k in ["not ", "instead", "change"]):
                pass
            else:
                return "CORRECTION"

        # 8. Customer Asking Question
        question_indicators = [
            "what kind of ai",
            "what do you do",
            "who are you",
            "what company is this",
            "why are you calling",
            "why do you need to know",
            "what are your prices",
            "how much does it cost",
            "are you a robot",
            "what is this about",
        ]
        if any(qi in text for qi in question_indicators) or (
            text.endswith("?") and not any(kw in text for kw in ["my name is", "i am", "we are"])
        ):
            return "CUSTOMER_QUESTION"

        # 9. Valid Answer
        return "ANSWER"

    def detect_customer_correction(
        self,
        transcript: str,
        answers: Dict[str, Any],
    ) -> Optional[Tuple[str, str, str]]:
        """
        Detects if customer utterance is correcting an earlier piece of information.
        Returns: (field_key, new_value, acknowledgement_message) or None.
        """
        text = transcript.strip()
        text_lower = text.lower()

        # 1. Budget correction (e.g. "Actually, my budget is $60,000, not $45,000" or "Change the budget to 60000")
        m_b = re.search(r'(?:actually|instead|change|make that|update)\s+(?:the\s+)?budget(?:\s+to|\s+is)?\s*([\$₹]?\d[\d,]*(?:\s*(?:dollars?|rupees?|inr|usd|k))?)', text_lower)
        if not m_b:
            m_b = re.search(r'(?:not\s+[\$₹]?\d+[\w\s,]*but|instead\s+of\s+[\$₹]?\d+[\w\s,]*make\s+it)\s*([\$₹]?\d[\d,]*(?:\s*(?:dollars?|rupees?|inr|usd|k))?)', text_lower)
        if m_b:
            val = m_b.group(1).strip()
            return ("budget", val, f"Got it, I've updated your budget to {val}.")

        # 2. Name correction (e.g. "Actually, my name is John, not Rahul" or "Change my name to John")
        m_n = re.search(r'(?:actually|instead|change my name to|call me)\s+(?:my name is\s+)?([A-Za-z]+(?:\s+[A-Za-z]+)?)', text, re.IGNORECASE)
        if m_n:
            val = m_n.group(1).strip()
            if self.is_plausible_name(val):
                return ("name", val, f"Understood, I have updated your name to {val}.")

        # 3. Quantity / Scope correction (e.g. "Make that 100 laptops instead of 50")
        m_q = re.search(r'(?:make that|change to|instead of\s+\d+[\w\s,]*make it|actually)\s+(\d+[\d,]*\s*(?:laptops?|desktops?|units?|items?|pcs?|licenses?))', text_lower)
        if m_q:
            val = m_q.group(1).strip()
            return ("quantity", val, f"Got it, I've updated the quantity to {val}.")

        # 4. Project / Product correction (e.g. "Actually it's a mobile app, not a website")
        if "mobile app" in text_lower:
            return ("project_type", "mobile app", "Understood, I've updated your requirement to a mobile app.")

        # 5. Timeline correction (e.g. "Actually next week instead of next month")
        m_t = re.search(r'(?:actually|instead)\s+((?:next\s+(?:week|month|year))|(?:within\s+[^,.]+)|tomorrow|asap)', text_lower)
        if m_t:
            val = m_t.group(1).strip()
            return ("expected_timeline", val, f"Understood, updated your timeline to {val}.")

        return None

    def extract_answer(self, transcript: str, question: Dict[str, Any]) -> str:
        """
        Extracts clean answer text tailored to the question key.
        """
        raw = transcript.strip()
        key = question.get("key", "")
        text_lower = raw.lower()

        if key == "name":
            match = re.search(
                r"(?:my name is|i am|i'm|this is|call me|name's)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)",
                raw,
                re.IGNORECASE,
            )
            if match:
                candidate = match.group(1).strip()
                if self.is_plausible_name(candidate):
                    return candidate
            words = raw.split()
            if 1 <= len(words) <= 3 and not any(w in text_lower for w in ["yes", "no", "hello", "hi", "okay"]):
                candidate = raw.strip(".!,")
                if self.is_plausible_name(candidate):
                    return candidate
            return None

        elif key == "company":
            match = re.search(
                r"(?:i work at|company is|represent|from|i am with|i'm with)\s+([A-Za-z0-9\s&.,'-]+)",
                raw,
                re.IGNORECASE,
            )
            if match:
                extracted = match.group(1).strip()
                return re.split(r"[.?!,]\s+", extracted)[0]
            return raw.strip(".!,")

        elif key == "industry":
            match = re.search(
                r"(?:we are in|industry is|sector is|we operate in|in the field of)\s+([A-Za-z0-9\s&.,'-]+)",
                raw,
                re.IGNORECASE,
            )
            if match:
                return re.split(r"[.?!,]\s+", match.group(1).strip())[0]
            return raw.strip(".!,")

        elif key == "role":
            match = re.search(
                r"(?:i am the|my role is|i work as|i'm a|i am a)\s+([A-Za-z0-9\s&.,'-]+)",
                raw,
                re.IGNORECASE,
            )
            if match:
                return re.split(r"[.?!,]\s+", match.group(1).strip())[0]
            return raw.strip(".!,")

        elif key == "ai_interest":
            if any(w in text_lower for w in ["yes", "definitely", "sure", "yeah", "interested", "absolutely", "we are"]):
                return "Yes, interested in AI solutions"
            elif any(w in text_lower for w in ["maybe", "exploring", "looking into"]):
                return "Exploring AI solutions"
            return raw.strip(".!,")

        elif key in ("products", "products_or_services"):
            return raw.strip(".!,")

        elif key in ("online_payments", "payments"):
            if any(w in text_lower for w in ["yes", "upi", "card", "credit", "gateway", "online"]):
                return raw.strip(".!,")
            elif any(w in text_lower for w in ["no", "not needed", "don't need"]):
                return "No online payments required"
            return raw.strip(".!,")

        elif key == "budget":
            m_b = re.search(r'(\$?\d[\d,]*(?:\s*(?:dollars?|rupees?|inr|usd|k))?)', text_lower)
            if m_b:
                return m_b.group(1).strip()
            return raw.strip(".!,")

        elif key in ("expected_timeline", "timeline", "deadline"):
            m_t = re.search(r'((?:next\s+(?:week|month|year))|(?:within\s+[^,.]+)|tomorrow|asap|\d+\s+(?:weeks?|months?|days?))', text_lower)
            if m_t:
                return m_t.group(1).strip()
            return raw.strip(".!,")

        elif key in ("project_type", "website_type"):
            return raw.strip(".!,")

        return raw.strip(".!,")

    def extract_additional_fields(self, transcript: str, current_key: str) -> Dict[str, Any]:
        """
        Detects if the customer provided answers to OTHER qualification or requirement fields
        in the same utterance (e.g. 'I need an e-commerce website for clothes with budget $10,000 next month').
        """
        additional = {}
        raw = transcript.strip()
        text_lower = raw.lower()

        # Company mention
        if current_key != "company":
            comp_match = re.search(
                r"(?:from|work at|with company)\s+([A-Za-z0-9\s&'-]+?)(?:\s+(?:in|and|for|as)|$|[.!,])",
                raw,
                re.IGNORECASE,
            )
            if comp_match:
                val = comp_match.group(1).strip()
                if len(val) > 1 and val.lower() not in ("home", "us", "india"):
                    additional["company"] = val

        # Name mention
        if current_key != "name":
            name_match = re.search(
                r"(?:my name is|i am|i'm)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)",
                raw,
                re.IGNORECASE,
            )
            if name_match:
                val = name_match.group(1).strip()
                if self.is_plausible_name(val):
                    additional["name"] = val

        # Budget mention
        if current_key != "budget":
            m_b = re.search(r'(?:budget(?:\s+is)?[:\s]*|cost[:\s]*|around[:\s]*|for[:\s]*)(\$?\d[\d,]*(?:\s*(?:dollars?|rupees?|inr|usd|lakhs?|crores?|k))?)', raw, re.IGNORECASE)
            if not m_b:
                m_b = re.search(r'(\$\d[\d,]*(?:\.\d+)?|\b\d[\d,]*\s*(?:dollars?|rupees?|inr|usd)\b)', raw, re.IGNORECASE)
            if m_b:
                val = m_b.group(1).strip()
                if any(c.isdigit() for c in val):
                    additional["budget"] = val

        # Timeline mention
        if current_key not in ("expected_timeline", "timeline", "deadline"):
            m_t = re.search(r'\b((?:next\s+(?:month|week|year))|(?:(?:in\s+)?\w+\s+(?:months?|weeks?|days?))|(?:within\s+[^,.]+)|tomorrow|asap|urgent)\b', raw, re.IGNORECASE)
            if m_t:
                additional["expected_timeline"] = m_t.group(1).strip()

        # Project type mention
        if current_key not in ("project_type", "website_type"):
            if "e-commerce" in text_lower or "ecommerce" in text_lower:
                additional["project_type"] = "An e-commerce website"
            elif any(w in text_lower for w in ["laptop", "computer", "laptops", "desktops"]):
                m_lp = re.search(r'(\d+[\d,]*\s*(?:laptops?|desktops?|computers?))', raw, re.IGNORECASE)
                additional["project_type"] = m_lp.group(1).strip() if m_lp else "Laptops"
            elif "mobile app" in text_lower:
                additional["project_type"] = "Mobile application"
            elif "voice agent" in text_lower or "ai voice" in text_lower:
                additional["project_type"] = "AI Voice Agent"

        # Products / Online payments mention
        if current_key not in ("online_payments", "payments"):
            if any(w in text_lower for w in ["upi", "credit cards", "credit card", "online payments", "online payment"]):
                additional["online_payments"] = "Yes, UPI and credit cards"

        if current_key not in ("products", "products_or_services"):
            if "clothes" in text_lower or "accessories" in text_lower:
                additional["products"] = "Clothes and accessories"

        return additional

    def get_dynamic_next_question(
        self,
        answers: Dict[str, Any],
        current_q_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Intelligently determines the next relevant question based on:
        - Domain context (e.g. e-commerce, hardware, software, AI)
        - Avoiding already-answered questions
        - Avoiding rigid sequences when customer answers make questions unnecessary
        """
        all_text = " ".join(str(v).lower() for v in answers.values())

        # Domain 1: E-Commerce
        if "e-commerce" in all_text or "ecommerce" in all_text or "online store" in all_text:
            if "products" not in answers and "products_or_services" not in answers:
                return {
                    "id": "q_products",
                    "key": "products",
                    "question": "What products will you sell?",
                    "required": True,
                }
            if "online_payments" not in answers and "payments" not in answers:
                return {
                    "id": "q_payments",
                    "key": "online_payments",
                    "question": "Do you need online payments?",
                    "required": True,
                }
            if "budget" not in answers:
                return {
                    "id": "q_budget",
                    "key": "budget",
                    "question": "What is your approximate budget?",
                    "required": True,
                }
            if "expected_timeline" not in answers and "timeline" not in answers and "deadline" not in answers:
                return {
                    "id": "q_timeline",
                    "key": "expected_timeline",
                    "question": "When are you planning to launch?",
                    "required": True,
                }
            # All essential e-commerce questions answered
            return None

        # Domain 2: Hardware / Laptops / Equipment
        if any(w in all_text for w in ["laptop", "desktop", "computer", "server", "hardware"]):
            if "required_features" not in answers and "specs" not in answers and "ram" not in all_text:
                return {
                    "id": "q_specs",
                    "key": "required_features",
                    "question": "What specifications or RAM do you require?",
                    "required": True,
                }
            if "expected_timeline" not in answers and "deadline" not in answers and "timeline" not in answers:
                return {
                    "id": "q_timeline",
                    "key": "expected_timeline",
                    "question": "When do you need these delivered?",
                    "required": True,
                }
            if "budget" not in answers:
                return {
                    "id": "q_budget",
                    "key": "budget",
                    "question": "What is your approximate budget for this purchase?",
                    "required": True,
                }
            return None

        # Domain 3: AI Voice Agent / Chatbot
        if any(w in all_text for w in ["voice agent", "ai agent", "chatbot", "calling agent", "voice ai"]):
            if "use_case" not in answers and "problem" not in answers:
                return {
                    "id": "q_use_case",
                    "key": "use_case",
                    "question": "What specific business workflows or calls will the AI handle?",
                    "required": True,
                }
            if "integrations" not in answers:
                return {
                    "id": "q_integrations",
                    "key": "integrations",
                    "question": "Do you need it integrated with your CRM or database?",
                    "required": True,
                }
            if "budget" not in answers:
                return {
                    "id": "q_budget",
                    "key": "budget",
                    "question": "What is your approximate budget for this solution?",
                    "required": True,
                }
            if "expected_timeline" not in answers and "timeline" not in answers:
                return {
                    "id": "q_timeline",
                    "key": "expected_timeline",
                    "question": "What is your target timeline for deployment?",
                    "required": True,
                }
            return None

        # Default Qualification Flow: Follow configured questions, skipping already answered slots
        questions = get_configured_questions()
        for q in questions:
            key = q["key"]
            if key not in answers:
                return q
        return None

    def calculate_missing_info(self, answers: Dict[str, Any]) -> List[str]:
        """Calculates list of uncollected important requirement keys."""
        all_text = " ".join(str(v).lower() for v in answers.values())
        missing = []
        if "name" not in answers:
            missing.append("Customer Name")
        if "e-commerce" in all_text or "ecommerce" in all_text:
            if "products" not in answers and "products_or_services" not in answers:
                missing.append("Products / Catalog")
            if "online_payments" not in answers:
                missing.append("Online Payments")
            if "budget" not in answers:
                missing.append("Approximate Budget")
            if "expected_timeline" not in answers and "timeline" not in answers:
                missing.append("Launch Timeline")
        elif any(w in all_text for w in ["laptop", "computer", "hardware"]):
            if "required_features" not in answers and "ram" not in all_text:
                missing.append("Specifications / RAM")
            if "expected_timeline" not in answers and "deadline" not in answers:
                missing.append("Delivery Timeline")
            if "budget" not in answers:
                missing.append("Budget")
        else:
            for q in get_configured_questions():
                if q["key"] not in answers:
                    missing.append(q.get("field", q["key"]))
        return missing

    def build_conversational_transition(
        self,
        previous_key: str,
        customer_answer: str,
        next_question: Dict[str, Any],
        customer_name: Optional[str] = None,
    ) -> str:
        """
        Generates a natural spoken transition acknowledging the customer's answer
        and asking the next question conversationally.
        """
        next_q_text = next_question["question"]

        if previous_key == "name":
            name_to_use = customer_name if self.is_plausible_name(customer_name) else (customer_answer if self.is_plausible_name(customer_answer) else None)
            if name_to_use:
                return f"Nice to meet you, {name_to_use}. {next_q_text}"
            return f"Thank you. {next_q_text}"
        elif previous_key in ("company", "company_name"):
            return f"Thanks! {next_q_text}"
        elif previous_key == "industry":
            return f"Understood. {next_q_text}"
        elif previous_key == "role":
            return f"Great! {next_q_text}"
        elif previous_key == "ai_interest":
            return f"Got it. {next_q_text}"
        elif previous_key in ("products", "products_or_services"):
            return f"Understood. {next_q_text}"
        elif previous_key in ("online_payments", "payments"):
            return f"Got it. {next_q_text}"
        elif previous_key == "budget":
            return f"Noted. {next_q_text}"
        elif previous_key in ("timeline", "expected_timeline", "deadline"):
            return f"Perfect. {next_q_text}"
        elif previous_key in ("project_type", "website_type"):
            return f"Understood. {next_q_text}"
        elif previous_key == "problem":
            return f"Thank you for sharing that. {next_q_text}"
        elif previous_key == "contact_time":
            return f"Perfect. {next_q_text}"
        else:
            return f"Thank you. {next_q_text}"

    async def answer_customer_question_briefly(self, transcript: str) -> str:
        """
        Answers a related/unrelated question asked by customer briefly.
        """
        text = transcript.lower()
        if "what kind of ai" in text or "what do you do" in text or "what is this" in text:
            return "We provide conversational AI voice agents that automate customer qualification, support, and calling for businesses."
        if "who are you" in text or "are you a robot" in text or "are you ai" in text:
            return "I am an AI voice assistant designed to help qualify your requirements and connect you with our team."
        if "pricing" in text or "cost" in text or "prices" in text:
            return "Our pricing is tailored based on call volume and integration requirements, which our team can provide during a consultation."
        if "why do you need" in text or "why are you asking" in text:
            return "We collect these details to ensure our solutions and follow-up consultation are tailored specifically to your business."

        try:
            prompt = (
                "You are a helpful AI voice assistant on a telephone call. "
                f"The customer asked: '{transcript}'. "
                "Answer their question in 1 polite, concise spoken sentence. Output only plain speech text."
            )
            ans = await generate_response(prompt, num_predict=35)
            clean = ans.strip().strip('"').strip("'")
            if clean and len(clean) > 5:
                return clean
        except Exception:
            pass

        return "I am an AI voice assistant helping with your inquiry."

    async def process_turn(
        self,
        call: Call,
        transcript: str,
        db: Session,
    ) -> Dict[str, Any]:
        """
        Core question flow turn processor with dynamic follow-ups and conversation memory.
        """
        state = self.get_or_create_state(db, call.id)
        current_q = self.get_question_by_id(state.current_question_id) if state.current_question_id else None

        intent = self.classify_intent(transcript, current_q)

        # -------------------------------------------------------------
        # Intent 1: Empty / Inaudible Speech
        # -------------------------------------------------------------
        if intent == "EMPTY_SPEECH":
            clarification_text = "I couldn't hear you clearly. Could you please repeat that?"
            return {
                "response": clarification_text,
                "action": "repeat",
                "advanced": False,
                "is_escalated": False,
                "call_status": "IN_PROGRESS",
                "conversation_state": {
                    "current_question_id": state.current_question_id,
                    "questions_asked": state.questions_asked or [],
                    "answers": state.answers or {},
                    "missing_info": state.missing_info or [],
                    "conversation_status": "AWAITING_RESPONSE",
                },
            }

        # -------------------------------------------------------------
        # Intent 2: Human Assistance Request
        # -------------------------------------------------------------
        if intent == "HUMAN_ESCALATION":
            _, esc_reason = EscalationService.should_escalate(transcript)
            EscalationService.create_escalation(
                db=db,
                call_id=call.id,
                reason=esc_reason or "Customer requested human assistance",
                language_selected=call.language or "en",
            )
            response_text = (
                "I completely understand. I am transferring you to a human representative right away. "
                "Please stay on the line while we connect you."
            )
            call.status = "ESCALATED"
            call.ended_at = datetime.now(timezone.utc)
            state.conversation_status = "ESCALATED"
            db.commit()

            return {
                "response": response_text,
                "action": "escalate",
                "advanced": False,
                "is_escalated": True,
                "call_status": "ESCALATED",
                "conversation_state": {
                    "current_question_id": state.current_question_id,
                    "questions_asked": state.questions_asked or [],
                    "answers": state.answers or {},
                    "missing_info": state.missing_info or [],
                    "conversation_status": "ESCALATED",
                },
            }

        # -------------------------------------------------------------
        # Intent 3: Customer Not Interested
        # -------------------------------------------------------------
        if intent == "NOT_INTERESTED":
            response_text = "I understand completely. Thank you for your time, and have a great day!"
            call.status = "RESOLVED"
            call.ended_at = datetime.now(timezone.utc)
            state.conversation_status = "COMPLETED"
            db.commit()

            return {
                "response": response_text,
                "action": "end",
                "advanced": False,
                "is_escalated": False,
                "call_status": "RESOLVED",
                "conversation_state": {
                    "current_question_id": state.current_question_id,
                    "questions_asked": state.questions_asked or [],
                    "answers": state.answers or {},
                    "missing_info": state.missing_info or [],
                    "conversation_status": "COMPLETED",
                },
            }

        # -------------------------------------------------------------
        # Intent 4: Customer Goodbye / Closing
        # -------------------------------------------------------------
        if intent == "GOODBYE":
            customer_name = state.answers.get("name") or (call.customer.name if call.customer else "")
            name_part = f", {customer_name}" if customer_name and customer_name.lower() not in ("customer", "unknown", "n/a") else ""
            response_text = f"Thank you so much for your time{name_part}. Have a wonderful day! Goodbye!"
            call.status = "RESOLVED"
            call.ended_at = datetime.now(timezone.utc)
            state.conversation_status = "COMPLETED"
            db.commit()

            return {
                "response": response_text,
                "action": "end",
                "advanced": False,
                "is_escalated": False,
                "call_status": "RESOLVED",
                "conversation_state": {
                    "current_question_id": state.current_question_id,
                    "questions_asked": state.questions_asked or [],
                    "answers": state.answers or {},
                    "missing_info": state.missing_info or [],
                    "conversation_status": "COMPLETED",
                },
            }

        # -------------------------------------------------------------
        # Intent 5: Refusal to Answer
        # -------------------------------------------------------------
        if intent == "REFUSAL":
            current_key = current_q["key"] if current_q else "unknown"
            updated_answers = dict(state.answers or {})
            updated_answers[current_key] = "[Skipped - Refused to answer]"

            missing = self.calculate_missing_info(updated_answers)
            next_q = self.get_dynamic_next_question(updated_answers, current_q.get("id") if current_q else None)

            if next_q:
                asked = list(state.questions_asked or [])
                if next_q["id"] not in asked:
                    asked.append(next_q["id"])
                state.current_question_id = next_q["id"]
                state.questions_asked = asked
                state.answers = updated_answers
                state.missing_info = missing
                state.conversation_status = "AWAITING_RESPONSE"
                db.commit()

                response_text = f"No problem at all, we can skip that. {next_q['question']}"
                return {
                    "response": response_text,
                    "action": "next_question",
                    "advanced": True,
                    "is_escalated": False,
                    "call_status": "IN_PROGRESS",
                    "conversation_state": {
                        "current_question_id": next_q["id"],
                        "questions_asked": asked,
                        "answers": updated_answers,
                        "missing_info": missing,
                        "conversation_status": "AWAITING_RESPONSE",
                    },
                }
            else:
                call.status = "RESOLVED"
                call.ended_at = datetime.now(timezone.utc)
                state.answers = updated_answers
                state.missing_info = []
                state.conversation_status = "COMPLETED"
                db.commit()

                customer_name = updated_answers.get("name")
                name_part = f", {customer_name}" if customer_name else ""
                response_text = f"Thank you so much{name_part}! We have recorded your details. Our team will be in touch with you soon. Have a great day!"
                return {
                    "response": response_text,
                    "action": "completed",
                    "advanced": True,
                    "is_escalated": False,
                    "call_status": "RESOLVED",
                    "conversation_state": {
                        "current_question_id": None,
                        "questions_asked": state.questions_asked or [],
                        "answers": updated_answers,
                        "missing_info": [],
                        "conversation_status": "COMPLETED",
                    },
                }

        # -------------------------------------------------------------
        # Intent 6: Customer Correction (Feature 2)
        # -------------------------------------------------------------
        if intent == "CORRECTION":
            corr = self.detect_customer_correction(transcript, state.answers or {})
            if corr:
                corr_key, corr_val, ack_msg = corr
                updated_answers = dict(state.answers or {})
                updated_answers[corr_key] = corr_val

                if corr_key == "name" and call.customer and self.is_plausible_name(corr_val):
                    call.customer.name = corr_val

                next_q = self.get_dynamic_next_question(updated_answers, state.current_question_id)
                missing = self.calculate_missing_info(updated_answers)

                if next_q:
                    asked = list(state.questions_asked or [])
                    if next_q["id"] not in asked:
                        asked.append(next_q["id"])
                    state.current_question_id = next_q["id"]
                    state.questions_asked = asked
                    state.answers = updated_answers
                    state.missing_info = missing
                    state.conversation_status = "AWAITING_RESPONSE"
                    db.commit()

                    response_text = f"{ack_msg} {next_q['question']}"
                    return {
                        "response": response_text,
                        "action": "correction_handled",
                        "advanced": True,
                        "is_escalated": False,
                        "call_status": "IN_PROGRESS",
                        "conversation_state": {
                            "current_question_id": next_q["id"],
                            "questions_asked": asked,
                            "answers": updated_answers,
                            "missing_info": missing,
                            "conversation_status": "AWAITING_RESPONSE",
                        },
                    }
                else:
                    call.status = "RESOLVED"
                    call.ended_at = datetime.now(timezone.utc)
                    state.current_question_id = None
                    state.answers = updated_answers
                    state.missing_info = []
                    state.conversation_status = "COMPLETED"
                    db.commit()

                    response_text = f"{ack_msg} All your requirements have been updated and recorded. Our team will follow up with you. Have a great day!"
                    return {
                        "response": response_text,
                        "action": "completed",
                        "advanced": True,
                        "is_escalated": False,
                        "call_status": "RESOLVED",
                        "conversation_state": {
                            "current_question_id": None,
                            "questions_asked": state.questions_asked or [],
                            "answers": updated_answers,
                            "missing_info": [],
                            "conversation_status": "COMPLETED",
                        },
                    }

        # -------------------------------------------------------------
        # Intent 7: Customer Asks a Question
        # -------------------------------------------------------------
        if intent == "CUSTOMER_QUESTION":
            brief_answer = await self.answer_customer_question_briefly(transcript)
            q_to_ask = current_q["question"] if current_q else "How may I help you?"
            response_text = f"{brief_answer} Returning to our conversation, {q_to_ask}"
            return {
                "response": response_text,
                "action": "answer_question",
                "advanced": False,
                "is_escalated": False,
                "call_status": "IN_PROGRESS",
                "conversation_state": {
                    "current_question_id": state.current_question_id,
                    "questions_asked": state.questions_asked or [],
                    "answers": state.answers or {},
                    "missing_info": state.missing_info or [],
                    "conversation_status": "AWAITING_RESPONSE",
                },
            }

        # -------------------------------------------------------------
        # Intent 8: Unclear Answer
        # -------------------------------------------------------------
        if intent == "UNCLEAR":
            clarification_prompt = current_q.get("clarification") if current_q else "Could you please clarify that?"
            response_text = f"I didn't quite catch that. {clarification_prompt}"
            return {
                "response": response_text,
                "action": "clarify",
                "advanced": False,
                "is_escalated": False,
                "call_status": "IN_PROGRESS",
                "conversation_state": {
                    "current_question_id": state.current_question_id,
                    "questions_asked": state.questions_asked or [],
                    "answers": state.answers or {},
                    "missing_info": state.missing_info or [],
                    "conversation_status": "AWAITING_RESPONSE",
                },
            }

        # -------------------------------------------------------------
        # Intent 9: Valid Answer (Clear, Dynamic, or Multi-field)
        # -------------------------------------------------------------
        current_key = current_q.get("key", "info") if current_q else "info"
        extracted_val = self.extract_answer(transcript, current_q) if current_q else transcript

        # If current question was name, but no plausible name could be extracted from speech
        if current_key == "name" and not extracted_val:
            response_text = "Thank you for sharing that. I have noted your requirements. Before we proceed, may I know your name please?"
            return {
                "response": response_text,
                "action": "ask_name_clarification",
                "advanced": False,
                "is_escalated": False,
                "call_status": "IN_PROGRESS",
                "conversation_state": {
                    "current_question_id": state.current_question_id,
                    "questions_asked": state.questions_asked or [],
                    "answers": state.answers or {},
                    "missing_info": state.missing_info or [],
                    "conversation_status": "AWAITING_RESPONSE",
                },
            }

        updated_answers = dict(state.answers or {})
        updated_answers[current_key] = extracted_val

        # Check if additional fields were mentioned in the same turn
        extra_fields = self.extract_additional_fields(transcript, current_key)
        for k, v in extra_fields.items():
            if k not in updated_answers:
                updated_answers[k] = v

        # If name was provided and customer object lacks a valid name, update it only if plausible
        if "name" in updated_answers and call.customer:
            name_val = updated_answers["name"]
            if name_val and self.is_plausible_name(name_val):
                if not call.customer.name or not self.is_plausible_name(call.customer.name):
                    call.customer.name = name_val

        # Determine next question dynamically based on domain and unanswered slots
        next_q = self.get_dynamic_next_question(updated_answers, current_q.get("id") if current_q else None)
        missing = self.calculate_missing_info(updated_answers)

        if next_q:
            # Advance to next question automatically
            asked = list(state.questions_asked or [])
            if next_q["id"] not in asked:
                asked.append(next_q["id"])

            state.current_question_id = next_q["id"]
            state.questions_asked = asked
            state.answers = updated_answers
            state.missing_info = missing
            state.conversation_status = "AWAITING_RESPONSE"
            db.commit()

            # Generate natural conversational transition
            customer_name = updated_answers.get("name")
            response_text = self.build_conversational_transition(
                previous_key=current_key,
                customer_answer=extracted_val,
                next_question=next_q,
                customer_name=customer_name,
            )

            return {
                "response": response_text,
                "action": "next_question",
                "advanced": True,
                "is_escalated": False,
                "call_status": "IN_PROGRESS",
                "conversation_state": {
                    "current_question_id": next_q["id"],
                    "questions_asked": asked,
                    "answers": updated_answers,
                    "missing_info": missing,
                    "conversation_status": "AWAITING_RESPONSE",
                },
            }
        else:
            # All required questions answered -> complete conversation
            call.status = "RESOLVED"
            call.ended_at = datetime.now(timezone.utc)
            state.current_question_id = None
            state.answers = updated_answers
            state.missing_info = []
            state.conversation_status = "COMPLETED"
            db.commit()

            customer_name = updated_answers.get("name")
            name_part = f", {customer_name}" if customer_name and customer_name.lower() not in ("customer", "unknown", "n/a") else ""

            proj = updated_answers.get("project_type") or updated_answers.get("products") or updated_answers.get("company")
            budget = updated_answers.get("budget")
            timeline = updated_answers.get("expected_timeline") or updated_answers.get("timeline") or updated_answers.get("deadline")

            summary_bits = []
            if proj:
                summary_bits.append(f"for {proj}")
            if budget:
                summary_bits.append(f"with budget {budget}")
            if timeline:
                summary_bits.append(f"targeted for {timeline}")
            summary_info = f" {' '.join(summary_bits)}" if summary_bits else ""

            response_text = (
                f"Thank you so much{name_part}! I have recorded all your requirements{summary_info}. "
                "Our team will review these details and be in touch soon. Have a wonderful day!"
            )

            return {
                "response": response_text,
                "action": "completed",
                "advanced": True,
                "is_escalated": False,
                "call_status": "RESOLVED",
                "conversation_state": {
                    "current_question_id": None,
                    "questions_asked": state.questions_asked or [],
                    "answers": updated_answers,
                    "missing_info": [],
                    "conversation_status": "COMPLETED",
                },
            }

