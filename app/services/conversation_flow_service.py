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

        # 7. Customer Asking Question
        # Checks if customer asks a question back without providing an answer
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

        # 8. Valid Answer
        return "ANSWER"


    def extract_answer(self, transcript: str, question: Dict[str, Any]) -> str:
        """
        Extracts clean answer text tailored to the question key.
        """
        raw = transcript.strip()
        key = question.get("key", "")
        text_lower = raw.lower()

        if key == "name":
            # Patterns like "My name is Rahul", "I am Rahul", "I'm Rahul Sharma", "This is Priya"
            match = re.search(
                r"(?:my name is|i am|i'm|this is|call me|name's)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)",
                raw,
                re.IGNORECASE,
            )
            if match:
                candidate = match.group(1).strip()
                if self.is_plausible_name(candidate):
                    return candidate
            # If short string (1-3 words) without stop phrases, take as name
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
                # strip trailing sentences
                extracted = re.split(r"[.?!,]\s+", extracted)[0]
                return extracted
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

        return raw.strip(".!,")

    def extract_additional_fields(self, transcript: str, current_key: str) -> Dict[str, str]:
        """
        Detects if the customer provided answers to OTHER qualification questions
        in the same utterance (e.g. 'I am Rahul from ABC Tech in IT').
        """
        additional = {}
        raw = transcript.strip()

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

        return additional

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
        elif previous_key == "company":
            return f"Thanks! {next_q_text}"
        elif previous_key == "industry":
            return f"Understood. {next_q_text}"
        elif previous_key == "role":
            return f"Great! {next_q_text}"
        elif previous_key == "ai_interest":
            return f"Got it. {next_q_text}"
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

        # Fallback to LLM for open questions if available
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
        Core question flow turn processor.
        Understands customer answer, updates state, determines next question,
        and generates natural conversational spoken response.
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

            # Update missing info
            all_questions = get_configured_questions()
            missing = [q["id"] for q in all_questions if q["key"] not in updated_answers]

            next_q = self.get_next_unanswered_question(updated_answers, current_q.get("id") if current_q else None)

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
                # All questions answered or skipped
                call.status = "RESOLVED"
                call.ended_at = datetime.now(timezone.utc)
                state.answers = updated_answers
                state.missing_info = []
                state.conversation_status = "COMPLETED"
                db.commit()

                customer_name = updated_answers.get("name")
                name_part = f", {customer_name}" if customer_name else ""
                response_text = f"Thank you so much{name_part}! We have recorded all your details. Our team will be in touch with you soon. Have a great day!"
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
        # Intent 6: Customer Asks a Question
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
        # Intent 7: Unclear Answer
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
        # Intent 8: Valid Answer (Clear or Multi-field)
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

        # Recalculate missing questions
        all_questions = get_configured_questions()
        missing = [q["id"] for q in all_questions if q["key"] not in updated_answers]

        # Determine next unanswered question
        next_q = self.get_next_unanswered_question(updated_answers, current_q.get("id") if current_q else None)

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
            # All questions answered -> complete conversation
            call.status = "RESOLVED"
            call.ended_at = datetime.now(timezone.utc)
            state.current_question_id = None
            state.answers = updated_answers
            state.missing_info = []
            state.conversation_status = "COMPLETED"
            db.commit()

            customer_name = updated_answers.get("name", "there")
            company_name = updated_answers.get("company")
            contact_time = updated_answers.get("contact_time")

            summary_parts = []
            if company_name:
                summary_parts.append(f"for {company_name}")
            if contact_time:
                summary_parts.append(f"and our team will follow up with you {contact_time}")

            extra_str = f" {' '.join(summary_parts)}" if summary_parts else ""
            response_text = (
                f"Thank you so much, {customer_name}! I have recorded all your information{extra_str}. "
                "We look forward to speaking with you. Have a wonderful day!"
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
