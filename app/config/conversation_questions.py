import os
import json
from typing import List, Dict, Any

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "questions.json",
)

DEFAULT_QUALIFICATION_QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": "q1_name",
        "key": "name",
        "question": "May I know your name?",
        "field": "customer_name",
        "required": True,
        "clarification": "Could you please tell me your name so I know who I am speaking with?",
        "order": 1,
    },
    {
        "id": "q2_company",
        "key": "company",
        "question": "Which company do you represent?",
        "field": "company_name",
        "required": True,
        "clarification": "Could you please mention the name of your company or organization?",
        "order": 2,
    },
    {
        "id": "q3_industry",
        "key": "industry",
        "question": "What industry does your company operate in?",
        "field": "industry",
        "required": True,
        "clarification": "Could you tell me what sector or industry your company operates in, such as IT, healthcare, retail, or manufacturing?",
        "order": 3,
    },
    {
        "id": "q4_role",
        "key": "role",
        "question": "What is your role in the company?",
        "field": "role",
        "required": True,
        "clarification": "Could you please share your designation or role at your company?",
        "order": 4,
    },
    {
        "id": "q5_ai_interest",
        "key": "ai_interest",
        "question": "Are you interested in our AI solutions?",
        "field": "interested_in_ai",
        "required": True,
        "clarification": "Could you let me know if your team is interested in exploring AI-powered solutions for your business?",
        "order": 5,
    },
    {
        "id": "q6_problem",
        "key": "problem",
        "question": "What problem would you like to solve?",
        "field": "problem_to_solve",
        "required": True,
        "clarification": "What specific challenge or problem would you like to address with AI?",
        "order": 6,
    },
    {
        "id": "q7_contact_time",
        "key": "contact_time",
        "question": "When would you prefer our team to contact you?",
        "field": "preferred_contact_time",
        "required": True,
        "clarification": "When would be the best day or time for our team to follow up with you?",
        "order": 7,
    },
]

_memory_questions_override = None


def get_configured_questions() -> List[Dict[str, Any]]:
    """
    Returns the active list of qualification questions, sorted by order.
    Checks memory override, then file-based configuration, then defaults.
    """
    global _memory_questions_override
    if _memory_questions_override is not None:
        return sorted(_memory_questions_override, key=lambda x: x.get("order", 0))

    config_path = os.getenv("QUESTIONS_CONFIG_PATH", DEFAULT_CONFIG_PATH)
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                questions = json.load(f)
                if isinstance(questions, list) and len(questions) > 0:
                    return sorted(questions, key=lambda x: x.get("order", 0))
        except Exception as e:
            print(f"[QuestionConfig Warning] Failed loading {config_path}: {e}")

    return sorted(DEFAULT_QUALIFICATION_QUESTIONS, key=lambda x: x.get("order", 0))


def set_configured_questions(questions: List[Dict[str, Any]], persist: bool = False):
    """
    Sets active qualification questions at runtime with optional file persistence.
    """
    global _memory_questions_override
    _memory_questions_override = questions

    if persist:
        config_path = os.getenv("QUESTIONS_CONFIG_PATH", DEFAULT_CONFIG_PATH)
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(questions, f, indent=2)
        except Exception as e:
            print(f"[QuestionConfig Warning] Failed saving {config_path}: {e}")


def reset_configured_questions():
    """Resets to default questions and cleans up any persistent override file."""
    global _memory_questions_override
    _memory_questions_override = None
    config_path = os.getenv("QUESTIONS_CONFIG_PATH", DEFAULT_CONFIG_PATH)
    if os.path.exists(config_path):
        try:
            os.remove(config_path)
        except OSError:
            pass

