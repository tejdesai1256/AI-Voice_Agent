import json
import re
from app.services.llm_service import generate_response


class RequirementService:
    def __init__(self):
        """
        Initializes the RequirementService.
        """
        pass

    async def extract_requirements(self, transcript: str) -> dict:
        """
        Sends the call transcript to Gemma 3 4B via llm_service to extract structured customer requirements.
        
        Returns structured JSON (Python dict) containing:
        - requirements
        - products_or_services
        - quantity
        - budget
        - deadline
        - important_details
        """
        prompt = (
            "You are a customer requirement extraction assistant. Analyze the following transcript of a customer call and extract structured requirement details.\n\n"
            "Return ONLY a valid JSON object with the following exact keys:\n"
            '- "requirements": string or null (summary of customer requirements)\n'
            '- "products_or_services": string or null (specific products or services requested)\n'
            '- "quantity": string or null (quantity or scope mentioned)\n'
            '- "budget": string or null (budget mentioned)\n'
            '- "deadline": string or null (timeframe or completion deadline mentioned)\n'
            '- "important_details": list of strings (any other specific key details or constraints mentioned)\n\n'
            "CRITICAL INSTRUCTIONS:\n"
            "1. If a field is not present in the transcript, set its value to null (or empty list [] for important_details).\n"
            "2. Do NOT invent, assume, or hallucinate any information not explicitly stated in the transcript.\n"
            "3. Output strictly raw JSON without markdown wrapping and without explanations.\n\n"
            f"Transcript:\n{transcript}\n"
        )

        raw_response = await generate_response(prompt, num_predict=250)
        parsed = self._parse_json_response(raw_response)

        # If LLM returned empty or null fields (e.g. when Ollama is offline or produces empty JSON),
        # perform resilient heuristic regex extraction on the transcript so customer requirements are never lost.
        has_val = any(parsed.get(k) is not None for k in ["requirements", "products_or_services", "quantity", "budget", "deadline"])
        if not has_val and transcript and transcript.strip():
            heuristics = self._heuristic_extract(transcript)
            for k, v in heuristics.items():
                if v and not parsed.get(k):
                    parsed[k] = v

        return parsed

    def _heuristic_extract(self, transcript: str) -> dict:
        """
        Resilient heuristic extraction from raw transcript text when LLM extraction
        fails or local Ollama is offline.
        """
        text = transcript.strip()
        result = {
            "requirements": None,
            "products_or_services": None,
            "quantity": None,
            "budget": None,
            "deadline": None,
            "important_details": [],
        }
        if not text:
            return result

        # 1. Budget detection ($45,000, 50000 rupees, budget is 45000 dollars, etc.)
        m_budget = re.search(
            r'(?:budget(?:\s+is)?[:\s]*|cost[:\s]*|around[:\s]*|for[:\s]*)'
            r'(\$?\d[\d,]*(?:\s*(?:dollars?|rupees?|inr|usd|lakhs?|crores?|k))?)',
            text,
            re.IGNORECASE,
        )
        if not m_budget:
            m_budget = re.search(
                r'(\$\d[\d,]*(?:\.\d+)?|\b\d[\d,]*\s*(?:dollars?|rupees?|inr|usd)\b)',
                text,
                re.IGNORECASE,
            )
        if m_budget:
            b_val = m_budget.group(1).strip()
            if b_val and any(c.isdigit() for c in b_val):
                result["budget"] = b_val

        # 2. Quantity detection (50 laptops, 10 units, 5 computers, etc.)
        m_qty = re.search(
            r'\b(\d+[\d,]*\s*(?:laptops?|desktops?|computers?|units?|items?|pcs?|pieces?|licenses?|seats?|users?|servers?))\b',
            text,
            re.IGNORECASE,
        )
        if not m_qty:
            m_qty = re.search(r'\b(\d+[\d,]*)\s+(?:of\s+)?([a-zA-Z]+)', text, re.IGNORECASE)
            if m_qty and m_qty.group(2).lower() not in ["dollars", "rupees", "inr", "usd", "months", "weeks", "days", "years", "hours"]:
                result["quantity"] = f"{m_qty.group(1)} {m_qty.group(2)}"
        else:
            result["quantity"] = m_qty.group(1).strip()

        # 3. Products / Services detection
        m_prod = re.search(
            r'\b((?:[a-zA-Z0-9\-]+\s+)?(?:laptops?|desktops?|computers?|servers?|e-commerce(?:\s+website)?|website|mobile\s+app|application|software|hardware|crm|erp)(?:\s+with\s+[^,.]+)?)\b',
            text,
            re.IGNORECASE,
        )
        if m_prod:
            result["products_or_services"] = m_prod.group(1).strip()

        # 4. Deadline detection (next month, two months, within 3 weeks, tomorrow, asap)
        m_dl = re.search(
            r'\b((?:next\s+(?:month|week|year))|(?:(?:in\s+)?\w+\s+(?:months?|weeks?|days?))|(?:within\s+[^,.]+)|tomorrow|asap|urgent)\b',
            text,
            re.IGNORECASE,
        )
        if m_dl:
            result["deadline"] = m_dl.group(1).strip()

        # 5. Summary
        prod = result.get("products_or_services")
        qty = result.get("quantity")
        if prod and qty:
            result["requirements"] = f"Customer requested {qty} of {prod}"
        elif prod:
            result["requirements"] = f"Customer requested {prod}"
        elif len(text) > 5:
            first_sent = text.split(".")[0].strip()
            result["requirements"] = first_sent[:150]

        return result

    def _parse_json_response(self, text: str) -> dict:
        """
        Cleans and parses the LLM raw output into a Python dictionary.
        Handles markdown codeblocks and fallback regex extraction.
        """
        cleaned = text.strip()

        # Remove markdown code block wrapping if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return self._normalize_req_dict(parsed)
        except json.JSONDecodeError:
            pass

        # Fallback 1: regex search for JSON object inside response
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    return self._normalize_req_dict(parsed)
            except json.JSONDecodeError:
                pass

        # Fallback 2: regex field-by-field extraction
        return self._regex_extract_fields(cleaned)

    def _normalize_req_dict(self, data: dict) -> dict:
        """Ensures all standard keys are present with appropriate types."""
        def clean_val(v):
            if v is None:
                return None
            s = str(v).strip()
            return None if s.lower() in ["null", "none", "n/a", "not mentioned", "not specified"] else s

        return {
            "requirements": clean_val(data.get("requirements")),
            "products_or_services": clean_val(data.get("products_or_services")),
            "quantity": clean_val(data.get("quantity")),
            "budget": clean_val(data.get("budget")),
            "deadline": clean_val(data.get("deadline")),
            "important_details": data.get("important_details", []) if isinstance(data.get("important_details"), list) else [],
        }

    def _regex_extract_fields(self, text: str) -> dict:
        """Regex fallback to extract fields when JSON formatting is broken."""
        def find_field(key: str):
            pattern = rf'"{key}"\s*:\s*"([^"]+)"'
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                return None if val.lower() in ["null", "none", "n/a"] else val
            return None

        return {
            "requirements": find_field("requirements"),
            "products_or_services": find_field("products_or_services"),
            "quantity": find_field("quantity"),
            "budget": find_field("budget"),
            "deadline": find_field("deadline"),
            "important_details": [],
            "raw_response": text[:300],
        }

