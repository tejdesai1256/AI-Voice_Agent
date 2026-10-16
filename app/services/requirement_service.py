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
        return self._parse_json_response(raw_response)

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

