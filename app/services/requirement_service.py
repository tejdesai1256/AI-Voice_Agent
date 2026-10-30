import json
import re
from typing import Optional, Dict, Any, List
from app.services.llm_service import generate_response


class RequirementService:
    def __init__(self):
        """
        Initializes the RequirementService for structured customer requirement extraction
        and automatic conversation summaries.
        """
        pass

    async def extract_requirements(
        self,
        transcript: str,
        existing_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Sends the call transcript to Gemma via llm_service to extract structured customer requirements.
        Extracts:
        - customer_name
        - company_name
        - business_type
        - project_type
        - required_features
        - budget
        - expected_timeline
        - preferred_language
        - customer_concerns
        - next_action
        - follow_up_requirements

        Preserves backwards compatibility with legacy keys:
        - requirements
        - products_or_services
        - quantity
        - deadline
        - important_details
        """
        prompt = (
            "You are a customer requirement extraction assistant. Analyze the following transcript of a customer call and extract structured requirement details.\n\n"
            "Return ONLY a valid JSON object with the following exact keys:\n"
            '- "customer_name": string or null (person\'s name if stated)\n'
            '- "company_name": string or null (company or organization name if stated)\n'
            '- "business_type": string or null (industry or business type, e.g. E-commerce, Retail, IT)\n'
            '- "project_type": string or null (specific project or service requested, e.g. E-commerce website, Laptops, AI voice agent)\n'
            '- "required_features": list of strings (features or specifications mentioned, e.g. ["16GB RAM", "Online payments"])\n'
            '- "budget": string or null (budget mentioned, e.g. "$45,000", "50000 rupees")\n'
            '- "expected_timeline": string or null (completion or delivery timeframe, e.g. "next month", "2 weeks")\n'
            '- "preferred_language": string or null (e.g. "en", "hi", "mr")\n'
            '- "customer_concerns": list of strings (any challenges, hesitations, or questions raised by customer)\n'
            '- "next_action": string or null (agreed next step, e.g. "Follow-up consultation", "Send proposal")\n'
            '- "follow_up_requirements": string or null (any specific follow-up details needed)\n\n'
            "CRITICAL INSTRUCTIONS:\n"
            "1. If a field is not present or unknown in the transcript, set its value to null (or empty list [] for list fields).\n"
            "2. Never invent, assume, or hallucinate any information not explicitly stated in the transcript.\n"
            "3. If the customer corrected a previous statement (e.g. 'actually my budget is 60k'), use the corrected value.\n"
            "4. Output strictly raw JSON without markdown wrapping and without explanations.\n\n"
            f"Transcript:\n{transcript}\n"
        )

        raw_response = await generate_response(prompt, num_predict=350)
        parsed = self._parse_json_response(raw_response)

        # Resilient heuristic fallback if LLM returned nulls/empty or offline
        has_val = any(
            parsed.get(k) is not None
            for k in ["project_type", "budget", "expected_timeline", "customer_name", "company_name"]
        )
        if not has_val and transcript and transcript.strip():
            heuristics = self._heuristic_extract(transcript)
            for k, v in heuristics.items():
                if v and not parsed.get(k):
                    parsed[k] = v

        # Apply customer correction detection over prior existing data
        if existing_data:
            parsed = self.apply_corrections(transcript, existing_data, parsed)

        # Backwards compatibility legacy fields
        self._ensure_legacy_compatibility(parsed)

        return parsed

    def apply_corrections(
        self,
        transcript: str,
        existing: Dict[str, Any],
        current: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Detects explicit customer corrections (e.g. 'actually my budget is $60,000, not $45,000'
        or 'change name to John') and guarantees the corrected value takes precedence.
        """
        merged = dict(existing)
        merged.update({k: v for k, v in current.items() if v is not None})

        text = transcript.lower()

        # Budget correction
        m_b_corr = re.search(r'(?:actually|instead|correction|change|make that|update)\s+(?:the\s+)?budget(?:\s+to|\s+is)?\s*([\$₹]?\d[\d,]*(?:\s*(?:dollars?|rupees?|inr|usd|k))?)', text)
        if m_b_corr:
            merged["budget"] = m_b_corr.group(1).strip()

        # Name correction
        m_n_corr = re.search(r'(?:actually|instead|correction|my name is actually|call me)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)', transcript, re.IGNORECASE)
        if m_n_corr:
            candidate = m_n_corr.group(1).strip()
            if candidate.lower() not in ("actually", "instead", "yes", "no"):
                merged["customer_name"] = candidate

        # Quantity / scope correction
        m_q_corr = re.search(r'(?:make that|change to|instead of\s+\d+[\w\s,]*make it|actually)\s+(\d+[\d,]*\s*(?:laptops?|desktops?|units?|items?|pcs?|licenses?))', text)
        if m_q_corr:
            merged["quantity"] = m_q_corr.group(1).strip()

        return merged

    def _heuristic_extract(self, transcript: str) -> Dict[str, Any]:
        """
        Resilient heuristic extraction from raw transcript text when LLM extraction
        fails or local Ollama is offline. Never invents customer information.
        """
        text = transcript.strip()
        result = {
            "customer_name": None,
            "company_name": None,
            "business_type": None,
            "project_type": None,
            "required_features": [],
            "budget": None,
            "expected_timeline": None,
            "preferred_language": "en",
            "customer_concerns": [],
            "next_action": None,
            "follow_up_requirements": None,
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

        # 2. Project / Products / Services detection
        m_prod = re.search(
            r'\b((?:[a-zA-Z0-9\-]+\s+)?(?:laptops?|desktops?|computers?|servers?|e-commerce(?:\s+website)?|website|mobile\s+app|application|software|hardware|crm|erp)(?:\s+with\s+[^,.]+)?)\b',
            text,
            re.IGNORECASE,
        )
        if m_prod:
            result["project_type"] = m_prod.group(1).strip()

        # 3. Features extraction
        features = []
        if re.search(r'\b(?:16gb\s+ram|32gb\s+ram|ssd|i7|i9|m3|m4)\b', text, re.IGNORECASE):
            spec_m = re.findall(r'\b(?:16gb\s+ram|32gb\s+ram|\d+gb\s+ssd|\d+tb\s+ssd|i7|i9)\b', text, re.IGNORECASE)
            features.extend([s.strip() for s in spec_m])
        if re.search(r'\b(?:online\s+payments?|upi|credit\s+cards?|payment\s+gateway)\b', text, re.IGNORECASE):
            features.append("Online payments (UPI, Cards)")
        if re.search(r'\b(?:clothes|clothing|accessories|apparel|products)\b', text, re.IGNORECASE):
            features.append("Catalog: Clothes & accessories")
        result["required_features"] = list(dict.fromkeys(features))

        # 4. Expected timeline detection
        m_dl = re.search(
            r'\b((?:next\s+(?:month|week|year))|(?:(?:in\s+)?\w+\s+(?:months?|weeks?|days?))|(?:within\s+[^,.]+)|tomorrow|asap|urgent)\b',
            text,
            re.IGNORECASE,
        )
        if m_dl:
            result["expected_timeline"] = m_dl.group(1).strip()

        # 5. Customer Name extraction
        m_name = re.search(r'(?:my name is|i am|i\'m|this is)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)', text, re.IGNORECASE)
        if m_name:
            cand = m_name.group(1).strip()
            if len(cand.split()) <= 3 and cand.lower() not in ("customer", "interested", "looking", "calling"):
                result["customer_name"] = cand

        # 6. Company Name extraction
        m_comp = re.search(r'(?:from|work at|company is|represent)\s+([A-Za-z0-9\s&.,\'-]+?)(?:\s+(?:in|and|for|as)|$|[.!,])', text, re.IGNORECASE)
        if m_comp:
            c_cand = m_comp.group(1).strip()
            if len(c_cand) > 1 and c_cand.lower() not in ("home", "us", "india"):
                result["company_name"] = c_cand

        # 7. Business Type
        if "e-commerce" in text.lower():
            result["business_type"] = "E-Commerce / Online Retail"
        elif any(w in text.lower() for w in ["it", "tech", "software"]):
            result["business_type"] = "Information Technology"
        elif "healthcare" in text.lower():
            result["business_type"] = "Healthcare"

        # 8. Customer Concerns
        concerns = []
        if any(w in text.lower() for w in ["pricing", "expensive", "too high", "costly", "high price"]):
            concerns.append("Pricing & budget sensitivity")
        if any(w in text.lower() for w in ["timeline", "delay", "urgent", "asap"]):
            concerns.append("Timeline and delivery constraints")
        result["customer_concerns"] = concerns

        # 9. Next Action
        if result["project_type"] or result["budget"]:
            result["next_action"] = "Prepare tailored technical proposal and schedule follow-up consultation"

        return result

    def _ensure_legacy_compatibility(self, data: Dict[str, Any]):
        """Preserves legacy keys used across existing Phase 1 tests and callers."""
        pt = data.get("project_type")
        budget = data.get("budget")
        timeline = data.get("expected_timeline")
        feats = data.get("required_features") or []

        if not data.get("requirements"):
            parts = []
            if pt:
                parts.append(f"Customer requested {pt}")
            if feats:
                parts.append(f"with {', '.join(feats)}")
            if budget:
                parts.append(f"budget {budget}")
            if timeline:
                parts.append(f"deadline {timeline}")
            data["requirements"] = " ".join(parts) if parts else (pt or "General consultation requested")

        data["products_or_services"] = pt
        data["deadline"] = timeline
        data["important_details"] = feats if feats else (data.get("important_details") or [])

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Cleans and parses the LLM raw output into a Python dictionary."""
        cleaned = text.strip()
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

        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    return self._normalize_req_dict(parsed)
            except json.JSONDecodeError:
                pass

        return self._regex_extract_fields(cleaned)

    def _normalize_req_dict(self, data: dict) -> dict:
        """Ensures all standard keys are present with appropriate types."""
        def clean_val(v):
            if v is None:
                return None
            s = str(v).strip()
            return None if s.lower() in ["null", "none", "n/a", "not mentioned", "not specified", "unknown"] else s

        def clean_list(v):
            if isinstance(v, list):
                return [clean_val(item) for item in v if clean_val(item)]
            if isinstance(v, str) and v.strip():
                return [v.strip()]
            return []

        return {
            "customer_name": clean_val(data.get("customer_name")),
            "company_name": clean_val(data.get("company_name")),
            "business_type": clean_val(data.get("business_type")),
            "project_type": clean_val(data.get("project_type")),
            "required_features": clean_list(data.get("required_features")),
            "budget": clean_val(data.get("budget")),
            "expected_timeline": clean_val(data.get("expected_timeline")),
            "preferred_language": clean_val(data.get("preferred_language")) or "en",
            "customer_concerns": clean_list(data.get("customer_concerns")),
            "next_action": clean_val(data.get("next_action")),
            "follow_up_requirements": clean_val(data.get("follow_up_requirements")),
            "requirements": clean_val(data.get("requirements")),
            "products_or_services": clean_val(data.get("products_or_services")),
            "quantity": clean_val(data.get("quantity")),
            "deadline": clean_val(data.get("deadline")),
            "important_details": clean_list(data.get("important_details")),
        }

    def _regex_extract_fields(self, text: str) -> dict:
        """Regex fallback to extract fields when JSON formatting is broken."""
        def find_field(key: str):
            pattern = rf'"{key}"\s*:\s*"([^"]+)"'
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                return None if val.lower() in ["null", "none", "n/a", "unknown"] else val
            return None

        return {
            "customer_name": find_field("customer_name"),
            "company_name": find_field("company_name"),
            "business_type": find_field("business_type"),
            "project_type": find_field("project_type") or find_field("products_or_services"),
            "required_features": [],
            "budget": find_field("budget"),
            "expected_timeline": find_field("expected_timeline") or find_field("deadline"),
            "preferred_language": "en",
            "customer_concerns": [],
            "next_action": find_field("next_action"),
            "follow_up_requirements": find_field("follow_up_requirements"),
            "requirements": find_field("requirements"),
            "products_or_services": find_field("products_or_services"),
            "quantity": find_field("quantity"),
            "deadline": find_field("deadline"),
            "important_details": [],
        }

    async def generate_summary(
        self,
        call_id: int,
        transcript: str,
        structured_req: Dict[str, Any],
        unanswered_questions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generates a concise, structured conversation summary containing:
        - Customer's main requirement
        - Important details collected
        - Budget and timeline, if provided
        - Unanswered questions
        - Customer concerns
        - Agreed next steps
        - Whether follow-up is required
        - Textual summary narrative
        """
        reqs = structured_req or {}
        main_req = reqs.get("project_type") or reqs.get("requirements") or "Inquiry & general consultation"
        budget = reqs.get("budget")
        timeline = reqs.get("expected_timeline") or reqs.get("deadline")
        details = reqs.get("required_features") or reqs.get("important_details") or []
        concerns = reqs.get("customer_concerns") or []
        unanswered = unanswered_questions or []
        next_action = reqs.get("next_action") or "Sales engineering team to prepare proposal and follow up"
        needs_follow_up = bool(main_req and main_req != "Inquiry & general consultation")

        # Synthesize concise readable paragraph
        narrative_parts = [f"Customer inquired regarding {main_req}."]
        if details:
            narrative_parts.append(f"Key specifications: {', '.join(details)}.")
        if budget:
            narrative_parts.append(f"Budget indicated: {budget}.")
        if timeline:
            narrative_parts.append(f"Target completion timeframe: {timeline}.")
        if concerns:
            narrative_parts.append(f"Noted customer concerns: {', '.join(concerns)}.")
        narrative_parts.append(f"Agreed next action: {next_action}.")

        concise_text = " ".join(narrative_parts)

        return {
            "call_id": call_id,
            "main_requirement": main_req,
            "important_details": details,
            "budget": budget,
            "timeline": timeline,
            "unanswered_questions": unanswered,
            "customer_concerns": concerns,
            "agreed_next_steps": next_action,
            "follow_up_required": needs_follow_up,
            "concise_summary": concise_text,
        }


