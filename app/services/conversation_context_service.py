from datetime import datetime
from typing import List
from app.database.models import Call


class ConversationContextService:
    def build_context(self, calls: List[Call]) -> str:
        """
        Formats a list of Call objects (and their associated messages) into a clean,
        chronological textual context string suitable for LLM prompting.
        
        Returns an empty string if calls is empty or contains no messages.
        """
        if not calls:
            return ""

        context_parts = []

        for index, call in enumerate(calls, start=1):
            if not call.messages:
                continue

            # Ensure messages within each call are sorted chronologically
            sorted_messages = sorted(
                call.messages,
                key=lambda m: (m.timestamp if m.timestamp is not None else datetime.min, m.id or 0),
            )

            call_lines = [f"--- Call {index} ---"]
            for msg in sorted_messages:
                speaker_label = "Customer" if msg.speaker.lower() == "customer" else "Assistant"
                call_lines.append(f"{speaker_label}: {msg.text}")

            context_parts.append("\n".join(call_lines))

        return "\n\n".join(context_parts)
