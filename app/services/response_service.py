from app.services.llm_service import generate_response


class ResponseService:
    def __init__(self):
        """Initializes the conversational response service."""
        pass

    async def generate_response(
        self, transcript: str, conversation_context: str = ""
    ) -> str:
        """
        Generates a warm, professional, concise spoken response to the customer's input,
        leveraging conversation history and the Ollama LLM service.
        """
        history_block = (
            f"Conversation History:\n{conversation_context.strip()}\n\n"
            if conversation_context and conversation_context.strip()
            else ""
        )

        prompt = (
            "You are a professional AI voice assistant handling incoming customer inquiries.\n"
            "Generate a warm, professional, natural spoken response to the customer's latest statement.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Respond directly to the current customer statement. Use prior conversation history for context without repeating everything.\n"
            "2. NEVER invent, assume, or hallucinate prices, budgets, timelines, product details, or business commitments.\n"
            "3. If key information (such as budget, timeline, or scope) remains missing after checking both history and the current statement, ask ONE clear, relevant follow-up question.\n"
            "4. Write in a clear, natural conversational style optimized to sound great when spoken aloud by a TTS system.\n"
            "5. Do NOT include markdown code blocks, formatting tags, emojis, or JSON. Output plain conversational text only.\n"
            "6. Keep your response concise (1 to 3 sentences maximum), ideal for voice calls.\n\n"
            f"{history_block}"
            f"Current Customer Statement:\n\"{transcript.strip()}\"\n\n"
            "Assistant Response:"
        )

        reply = await generate_response(prompt, num_predict=75)
        cleaned = reply.strip().strip('"').strip("'")
        return cleaned

    async def check_resolution(
        self, transcript: str, conversation_context: str = ""
    ) -> bool:
        """
        Determines whether the customer inquiry is concluded or resolved.
        Evaluates closing intents and phrases to identify call completion.
        """
        text = transcript.lower().strip()
        closing_phrases = [
            "thank you",
            "thanks",
            "that's all",
            "that is all",
            "that's it",
            "that is it",
            "goodbye",
            "bye",
            "have a good day",
            "have a nice day",
            "all set",
            "nothing else",
            "no that's all",
            "no that is all",
            "no thanks",
            "no, thanks",
            "take care",
            "done, thank you",
            "done thanks",
        ]
        return any(phrase in text for phrase in closing_phrases)
