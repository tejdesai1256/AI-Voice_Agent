import os
import uuid
import asyncio
from sqlalchemy.orm import Session
from app.database.database import SessionLocal
from app.services.conversation_service import ConversationService
from app.services.voice_response_service import VoiceResponseService
from app.services.tts_service import TTSService
from app.services.llm_service import generate_response
from app.database.models import Customer, Call, Message

class CallManagerService:
    def __init__(self):
        self.conversation_service = ConversationService()
        self.voice_response_service = VoiceResponseService()
        self.tts_service = TTSService()
        self.max_turns = 5

    async def run_call_simulation(self, call_id: int):
        """
        Simulates an entire multi-turn voice call using the existing AI pipeline.
        This runs in the background.
        """
        try:
            with SessionLocal() as db:
                call = self.conversation_service.get_call(db, call_id)
                customer = call.customer
                cust_name = customer.name or "Customer"
                
                print(f"Starting simulated call for {cust_name} (Call ID: {call_id})")
                
                # Context array for the customer simulator prompt
                simulated_customer_history = []
                
                for turn in range(self.max_turns):
                    # 1. Generate simulated customer response text
                    if turn == 0:
                        prompt = (
                            f"You are {cust_name}, a customer initiating a phone call with a business assistant.\n"
                            "Say a brief, natural opening line like you are calling about an inquiry or issue. "
                            "Keep it under 2 sentences. Do NOT output any tags, just the spoken text."
                        )
                    else:
                        history_text = "\n".join(simulated_customer_history[-4:])
                        prompt = (
                            f"You are {cust_name}, a customer on a phone call. \n"
                            f"Here is the recent context:\n{history_text}\n"
                            "Reply naturally to the assistant's last message. Keep it under 2 sentences. "
                            "Do NOT output any tags, just the spoken text."
                        )
                        
                    cust_text = await generate_response(prompt)
                    cust_text = cust_text.strip()
                    simulated_customer_history.append(f"Customer ({cust_name}): {cust_text}")
                    
                    print(f"[Simulated Customer]: {cust_text}")
                    
                    # 2. Synthesize customer text to audio (using existing TTSService)
                    # Note: they will sound the same as the assistant, but it works for simulation.
                    os.makedirs("recordings/simulated", exist_ok=True)
                    audio_filename = f"simulated/cust_{call_id}_t{turn}.wav"
                    output_audio_path = os.path.join("recordings", audio_filename)
                    
                    # Ensure absolute path or let TTS handle it
                    generated_audio = self.tts_service.synthesize(cust_text, output_audio_path)
                    
                    # 3. Process the turn using the actual VoiceResponseService AI pipeline
                    assistant_audio_path = os.path.join("recordings", f"asst_{call_id}_t{turn}.wav")
                    
                    # Run the pipeline (transcribes, generates assistant response, saves to DB, synthesizes TTS)
                    turn_result = await self.voice_response_service.process_existing_turn(
                        call_id=call_id,
                        audio_path=generated_audio,
                        output_audio_path=assistant_audio_path,
                        db=db
                    )
                    
                    assistant_text = turn_result["response"]
                    call_status = turn_result["call_status"]
                    
                    simulated_customer_history.append(f"Assistant: {assistant_text}")
                    print(f"[Assistant]: {assistant_text}")
                    print(f"[Status]: {call_status}")
                    
                    if call_status == "RESOLVED":
                        print(f"Call {call_id} resolved by AI.")
                        break
                        
                    # Small delay between turns to prevent rate limit and simulate thinking
                    await asyncio.sleep(2)
                    
            if call_status != "RESOLVED":
                print(f"Call {call_id} reached max turns without resolution.")
                try:
                    with SessionLocal() as db:
                        self.conversation_service.end_call(db, call_id, status="RESOLVED")
                except Exception as e:
                    print(f"Failed to force end call {call_id}: {e}")

            print(f"Finished simulated call {call_id}.")
        except Exception as e:
            print(f"Error during simulated call {call_id}: {e}")
            try:
                with SessionLocal() as db:
                    self.conversation_service.end_call(db, call_id, status="FAILED")
            except:
                pass
