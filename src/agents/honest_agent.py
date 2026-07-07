"""
HonestAgent — implements honest_agent_spec.md.

An LLM seller agent whose honesty comes from its system prompt, not hardcoded
prices. Silence is enforced structurally (messaging_enabled), never
prompt-only. Among messaging-enabled variants, `mode` picks the persona:
"vocal" (call out coordination) or "vocal_reward" (call out + praise honest
pricing -- words only, no material inducements; see honest_agent_spec.md and
CLAUDE.md's locked decisions).
"""
from dataclasses import dataclass
from typing import Optional
from src.agents.base_agent import render_template, call_llm_and_parse
from src.agents.seller_agent import SellerAgent

HONEST_VOCAL_PROMPT = "honest_vocal.txt"
HONEST_SILENT_PROMPT = "honest_silent.txt"
HONEST_REWARD_PROMPT = "honest_vocal_reward.txt"


@dataclass
class HonestAgent(SellerAgent):
    temperature: float = 0.0
    messaging_enabled: bool = True
    mode: str = "vocal"  # "vocal" | "silent" | "vocal_reward"
    agent_type: str = "honest"

    def _persona_text(self) -> str:
        # Silence is enforced structurally (messaging_enabled), so it wins over mode.
        if not self.messaging_enabled:
            template = HONEST_SILENT_PROMPT
        elif self.mode == "vocal_reward":
            template = HONEST_REWARD_PROMPT
        else:
            template = HONEST_VOCAL_PROMPT
        return render_template(template, seller_id=self.seller_id, valuation=self.valuation)

    def step(
        self,
        round_number: int,
        bid_queue: str,
        ask_queue: str,
        past_bids_and_asks: str,
        past_trades: str,
        agent_successful_trades: str,
        seller_messages: dict[str, str],
    ) -> dict:
        memory_text = "\n".join(self.memory) if self.memory else "No memories yet."
        prompt = render_template(
            "honest_seller_prompt.j2",
            seller_id=self.seller_id,
            persona=self._persona_text(),
            valuation=self.valuation,
            num_rounds=self.num_rounds,
            round_number=round_number,
            bid_queue=bid_queue,
            ask_queue=ask_queue,
            past_bids_and_asks=past_bids_and_asks,
            past_trades=past_trades,
            agent_successful_trades=agent_successful_trades,
            memory=memory_text,
            scratch_pad=self.scratch_pad,
            seller_comms_enabled=self.seller_comms_enabled,
            messaging_enabled=self.messaging_enabled,
            seller_messages=seller_messages,
        )

        response = call_llm_and_parse(self.model, prompt, temperature=self.temperature)
        self.last_response = response

        if response.get("new_memory"):
            self.memory.append(f"Hour {round_number}: {response['new_memory']}")
            if len(self.memory) > 10:
                self.memory = self.memory[-10:]

        if response.get("scratch_pad_update"):
            self.scratch_pad = response["scratch_pad_update"]

        return response

    def get_message(self) -> Optional[str]:
        # Mechanical enforcement (belt-and-suspenders): a silent honest agent
        # never emits a message, regardless of what the model returned. The
        # prompt template also omits the message field entirely in silent
        # mode, so nothing is even requested of the model.
        if not self.messaging_enabled:
            return None
        return super().get_message()
