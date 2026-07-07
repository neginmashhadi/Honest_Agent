from dataclasses import dataclass, field
from typing import Optional
from src.agents.base_agent import render_template, call_llm_and_parse


@dataclass
class SellerAgent:
    seller_id: str
    company: str
    model: str
    valuation: float
    num_rounds: int
    seller_comms_enabled: bool = True
    oversight_enabled: bool = False
    boss_pressure: bool = False

    memory: list[str] = field(default_factory=list)
    scratch_pad: str = ""
    is_gagged: bool = False
    last_response: Optional[dict] = None

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
            "seller_prompt.j2",
            seller_id=self.seller_id,
            company=self.company,
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
            oversight_enabled=self.oversight_enabled,
            boss_pressure=self.boss_pressure,
            is_gagged=self.is_gagged,
            seller_messages=seller_messages,
        )

        response = call_llm_and_parse(self.model, prompt)
        self.last_response = response

        # Update persistent state
        if response.get("new_memory"):
            self.memory.append(f"Hour {round_number}: {response['new_memory']}")
            # Keep only last 10 memories to avoid context bloat
            if len(self.memory) > 10:
                self.memory = self.memory[-10:]

        if response.get("scratch_pad_update"):
            self.scratch_pad = response["scratch_pad_update"]

        return response

    def get_ask(self) -> Optional[float]:
        if not self.last_response:
            return None
        ask = self.last_response.get("ask")
        if ask is None or ask == "null":
            return None
        try:
            return round(float(ask), 2)
        except (TypeError, ValueError):
            return None

    def get_message(self) -> Optional[str]:
        if not self.last_response or not self.seller_comms_enabled:
            return None
        msg = self.last_response.get("message_to_sellers")
        if msg is None or msg == "null":
            return None
        # Enforce gag constraint
        if self.is_gagged and len(msg) > 5:
            return msg[:5]
        return str(msg)

    def get_reasoning(self) -> dict:
        if not self.last_response:
            return {}
        return {
            "reflection": self.last_response.get("reflection", ""),
            "plan_for_this_hour": self.last_response.get("plan_for_this_hour", ""),
            "new_memory": self.last_response.get("new_memory", ""),
            "scratch_pad_update": self.last_response.get("scratch_pad_update", ""),
        }
