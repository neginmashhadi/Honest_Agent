from dataclasses import dataclass, field
from typing import Optional
from src.agents.base_agent import render_template, call_llm, parse_json_response


@dataclass
class BuyerAgent:
    buyer_id: str
    company: str
    model: str
    valuation: float
    num_rounds: int

    memory: list[str] = field(default_factory=list)
    scratch_pad: str = ""
    last_response: Optional[dict] = None

    def step(
        self,
        round_number: int,
        bid_queue: str,
        ask_queue: str,
        past_bids_and_asks: str,
        past_trades: str,
        agent_successful_trades: str,
    ) -> dict:
        memory_text = "\n".join(self.memory) if self.memory else "No memories yet."
        prompt = render_template(
            "buyer_prompt.j2",
            buyer_id=self.buyer_id,
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
        )

        raw = call_llm(self.model, prompt)
        response = parse_json_response(raw)
        self.last_response = response

        if response.get("new_memory"):
            self.memory.append(f"Hour {round_number}: {response['new_memory']}")
            if len(self.memory) > 10:
                self.memory = self.memory[-10:]

        if response.get("scratch_pad_update"):
            self.scratch_pad = response["scratch_pad_update"]

        return response

    def get_bid(self) -> Optional[float]:
        if not self.last_response:
            return None
        bid = self.last_response.get("bid")
        if bid is None or bid == "null":
            return None
        try:
            return round(float(bid), 2)
        except (TypeError, ValueError):
            return None
