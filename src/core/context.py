import logging
import tiktoken
from src.schemas.state import AgentState

logger = logging.getLogger(__name__)

class ContextManager:
    @staticmethod
    def get_token_count(text: str, model: str = "gpt-4o") -> int:
        """Accurately count tokens using tiktoken."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("o200k_base")
        return len(encoding.encode(text))

    @staticmethod
    def check_budget(state: AgentState, agent_name: str, new_text: str) -> bool:
        """
        Exposed method for agents to call to check remaining budget before adding to context.
        """
        budget = state.get("context_budget", {}).get(agent_name)
        if not budget:
            return True # No budget explicitly set, assume infinite
            
        new_tokens = ContextManager.get_token_count(new_text)
        return (budget.used_tokens + new_tokens) <= budget.max_tokens

    @staticmethod
    def consume_budget(state: AgentState, agent_name: str, tokens: int) -> bool:
        """
        Consumes tokens. 
        Agents that ignore budget constraints and overflow must be caught and logged as a policy violation.
        """
        budget = state.get("context_budget", {}).get(agent_name)
        if not budget:
            return True
            
        budget.used_tokens += tokens
        if budget.used_tokens > budget.max_tokens:
            logger.error(
                f"POLICY VIOLATION: Agent '{agent_name}' exceeded context budget "
                f"({budget.used_tokens}/{budget.max_tokens} tokens)"
            )
            return False
            
        return True
