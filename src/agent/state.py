"""State schema for The Planeswalker Agent."""

from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict):
    """
    State object that flows through the LangGraph workflow.

    Attributes:
        user_query: The original user query
        query_type: Classification of query ("constructed" or "limited")
        oracle_results: Semantic card search results from ChromaDB
        synergy_results: Card synergies from NetworkX synergy graph
        metagame_results: Statistics from EDHREC or 17Lands
        final_response: The synthesized answer to return to the user
        metadata: Additional context and debugging information
    """
    user_query: str
    query_type: Optional[str]
    oracle_results: Optional[List[Dict[str, Any]]]
    synergy_results: Optional[Dict[str, Any]]
    metagame_results: Optional[Dict[str, Any]]
    final_response: Optional[str]
    metadata: Dict[str, Any]
