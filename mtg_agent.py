"""
The Planeswalker Agent - Main Entry Point

An AI agent for Magic: The Gathering that combines semantic card search
with metagame statistics to provide expert recommendations.
"""

import sys
import time
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import (
    router_node,
    oracle_node,
    synergy_node,
    constructed_metagame_node,
    limited_metagame_node,
    synthesizer_node
)
from src.data.chroma import get_vector_store
from src.cognitive import get_synergy_graph


def _initialize_resources():
    """
    Pre-initialize expensive resources (VectorStore and SynergyGraph).

    This ensures that the first query doesn't have to wait for initialization.
    Resources are loaded as singletons and cached for subsequent queries.
    """
    print("\n[Agent] Pre-initializing resources...")
    start_time = time.time()

    # Initialize VectorStore (ChromaDB)
    get_vector_store()

    # Initialize SynergyGraph
    get_synergy_graph()

    elapsed = time.time() - start_time
    print(f"[Agent] Resources initialized in {elapsed:.2f}s\n")


def route_to_metagame(state: AgentState) -> str:
    """Conditional edge: Route to appropriate metagame node."""
    query_type = state.get("query_type")
    if query_type == "limited":
        return "limited_metagame"
    elif query_type in ["constructed", "standard", "modern", "pioneer", "legacy", "pauper", "commander"]:
        return "constructed_metagame"
    else:
        return "synthesizer"


def create_agent() -> StateGraph:
    """
    Create the LangGraph StateGraph workflow.

    Workflow:
    1. Router: Classify query as Constructed or Limited
    2. Oracle: Semantic card search (always runs)
    3. Synergy: Analyze card interactions (always runs)
    4. Metagame: Fetch EDHREC or 17Lands data (conditional)
    5. Synthesizer: Combine results into final response

    Returns:
        Compiled StateGraph ready to execute
    """
    # Pre-initialize expensive resources (VectorStore + SynergyGraph)
    # This ensures fast response times for all queries
    _initialize_resources()

    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("oracle", oracle_node)
    workflow.add_node("synergy", synergy_node)
    workflow.add_node("constructed_metagame", constructed_metagame_node)
    workflow.add_node("limited_metagame", limited_metagame_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # Set entry point
    workflow.set_entry_point("router")

    # Add edges
    # Router -> Oracle (always)
    workflow.add_edge("router", "oracle")

    # Oracle -> Synergy (always)
    workflow.add_edge("oracle", "synergy")

    # Synergy -> Metagame (conditional based on query type)
    workflow.add_conditional_edges(
        "synergy",
        route_to_metagame,
        {
            "constructed_metagame": "constructed_metagame",
            "limited_metagame": "limited_metagame",
            "synthesizer": "synthesizer"
        }
    )

    # Metagame nodes -> Synthesizer
    workflow.add_edge("constructed_metagame", "synthesizer")
    workflow.add_edge("limited_metagame", "synthesizer")

    # Synthesizer -> End
    workflow.add_edge("synthesizer", END)

    # Compile the graph
    return workflow.compile()


def run_query(agent: StateGraph, query: str) -> Dict[str, Any]:
    """
    Execute a query through the agent workflow.

    Args:
        agent: Compiled StateGraph agent
        query: User's question or request

    Returns:
        Final state dictionary with results
    """
    # Initialize state
    initial_state: AgentState = {
        "user_query": query,
        "query_type": None,
        "oracle_results": None,
        "synergy_results": None,
        "metagame_results": None,
        "final_response": None,
        "metadata": {}
    }

    print("\n" + "="*60)
    print("PLANESWALKER AGENT")
    print("="*60)
    print()

    # Run the workflow
    final_state = agent.invoke(initial_state)

    return final_state


def interactive_mode():
    """Run the agent in interactive CLI mode."""
    print("\n" + "="*60)
    print("THE PLANESWALKER AGENT - Interactive Mode")
    print("="*60)
    print("\nAn AI expert for Magic: The Gathering")
    print("Combining card knowledge with metagame statistics\n")
    print("Commands:")
    print("  - Type your question about MTG")
    print("  - 'quit' or 'exit' to stop")
    print("  - 'help' for examples")
    print("="*60)
    print()

    # Create agent once
    agent = create_agent()

    while True:
        try:
            # Get user input
            query = input("\n> ").strip()

            if not query:
                continue

            # Handle commands
            if query.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break

            if query.lower() == "help":
                print("\nExample questions:")
                print("  - What are good cards for an Atraxa deck?")
                print("  - Show me powerful Commander staples")
                print("  - What's the best color pair in MKM draft?")
                print("  - Find me cards that draw when they enter")
                print("  - What removal spells work in Commander?")
                continue

            # Run the query
            result = run_query(agent, query)

            # Display response
            if result.get("final_response"):
                print("\n" + result["final_response"])
            else:
                print("\nNo response generated. Please try a different query.")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Single query mode
        query = " ".join(sys.argv[1:])
        agent = create_agent()
        result = run_query(agent, query)

        if result.get("final_response"):
            print("\n" + result["final_response"])
        else:
            print("\nNo response generated.")
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()
