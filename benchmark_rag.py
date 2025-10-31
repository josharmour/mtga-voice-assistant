import time
import logging
from rag_advisor import RAGSystem, load_sample_17lands_data

# Setup logging to see RAG system output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def run_benchmark(rag_system, board_state, num_runs=20):
    """
    Runs the benchmark for the RAG system.

    Args:
        rag_system: An initialized RAGSystem instance.
        board_state: A mock board state dictionary.
        num_runs: The number of times to call enhance_prompt.

    Returns:
        A dictionary with benchmark results.
    """
    timings = []
    base_prompt = "Analyze the current board state and provide tactical advice."

    for i in range(num_runs):
        start_time = time.perf_counter()
        _ = rag_system.enhance_prompt(board_state, base_prompt)
        end_time = time.perf_counter()
        duration = end_time - start_time
        timings.append(duration)
        logging.info(f"Run {i+1}/{num_runs} completed in {duration:.4f} seconds.")

    total_time = sum(timings)
    avg_time = total_time / num_runs
    min_time = min(timings)
    max_time = max(timings)

    return {
        "total_time": total_time,
        "average_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "num_runs": num_runs
    }

def main():
    """
    Main function to set up and run the benchmark.
    """
    logging.info("--- Starting RAG System Benchmark ---")

    # 1. Initialize RAG System
    logging.info("Initializing RAG system...")
    rag = RAGSystem(rules_path="data/MagicCompRules.txt")

    # Initialize rules (force recreate to ensure a clean state)
    rag.initialize_rules(force_recreate=True)

    # Load sample card data
    logging.info("Loading sample card statistics...")
    load_sample_17lands_data(rag.card_stats)

    # 2. Define Mock Board State
    mock_board_state = {
        'phase': 'combat',
        'stack_size': 1,
        'battlefield': {
            'player': [
                {'name': 'Llanowar Elves'},
                {'name': 'Grizzly Bears'} # A card not in our sample stats
            ],
            'opponent': [
                {'name': 'Lightning Bolt'}, # In hand, but we check all battlefield
                {'name': 'Counterspell'}
            ]
        }
    }
    logging.info("Using mock board state for benchmark.")

    # 3. Run Benchmark
    logging.info("--- Running Benchmark ---")
    results = run_benchmark(rag, mock_board_state)
    logging.info("--- Benchmark Complete ---")

    # 4. Report Results
    print("\n--- Benchmark Results ---")
    print(f"Number of runs:      {results['num_runs']}")
    print(f"Total time:          {results['total_time']:.4f} seconds")
    print(f"Average time per call: {results['average_time']:.4f} seconds")
    print(f"Fastest call:        {results['min_time']:.4f} seconds")
    print(f"Slowest call:        {results['max_time']:.4f} seconds")
    print("-------------------------\n")

    rag.close()

if __name__ == "__main__":
    main()