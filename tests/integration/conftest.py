import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mtg_agent import create_agent


@pytest.fixture(scope="session")
def agent():
    """Create agent once per test session to avoid reloading heavy resources."""
    return create_agent()


@pytest.fixture
def similarity_threshold():
    """Default similarity threshold for comparing responses."""
    return 0.6


@pytest.fixture
def test_results_tracker():
    """Track test results for final success rate calculation."""
    results = {"passed": 0, "failed": 0, "tests": []}
    yield results
    # Final summary is printed by test runner


@pytest.fixture
def audit_logger():
    """Helper to log detailed test audit information."""
    def log_test_info(query, source_url, expected_data, test_name, agent_response=None):
        """
        Log detailed audit information for a test.

        Args:
            query: The query string being tested
            source_url: URL to the online source for manual verification
            expected_data: Dictionary of expected data from the source
            test_name: Name of the test for context
            agent_response: Optional agent response to display
        """
        separator = "=" * 80
        print(f"\n{separator}")
        print(f"TEST AUDIT: {test_name}")
        print(f"{separator}")
        print(f"\nQUERY:")
        print(f"  {query}")
        print(f"\nSOURCE URL (for manual audit):")
        print(f"  {source_url}")
        print(f"\nEXPECTED DATA FROM SOURCE:")

        if isinstance(expected_data, dict):
            for key, value in expected_data.items():
                if isinstance(value, list):
                    print(f"  {key}:")
                    for i, item in enumerate(value[:10], 1):  # Limit to first 10 items
                        if isinstance(item, dict):
                            print(f"    {i}. {item}")
                        else:
                            print(f"    {i}. {item}")
                    if len(value) > 10:
                        print(f"    ... and {len(value) - 10} more")
                else:
                    print(f"  {key}: {value}")
        elif isinstance(expected_data, list):
            print(f"  Found {len(expected_data)} items:")
            for i, item in enumerate(expected_data[:10], 1):
                print(f"    {i}. {item}")
            if len(expected_data) > 10:
                print(f"    ... and {len(expected_data) - 10} more")
        else:
            print(f"  {expected_data}")

        if agent_response:
            print(f"\nAGENT RESPONSE:")
            # Wrap text at 80 characters for readability
            response_lines = agent_response.split('\n')
            for line in response_lines:
                if len(line) <= 76:
                    print(f"  {line}")
                else:
                    # Wrap long lines
                    words = line.split()
                    current_line = "  "
                    for word in words:
                        if len(current_line) + len(word) + 1 <= 78:
                            current_line += word + " "
                        else:
                            print(current_line.rstrip())
                            current_line = "  " + word + " "
                    if current_line.strip():
                        print(current_line.rstrip())

        print(f"\n{separator}\n")

    return log_test_info
