"""Mock falsification engine for testing purposes."""

from typing import Any, Dict, Optional


class FalsificationEngine:
    """Mock falsification engine for testing purposes."""

    def __init__(self) -> None:
        self.test_results: Dict[str, Any] = {}

    def run_falsification_test(self, test_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run a falsification test."""
        result_id = f"test_{hash(str(parameters)) % 10000:04d}"
        result = {
            "result_id": result_id,
            "test_name": test_name,
            "framework_falsified": False,
            "confidence_level": 0.95,
            "p_value": 0.03,
            "parameters": parameters,
            "timestamp": "2024-01-01T00:00:00Z",
        }
        self.test_results[result_id] = result
        return result

    def get_result(self, result_id: str) -> Optional[Dict[str, Any]]:
        """Get test result by ID."""
        return self.test_results.get(result_id)
