import unittest

from llm4drd.tools.verify_review_ui import is_cancelled_request_lifecycle


class VerifyReviewUiPredicateTests(unittest.TestCase):
    def test_accepts_exact_aborted_transport_lifecycle(self):
        self.assertTrue(is_cancelled_request_lifecycle(
            outcome="failed",
            failure=" net::ERR_ABORTED ",
            identity_matched=True,
            route_released_at_terminal=False,
        ))

    def test_rejects_finished_or_generic_network_failures(self):
        for kwargs in (
            {
                "outcome": "finished",
                "failure": None,
                "identity_matched": True,
                "route_released_at_terminal": False,
            },
            {
                "outcome": "failed",
                "failure": "net::ERR_CONNECTION_RESET",
                "identity_matched": True,
                "route_released_at_terminal": False,
            },
            {
                "outcome": "failed",
                "failure": "net::ERR_ABORTED",
                "identity_matched": False,
                "route_released_at_terminal": False,
            },
            {
                "outcome": "failed",
                "failure": "net::ERR_ABORTED",
                "identity_matched": True,
                "route_released_at_terminal": True,
            },
        ):
            with self.subTest(kwargs=kwargs):
                self.assertFalse(is_cancelled_request_lifecycle(**kwargs))

    def test_accepts_normalized_explicit_abort_or_cancel_messages(self):
        for failure in (
            "AbortError: The operation was aborted.",
            "Request was canceled by the client",
            "Fetch operation cancelled",
        ):
            with self.subTest(failure=failure):
                self.assertTrue(is_cancelled_request_lifecycle(
                    outcome="failed",
                    failure=failure,
                    identity_matched=True,
                    route_released_at_terminal=False,
                ))


if __name__ == "__main__":
    unittest.main()
