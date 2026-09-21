"""
Pytest suite for KnightCash API (/balance and /transfer).

Tests encode the intended contract (happy path, negatives, edges, boundaries).
Failures against the current implementation surface the deliberate lab defects.
Requires: Python 3.13+, fastapi, httpx, pytest.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
from fastapi.testclient import TestClient

import knight_cash_api as api
from knight_cash_api import MAX_TRANSFER_AMOUNT, app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEED_ACCOUNTS: dict[str, dict[str, Any]] = {
    "user-1": {"name": "Alice", "balance": 500.00, "active": True},
    "user-2": {"name": "Bob", "balance": 150.00, "active": True},
    "user-3": {"name": "Carla", "balance": 0.00, "active": True},
    "user-4": {"name": "Dax", "balance": 75.00, "active": False},
}


@pytest.fixture
def client() -> TestClient:
    """Fresh TestClient bound to the KnightCash FastAPI app."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_accounts_db() -> None:
    """Restore the in-memory DB before every test so cases stay isolated."""
    api.accounts_db.clear()
    api.accounts_db.update(copy.deepcopy(SEED_ACCOUNTS))


@pytest.fixture
def alice() -> str:
    """Active account with a healthy balance (500.00)."""
    return "user-1"


@pytest.fixture
def bob() -> str:
    """Active account with a moderate balance (150.00)."""
    return "user-2"


@pytest.fixture
def carla() -> str:
    """Active account with a zero balance."""
    return "user-3"


@pytest.fixture
def dax() -> str:
    """Inactive account (still present in the DB)."""
    return "user-4"


@pytest.fixture
def unknown_user() -> str:
    """User id that does not exist in the accounts DB."""
    return "user-does-not-exist"


def _transfer_payload(
    sender_id: str,
    recipient_id: str,
    amount: float,
) -> dict[str, Any]:
    """Build a JSON body for POST /transfer."""
    return {
        "sender_id": sender_id,
        "recipient_id": recipient_id,
        "amount": amount,
    }


# ===========================================================================
# GET /balance/{user_id}
# ===========================================================================


class TestBalanceHappyPath:
    """Successful balance lookups for known active accounts."""

    def test_active_account_returns_200_and_balance(self, client, alice):
        response = client.get(f"/balance/{alice}")

        assert response.status_code == 200
        body = response.json()
        assert body["user_id"] == alice
        assert body["name"] == "Alice"
        assert body["balance"] == pytest.approx(500.00)

    def test_zero_balance_active_account_still_returns_balance(self, client, carla):
        response = client.get(f"/balance/{carla}")

        assert response.status_code == 200
        body = response.json()
        assert body["user_id"] == carla
        assert body["name"] == "Carla"
        assert body["balance"] == pytest.approx(0.00)

    def test_balance_response_shape(self, client, bob):
        response = client.get(f"/balance/{bob}")

        assert response.status_code == 200
        body = response.json()
        assert set(body.keys()) == {"user_id", "name", "balance"}
        assert isinstance(body["balance"], (int, float))


class TestBalanceNegativeAndEdge:
    """Missing users, inactive accounts, and odd path values."""

    def test_unknown_user_returns_404(self, client, unknown_user):
        response = client.get(f"/balance/{unknown_user}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_inactive_account_returns_403(self, client, dax):
        response = client.get(f"/balance/{dax}")

        assert response.status_code == 403
        detail = response.json()["detail"].lower()
        assert "inactive" in detail
        assert dax in response.json()["detail"]

    def test_empty_user_id_path_is_not_found(self, client):
        # FastAPI treats trailing slash /balance/ as a routing miss for this path.
        response = client.get("/balance/")
        assert response.status_code in {404, 405, 422}

    def test_balance_does_not_mutate_account(self, client, alice):
        before = copy.deepcopy(api.accounts_db[alice])
        client.get(f"/balance/{alice}")
        assert api.accounts_db[alice] == before


# ===========================================================================
# POST /transfer — happy path
# ===========================================================================


class TestTransferHappyPath:
    """Valid transfers between distinct active accounts with sufficient funds."""

    def test_successful_transfer_updates_both_balances(self, client, alice, bob):
        amount = 50.00
        alice_before = api.accounts_db[alice]["balance"]
        bob_before = api.accounts_db[bob]["balance"]

        response = client.post(
            "/transfer",
            json=_transfer_payload(alice, bob, amount),
        )

        assert response.status_code == 200
        body = response.json()
        assert body["message"] == "Transfer successful"
        assert body["sender_id"] == alice
        assert body["recipient_id"] == bob
        assert body["amount"] == pytest.approx(amount)
        assert body["sender_balance"] == pytest.approx(alice_before - amount)
        assert body["recipient_balance"] == pytest.approx(bob_before + amount)
        assert api.accounts_db[alice]["balance"] == pytest.approx(alice_before - amount)
        assert api.accounts_db[bob]["balance"] == pytest.approx(bob_before + amount)

    def test_small_positive_transfer(self, client, alice, bob):
        response = client.post(
            "/transfer",
            json=_transfer_payload(alice, bob, 0.01),
        )

        assert response.status_code == 200
        assert response.json()["amount"] == pytest.approx(0.01)

    def test_transfer_entire_balance_when_exact(self, client, bob, alice):
        """Sender may transfer exactly their full balance (boundary: balance == amount)."""
        amount = api.accounts_db[bob]["balance"]

        response = client.post(
            "/transfer",
            json=_transfer_payload(bob, alice, amount),
        )

        assert response.status_code == 200
        assert api.accounts_db[bob]["balance"] == pytest.approx(0.00)
        assert response.json()["sender_balance"] == pytest.approx(0.00)


# ===========================================================================
# POST /transfer — identity / same-account guards
# ===========================================================================


class TestTransferSameAccount:
    """Sender and recipient must be different accounts (value equality, not identity)."""

    def test_same_sender_and_recipient_rejected(self, client, alice):
        response = client.post(
            "/transfer",
            json=_transfer_payload(alice, alice, 10.00),
        )

        assert response.status_code == 400
        assert "differ" in response.json()["detail"].lower()

    def test_same_account_does_not_change_balance(self, client, alice):
        before = api.accounts_db[alice]["balance"]
        client.post("/transfer", json=_transfer_payload(alice, alice, 10.00))
        assert api.accounts_db[alice]["balance"] == pytest.approx(before)


# ===========================================================================
# POST /transfer — amount validation (negative, zero, max boundary)
# ===========================================================================


class TestTransferAmountValidation:
    """Non-positive amounts and amounts above MAX_TRANSFER_AMOUNT."""

    def test_negative_amount_rejected(self, client, alice, bob):
        response = client.post(
            "/transfer",
            json=_transfer_payload(alice, bob, -1.00),
        )

        assert response.status_code == 400
        assert "greater than zero" in response.json()["detail"].lower()

    def test_large_negative_amount_rejected(self, client, alice, bob):
        response = client.post(
            "/transfer",
            json=_transfer_payload(alice, bob, -999_999.99),
        )

        assert response.status_code == 400

    def test_zero_amount_rejected(self, client, alice, bob):
        """Zero is not a valid transfer (boundary: amount == 0)."""
        response = client.post(
            "/transfer",
            json=_transfer_payload(alice, bob, 0.0),
        )

        assert response.status_code == 400
        assert "greater than zero" in response.json()["detail"].lower()

    def test_amount_exactly_at_max_accepted(self, client, alice, bob):
        """Boundary: amount == MAX_TRANSFER_AMOUNT should be allowed if funds exist."""
        api.accounts_db[alice]["balance"] = MAX_TRANSFER_AMOUNT + 1.00

        response = client.post(
            "/transfer",
            json=_transfer_payload(alice, bob, MAX_TRANSFER_AMOUNT),
        )

        assert response.status_code == 200
        assert response.json()["amount"] == pytest.approx(MAX_TRANSFER_AMOUNT)

    def test_amount_just_over_max_rejected(self, client, alice, bob):
        """Boundary: amount == MAX_TRANSFER_AMOUNT + epsilon must be rejected."""
        api.accounts_db[alice]["balance"] = MAX_TRANSFER_AMOUNT + 100.00
        over = MAX_TRANSFER_AMOUNT + 0.01

        response = client.post(
            "/transfer",
            json=_transfer_payload(alice, bob, over),
        )

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "exceeds maximum" in detail.lower()
        assert str(int(MAX_TRANSFER_AMOUNT)) in detail or str(MAX_TRANSFER_AMOUNT) in detail

    def test_amount_far_above_max_rejected(self, client, alice, bob):
        api.accounts_db[alice]["balance"] = 1_000_000.00

        response = client.post(
            "/transfer",
            json=_transfer_payload(alice, bob, 50_000.00),
        )

        assert response.status_code == 400


# ===========================================================================
# POST /transfer — account existence
# ===========================================================================


class TestTransferMissingAccounts:
    """Unknown sender/recipient must yield clean 404s, not 500s."""

    def test_unknown_sender_returns_404(self, client, bob, unknown_user):
        response = client.post(
            "/transfer",
            json=_transfer_payload(unknown_user, bob, 10.00),
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_unknown_recipient_returns_404(self, client, alice, unknown_user):
        response = client.post(
            "/transfer",
            json=_transfer_payload(alice, unknown_user, 10.00),
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_unknown_recipient_does_not_debit_sender(self, client, alice, unknown_user):
        before = api.accounts_db[alice]["balance"]
        client.post(
            "/transfer",
            json=_transfer_payload(alice, unknown_user, 10.00),
        )
        assert api.accounts_db[alice]["balance"] == pytest.approx(before)

    def test_both_parties_unknown_returns_404(self, client, unknown_user):
        response = client.post(
            "/transfer",
            json=_transfer_payload(unknown_user, "also-missing", 10.00),
        )

        assert response.status_code == 404


# ===========================================================================
# POST /transfer — inactive accounts
# ===========================================================================


class TestTransferInactiveAccounts:
    """Inactive sender or recipient must be forbidden."""

    def test_inactive_sender_returns_403(self, client, dax, alice):
        response = client.post(
            "/transfer",
            json=_transfer_payload(dax, alice, 10.00),
        )

        assert response.status_code == 403
        assert "inactive" in response.json()["detail"].lower()
        assert "sender" in response.json()["detail"].lower()

    def test_inactive_recipient_returns_403(self, client, alice, dax):
        response = client.post(
            "/transfer",
            json=_transfer_payload(alice, dax, 10.00),
        )

        assert response.status_code == 403
        assert "inactive" in response.json()["detail"].lower()
        assert "recipient" in response.json()["detail"].lower()

    def test_inactive_parties_do_not_mutate_balances(self, client, alice, dax):
        alice_before = api.accounts_db[alice]["balance"]
        dax_before = api.accounts_db[dax]["balance"]

        client.post("/transfer", json=_transfer_payload(alice, dax, 10.00))
        client.post("/transfer", json=_transfer_payload(dax, alice, 10.00))

        assert api.accounts_db[alice]["balance"] == pytest.approx(alice_before)
        assert api.accounts_db[dax]["balance"] == pytest.approx(dax_before)


# ===========================================================================
# POST /transfer — insufficient funds
# ===========================================================================


class TestTransferInsufficientFunds:
    """Transfers larger than the sender balance must be rejected."""

    def test_overdraft_rejected(self, client, bob, alice):
        amount = api.accounts_db[bob]["balance"] + 0.01

        response = client.post(
            "/transfer",
            json=_transfer_payload(bob, alice, amount),
        )

        assert response.status_code == 400
        assert "insufficient" in response.json()["detail"].lower()

    def test_zero_balance_sender_cannot_transfer_positive(self, client, carla, alice):
        response = client.post(
            "/transfer",
            json=_transfer_payload(carla, alice, 1.00),
        )

        assert response.status_code == 400
        assert "insufficient" in response.json()["detail"].lower()

    def test_failed_overdraft_leaves_balances_unchanged(self, client, bob, alice):
        bob_before = api.accounts_db[bob]["balance"]
        alice_before = api.accounts_db[alice]["balance"]

        client.post(
            "/transfer",
            json=_transfer_payload(bob, alice, bob_before + 1.00),
        )

        assert api.accounts_db[bob]["balance"] == pytest.approx(bob_before)
        assert api.accounts_db[alice]["balance"] == pytest.approx(alice_before)

    def test_amount_just_under_balance_succeeds(self, client, bob, alice):
        """Boundary: amount == balance - epsilon should succeed."""
        amount = api.accounts_db[bob]["balance"] - 0.01

        response = client.post(
            "/transfer",
            json=_transfer_payload(bob, alice, amount),
        )

        assert response.status_code == 200
        assert api.accounts_db[bob]["balance"] == pytest.approx(0.01)


# ===========================================================================
# POST /transfer — request validation / type errors
# ===========================================================================


class TestTransferRequestValidation:
    """Malformed bodies should produce 422 from FastAPI/Pydantic, not 500."""

    def test_missing_amount_field(self, client, alice, bob):
        response = client.post(
            "/transfer",
            json={"sender_id": alice, "recipient_id": bob},
        )
        assert response.status_code == 422

    def test_missing_sender_id(self, client, bob):
        response = client.post(
            "/transfer",
            json={"recipient_id": bob, "amount": 10.0},
        )
        assert response.status_code == 422

    def test_missing_recipient_id(self, client, alice):
        response = client.post(
            "/transfer",
            json={"sender_id": alice, "amount": 10.0},
        )
        assert response.status_code == 422

    def test_amount_as_non_numeric_string(self, client, alice, bob):
        response = client.post(
            "/transfer",
            json={
                "sender_id": alice,
                "recipient_id": bob,
                "amount": "not-a-number",
            },
        )
        assert response.status_code == 422

    def test_amount_as_null(self, client, alice, bob):
        response = client.post(
            "/transfer",
            json={"sender_id": alice, "recipient_id": bob, "amount": None},
        )
        assert response.status_code == 422

    def test_empty_body(self, client):
        response = client.post("/transfer", json={})
        assert response.status_code == 422

    def test_wrong_http_method_on_transfer(self, client):
        response = client.get("/transfer")
        assert response.status_code == 405


# ===========================================================================
# Cross-endpoint / sequencing edge cases
# ===========================================================================


class TestCrossEndpointConsistency:
    """Balance endpoint should reflect transfer side-effects."""

    def test_balance_reflects_successful_transfer(self, client, alice, bob):
        amount = 25.00
        transfer = client.post(
            "/transfer",
            json=_transfer_payload(alice, bob, amount),
        )
        assert transfer.status_code == 200

        alice_bal = client.get(f"/balance/{alice}")
        bob_bal = client.get(f"/balance/{bob}")

        assert alice_bal.status_code == 200
        assert bob_bal.status_code == 200
        assert alice_bal.json()["balance"] == pytest.approx(500.00 - amount)
        assert bob_bal.json()["balance"] == pytest.approx(150.00 + amount)

    def test_chained_transfers_accumulate(self, client, alice, bob):
        r1 = client.post("/transfer", json=_transfer_payload(alice, bob, 10.00))
        r2 = client.post("/transfer", json=_transfer_payload(alice, bob, 15.00))

        assert r1.status_code == 200
        assert r2.status_code == 200
        assert api.accounts_db[alice]["balance"] == pytest.approx(475.00)
        assert api.accounts_db[bob]["balance"] == pytest.approx(175.00)


# ===========================================================================
# Parametrized boundary matrix
# ===========================================================================


@pytest.mark.parametrize(
    ("amount", "expect_status"),
    [
        (-0.01, 400),
        (0.0, 400),
        (0.01, 200),
        (1.0, 200),
        (MAX_TRANSFER_AMOUNT, 200),
        (MAX_TRANSFER_AMOUNT + 0.01, 400),
    ],
    ids=[
        "just_below_zero",
        "exactly_zero",
        "penny",
        "one_dollar",
        "exactly_max",
        "just_over_max",
    ],
)
def test_transfer_amount_boundary_matrix(
    client,
    alice,
    bob,
    amount,
    expect_status,
):
    """Compact boundary sweep for amount validation (funds topped up as needed)."""
    api.accounts_db[alice]["balance"] = MAX_TRANSFER_AMOUNT + 100.00

    response = client.post(
        "/transfer",
        json=_transfer_payload(alice, bob, amount),
    )
    assert response.status_code == expect_status
