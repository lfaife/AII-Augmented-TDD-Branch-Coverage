"""
KnightCash API
--------------
A simulated financial transaction service.

NOTE: This is a SAMPLE reconstruction, not the original file. The real
knight_cash_api.py referenced by the Week 4 assignment/slides was not found
in the Hands-On doc, the Assignment doc, or the slide deck — per Slide 36,
the actual spec/file is distributed separately on Canvas. This sample is
built to match the description on Slide 32 ("partially implemented Python
API with no tests", /transfer and /balance endpoints) plus the bug classes
called out on Slides 26-27 and in the Assignment (negative-amount transfers,
type errors, and 500 responses that leak internals) so it's usable as a
drop-in stand-in for practicing the AI-augmented TDD workflow.

The developer "claims it works perfectly" -- and it does, on the happy path.
Everything else is deliberately broken, on purpose, for the lab.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="KnightCash API")

# --- Mock "database" -------------------------------------------------------
accounts_db = {
    "user-1": {"name": "Alice", "balance": 500.00, "active": True},
    "user-2": {"name": "Bob", "balance": 150.00, "active": True},
    "user-3": {"name": "Carla", "balance": 0.00, "active": True},
    "user-4": {"name": "Dax", "balance": 75.00, "active": False},
}

MAX_TRANSFER_AMOUNT = 10_000.00


class TransferRequest(BaseModel):
    sender_id: str
    recipient_id: str
    amount: float


def get_account_or_404(user_id: str) -> dict:
    account = accounts_db.get(user_id)
    if account is None:
        raise HTTPException(status_code=404, detail=f"Account '{user_id}' not found")
    return account


@app.get("/balance/{user_id}")
def get_balance(user_id: str):
    account = get_account_or_404(user_id)

    if not account["active"]:
        raise HTTPException(status_code=403, detail=f"Account '{user_id}' is inactive")
    return {"user_id": user_id, "name": account["name"], "balance": account["balance"]}
 


@app.post("/transfer")
def transfer(request: TransferRequest):
    sender_id = request.sender_id
    recipient_id = request.recipient_id
    amount = request.amount

    if sender_id == recipient_id:
        raise HTTPException(status_code=400, detail="Sender and recipient must differ")
   

    if amount <= 0:
        raise HTTPException(status_code=400, detail="Transfer amount must be greater than zero")
   

    if amount > MAX_TRANSFER_AMOUNT:
        raise HTTPException(
            status_code=400,
            detail=f"Transfer amount exceeds maximum of {MAX_TRANSFER_AMOUNT}",
        )

    sender = get_account_or_404(sender_id)
    recipient = get_account_or_404(recipient_id)

    if not sender["active"]:
        raise HTTPException(status_code=403, detail=f"Sender account '{sender_id}' is inactive")

    if not recipient["active"]:
        raise HTTPException(status_code=403, detail=f"Recipient account '{recipient_id}' is inactive")

    if sender["balance"] < amount:
        raise HTTPException(status_code=400, detail="Insufficient funds")
   

    sender["balance"] -= amount
    recipient["balance"] += amount

    return {
        "message": "Transfer successful",
        "sender_id": sender_id,
        "recipient_id": recipient_id,
        "amount": amount,
        "sender_balance": sender["balance"],
        "recipient_balance": recipient["balance"],
    }