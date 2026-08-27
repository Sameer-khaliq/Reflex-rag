# Chargeback Disputes

A chargeback dispute occurs when a merchant's customer contacts their own card-issuing bank to reverse a charge, rather than contacting the merchant or NimbusPay support directly. This is a formal card-network process, separate from any billing question the merchant themselves might raise about their own NimbusPay invoice.

## How a Chargeback Starts

The customer's bank notifies NimbusPay of the chargeback, along with a reason code (e.g. "product not received", "unauthorized transaction", "duplicate charge"). NimbusPay immediately notifies the merchant and places a hold on the disputed transaction amount.

## Merchant Response Window

Merchants have 7 calendar days from notification to submit evidence contesting the chargeback, or to accept it. See `dispute_evidence_submission.md` for what evidence is accepted and how to submit it.

## Chargeback Fees

A non-refundable chargeback processing fee is applied to the merchant's account regardless of the chargeback's outcome, since NimbusPay incurs this fee from the card network either way.

## This Is Not the Same as a Billing Complaint

If a merchant themselves has a question or complaint about a line item on their own NimbusPay invoice, that is not a chargeback and is not handled through this process — see `billing_discrepancy_review.md` instead.
