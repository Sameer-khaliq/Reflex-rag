# Refund Processing — Card Payments

When a refund is approved for a charge originally paid by credit or debit card, NimbusPay submits the refund to the card network within 1 business day of approval.

## Timing

Card refunds typically appear on the customer's statement within **5 to 10 business days** of submission, though the exact timing depends on the customer's card-issuing bank and is outside NimbusPay's direct control.

## Partial Card Refunds

A card refund can be issued for less than the full original charge amount. Partial refunds follow the same 5–10 business day timing as full refunds. See `partial_refunds.md` for how partial amounts are calculated.

## Failed Refund Submission

If a card refund submission fails (for example, because the original card has since expired), NimbusPay automatically notifies the merchant and offers to issue the refund as account credit instead.
