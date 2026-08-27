# Refund Processing — General

NimbusPay processes approved refunds back to whichever payment method the original charge was made with. Processing time varies by payment method, since each method settles through a different underlying network.

## Card Payments

Card refund timing is documented separately — see `refund_processing_card.md`.

## Bank Transfer (ACH) Payments

Refunds for charges originally paid via ACH bank transfer are processed differently from card refunds, since ACH transactions do not support the same real-time reversal mechanism that card networks provide. An ACH refund is issued as a new, separate ACH credit transaction back to the merchant's bank account rather than a reversal of the original debit. Processing time depends on the receiving bank's own ACH intake schedule, and NimbusPay does not guarantee a fixed timeframe for ACH refund completion. Merchants should confirm receipt directly with their bank rather than relying on an estimated date from NimbusPay.

## Wallet Credit Refunds

If a refund is issued as NimbusPay wallet credit instead of returning to the original payment method, the credit is applied to the merchant's account balance instantly and is available for immediate use.

## Refund Status Tracking

The status of any refund (Pending, Submitted, Completed, or Failed) is visible in Billing > Refunds at all times, regardless of the original payment method.
