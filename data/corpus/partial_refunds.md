# Partial Refunds

## When a Partial Refund Applies

A partial refund is used when only a portion of an original charge should be returned to the merchant — for example, if a subscription was used for part of a billing period before an eligible cancellation, or if only some line items on an invoice were incorrect.

## Calculating the Amount

For subscription charges, the partial refund amount is calculated on a daily basis, similar to proration: the unused portion of the billing period is refunded, while the portion already consumed is retained.

## Multiple Partial Refunds

A single original charge can have more than one partial refund issued against it over time, as long as the cumulative refunded amount never exceeds the original charge amount.

## Processing Time

Partial refunds follow the same processing timeline as full refunds for the same payment method — see `refund_processing_card.md` for card payments and `refund_processing_general.md` for other methods.
