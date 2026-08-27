# Payment Methods

NimbusPay merchants can pay their subscription invoice using any of the following methods, managed under Billing > Payment Methods in the dashboard.

## Supported Methods

- **Credit or debit card** — Visa, Mastercard, American Express, and Discover are accepted. Cards are charged automatically on the billing cycle date.
- **ACH bank transfer** — Available to merchants on the Growth or Enterprise plan. ACH payments must be authorized at least 5 business days before the billing date to avoid a late-payment flag, since ACH settlement is not instant.
- **Wallet balance** — Merchants can preload a NimbusPay wallet balance, which is drawn down automatically before any other payment method is charged.

## Changing a Default Payment Method

A merchant can set a new default payment method at any time. The change applies starting from the next billing cycle; it does not retroactively affect an invoice that has already been generated.

## Card Expiry Handling

NimbusPay automatically attempts to update expiring cards using card-network account updater services where supported. If a card cannot be updated automatically, the merchant receives an email reminder 14 days before expiry.
