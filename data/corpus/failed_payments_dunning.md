# Failed Payments and Dunning

When a scheduled subscription charge fails, NimbusPay follows a fixed dunning (payment retry) schedule before taking any action on the merchant's account.

## Retry Schedule

1. **Attempt 1** — Immediately on the billing date.
2. **Attempt 2** — 3 days after the first failure.
3. **Attempt 3** — 5 days after the second failure.
4. **Final attempt** — 7 days after the third failure.

If all four attempts fail, the merchant's account is moved to a "payment overdue" state. Core payment-processing functionality is suspended, but the dashboard and historical data remain accessible.

## Dunning Emails

An email notification is sent to the merchant's billing contact after every failed attempt, explaining the reason for the decline (where available from the card network) and providing a link to update the payment method.

## Reactivation

Once a valid payment method successfully clears the outstanding balance, account access is restored automatically within minutes. No manual support request is required for standard reactivation.
