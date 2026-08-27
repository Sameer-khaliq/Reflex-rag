# Billing Overview

NimbusPay bills merchants on a recurring subscription cycle tied to their selected plan: Starter, Growth, or Enterprise. Each plan has a fixed monthly or annual base fee plus usage-based transaction fees calculated at the end of each billing cycle.

## Billing Cycle

A billing cycle begins on the date a merchant's subscription was first activated and renews automatically every 30 days for monthly plans, or every 365 days for annual plans. Invoices are generated within 24 hours of the cycle closing and are available immediately in the merchant dashboard under Billing > Invoices.

## Plan Tiers

- **Starter** — Monthly billing only, up to 500 transactions/month included, overage billed per-transaction.
- **Growth** — Monthly or annual billing, up to 5,000 transactions/month included, discounted overage rate.
- **Enterprise** — Annual billing with custom negotiated terms, volume discounts, and a dedicated account manager. Enterprise contract terms (including any custom SLAs) are documented in the merchant's individual signed agreement, not in this public documentation.

## Upgrading or Downgrading

Merchants can change plans at any time from the dashboard. Plan changes take effect immediately, and any cost difference for the remainder of the current cycle is handled according to NimbusPay's proration policy (see `proration_policy.md`).

## Payment Failure

If a scheduled subscription charge fails, NimbusPay does not immediately suspend the account. See `failed_payments_dunning.md` for the retry schedule and grace period.
