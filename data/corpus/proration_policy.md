# Proration Policy

When a merchant changes plans in the middle of a billing cycle, NimbusPay calculates a prorated charge or credit rather than waiting until the next cycle.

## How Proration Is Calculated

Proration is calculated on a daily basis using the number of days remaining in the current cycle:

```
prorated_amount = (new_plan_price - old_plan_price) / days_in_cycle * days_remaining
```

If the result is positive (an upgrade), the merchant is charged the prorated difference immediately. If negative (a downgrade), the difference is applied as account credit toward the next invoice.

## Downgrades and Usage Overage

If a merchant downgrades to a plan with a lower included-transaction limit, and their usage so far in the current cycle already exceeds the new plan's limit, the overage is billed at the new plan's overage rate for the remainder of the cycle.

## Annual-to-Monthly Conversion

Switching from an annual plan to a monthly plan mid-term is treated as a cancellation of the annual plan (prorated refund of unused months as account credit) followed by a new monthly subscription starting immediately.
