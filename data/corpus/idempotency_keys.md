# Idempotency Keys

## Purpose

An idempotency key ensures that if a request is sent more than once (for example, due to a network retry), NimbusPay processes it only once — preventing duplicate charges or duplicate refunds.

## How to Use One

Include an `Idempotency-Key` header with any POST request that creates or modifies a resource (charges, refunds, payouts). The value should be a unique string per logical operation, such as a UUID generated client-side before the first attempt.

## Retention Window

NimbusPay stores idempotency keys for 24 hours. If the same key is reused after 24 hours, it is treated as a new, independent request rather than a duplicate.

## Conflicting Requests

If the same idempotency key is reused with a different request body within the 24-hour window, NimbusPay returns an HTTP 409 Conflict rather than processing either version, since it cannot determine which payload was intended.
