# API Rate Limits

## Default Limits

Every NimbusPay API key is subject to a default rate limit of 100 requests per minute, with a burst allowance of up to 150 requests in any 10-second window. Limits are enforced per API key, not per merchant account, so a merchant with multiple keys (e.g. one for production, one for testing) has independent limits for each.

## Rate Limit Headers

Every API response includes three headers indicating current rate-limit status:

- `X-RateLimit-Limit` — the total requests allowed per minute
- `X-RateLimit-Remaining` — requests remaining in the current window
- `X-RateLimit-Reset` — Unix timestamp when the current window resets

## Exceeding the Limit

Requests beyond the limit receive an HTTP 429 response with a `Retry-After` header indicating how many seconds to wait before retrying. NimbusPay recommends exponential backoff rather than immediate retry.

## Requesting a Higher Limit

Enterprise plan merchants can request an increased rate limit by contacting their account manager. Increased limits are applied at the API-key level within 2 business days of approval.
