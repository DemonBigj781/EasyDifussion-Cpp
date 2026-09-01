# Split Attention feature

Structural placeholder for Split Attention as a separate attention feature beside xFormers.

This stage defines only the feature location and architecture. Backend method names and implementation details are intentionally deferred until Split Attention's required operations are documented.

Planned architecture follows the same translate-first, unify-second model:

- backend-specific translation/definitions live under `[backend]/definition/`
- unified cross-backend code will live directly under `common/`
- application code should ultimately call only the common feature API

Do not infer that Split Attention uses the same method set as xFormers unless that is established by later analysis.
