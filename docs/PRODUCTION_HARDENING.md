# RailNexus — Production Hardening

## Service boundaries

The API serves validated synthetic analytics and a persisted Stage 6 booking
propensity model. DuckDB is opened read-only for request-time analytics.

## Reliability controls

- liveness endpoint;
- readiness endpoint;
- model loaded once during application lifespan;
- bounded query limits;
- deterministic request validation;
- explicit API versioning;
- request IDs and processing-time headers.

## Model governance controls

- model metadata is loaded with the artifact;
- prediction threshold source must be `validation`;
- only the leakage-aware feature contract is exposed;
- post-booking and experiment-assignment fields are excluded;
- unknown request fields are rejected;
- synthetic-data disclosure is returned by the service.

## Security controls

- CORS is disabled unless an explicit allowlist is supplied;
- `X-Content-Type-Options: nosniff`;
- `X-Frame-Options: DENY`;
- `Cache-Control: no-store`;
- container runs as a non-root user;
- Compose drops Linux capabilities and enables `no-new-privileges`;
- container root filesystem is read-only in Compose.

## CI/CD controls

GitHub Actions rebuilds a deterministic synthetic smoke dataset, rebuilds the
analytics/model stages, validates Streamlit and API production configuration,
checks Ruff formatting/lint, runs pytest, and performs a Docker image build.

## Deployment note

The Docker image intentionally contains only the processed DuckDB file,
persisted model artifacts, application code and API dependencies. Raw synthetic
data and local development artifacts are excluded from the image.

For a real production system, DuckDB/model artifacts would normally be replaced
or distributed through governed object storage, a warehouse/lakehouse and a
model registry with authentication, observability and deployment controls.
