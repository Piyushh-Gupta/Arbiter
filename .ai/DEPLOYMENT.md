# Arbiter Deployment Guide

This document outlines the operational contract for deploying the Arbiter application in a production environment. Arbiter is designed with strict mathematical purity and fail-fast guarantees. It enforces rigid lifecycle contracts to ensure absolute operational safety.

## 1. Startup Guarantees (Fail-Fast Initialization)

Arbiter implements a strictly synchronous, deterministic boot sequence via the FastAPI `lifespan` hook. 

- **Validation:** On startup, the `AppConfig` validates all environment variables. The composition root (`src/core/bootstrap.py`) instantiates and binds all registries and dummy models. 
- **Failure Condition:** If *any* step fails (e.g. invalid profile pairing, missing environment variable, incompatible index), the application will crash violently with an exception (e.g. `ValidationError`) before binding to the HTTP port.
- **Lazy Initialization:** Arbiter explicitly **forbids** lazy initialization. A container is guaranteed to be 100% operationally ready if it successfully binds to its port.

## 2. Health Probes

Arbiter exposes two cloud-native health endpoints designed for Kubernetes or standard load-balancer orchestration:

### Liveness Probe (`GET /health/live`)
- **Purpose:** Indicates the HTTP event loop is alive and the process has not deadlocked.
- **Returns:** HTTP 200 `{"status": "alive"}`
- **Usage:** Bind to Kubernetes `livenessProbe`.

### Readiness Probe (`GET /health/ready`)
- **Purpose:** Indicates the fail-fast startup sequence succeeded and the `ArbiterPipeline` is mounted in application state.
- **Returns:** HTTP 200 `{"status": "ready"}` if initialized, HTTP 503 `{"status": "not_ready"}` otherwise.
- **Usage:** Bind to Kubernetes `readinessProbe`.

*Note: The readiness probe is strictly declarative of the startup success. It does not perform ongoing deep validation or reconstruct registries on every request, preserving stateless performance.*

## 3. Graceful Shutdown

Upon receiving a termination signal (SIGTERM/SIGINT), Arbiter initiates a graceful shutdown sequence.
- The pipeline reference is released from application state (`app.state.pipeline = None`).
- Lingering resources (when Component implementations are added in future milestones) are explicitly detached.
- **Guarantee:** No dangling connections or corrupted state are left behind.

## 4. Required Configuration

The configuration is managed by `Pydantic Settings`. Operational policy defaults are applied via the `environment` parameter.

| Environment Variable | Description | Default | Bounds / Allowed Values |
| --- | --- | --- | --- |
| `ARBITER_APP_NAME` | The application name. | `"Arbiter API"` | Any string |
| `ARBITER_ENVIRONMENT` | Operational environment. | `"development"` | `"development"`, `"production"`, `"test"` |
| `ARBITER_LOG__LEVEL` | Infrastructure logging level. | `"INFO"` | `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"` |

## 5. Structured Logging

Arbiter's domains and business logic contain **zero logging** to preserve pure immutability. All logging is constrained exclusively to the infrastructure and transport layers.
- In `production`, logging is forcibly formatted as JSON to integrate natively with centralized log aggregators (e.g., Datadog, ELK).
- Unhandled HTTP 500 exceptions are logged fully at the infrastructure layer but entirely sanitized for external HTTP API consumers.
