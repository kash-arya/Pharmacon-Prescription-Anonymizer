"""
Pharmacon Web Application Entrypoint.

Re-exports FastAPI application and concurrency semaphore from api.server.
"""

try:
    from api.server import app, PIPELINE_SEMAPHORE
except ImportError:
    try:
        from prescription_anonymizer.api.server import app, PIPELINE_SEMAPHORE
    except ImportError:
        app = None
        PIPELINE_SEMAPHORE = None

__all__ = ["app", "PIPELINE_SEMAPHORE"]
