"""Optional OpenTelemetry exporter for SessionEvents."""

from __future__ import annotations

from typing import Any

from agent.observability.events import SessionEvent


class OTelExporter:
    """Best-effort OpenTelemetry tracer that no-ops when the SDK is absent."""

    def __init__(self, service_name: str = "agent-harness", endpoint: str | None = None) -> None:
        """Configure an OTLP exporter pointing at ``endpoint``."""
        self.service_name = service_name
        self.endpoint = endpoint
        self._tracer = None
        self._enabled = False
        self._session_span = None
        self._task_spans: dict[str, Any] = {}
        if not endpoint:
            return
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError:
            return
        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        trace.set_tracer_provider(provider)
        self._tracer = trace.get_tracer("agent_harness")
        self._enabled = True

    def on_event(self, evt: SessionEvent) -> None:
        """Forward ``evt`` to OTel, opening / closing spans as appropriate."""
        if not self._enabled or self._tracer is None:
            return
        if evt.type == "session_started":
            self._session_span = self._tracer.start_span("session")
            self._session_span.set_attribute("objective", str(evt.payload.get("objective", "")))
        elif evt.type == "session_ended" and self._session_span is not None:
            self._session_span.set_attribute("outcome", str(evt.payload.get("outcome", "")))
            self._session_span.end()
            self._session_span = None
        elif evt.type == "task_started":
            tid = str(evt.payload.get("task_id", ""))
            span = self._tracer.start_span(f"task:{tid}")
            for k, v in evt.payload.items():
                span.set_attribute(k, str(v))
            self._task_spans[tid] = span
        elif evt.type in {"task_completed", "task_failed"}:
            tid = str(evt.payload.get("task_id", ""))
            span = self._task_spans.pop(tid, None)
            if span is not None:
                span.set_attribute("outcome", evt.type)
                span.end()
        elif evt.type == "llm_call" and self._tracer is not None:
            with self._tracer.start_as_current_span("llm_call") as span:
                for k, v in evt.payload.items():
                    span.set_attribute(k, str(v))
        elif evt.type == "tool_called" and self._tracer is not None:
            with self._tracer.start_as_current_span(f"tool:{evt.payload.get('tool_name', '?')}") as span:
                for k, v in evt.payload.items():
                    span.set_attribute(k, str(v))
