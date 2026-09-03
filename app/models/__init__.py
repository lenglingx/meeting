"""ORM models."""


def import_models() -> None:
    from app.models.business import (  # noqa: F401
        AnalysisTask,
        Meeting,
        RealtimeTranscriptEvent,
        Speaker,
        TranscriptSegment,
        VoiceprintSample,
    )
