# routes/tts_routes.py
"""
TTS API routes — multi-provider (local Kokoro, API endpoint, browser).
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel
import logging

from src.auth_helpers import effective_user

logger = logging.getLogger(__name__)

class TTSRequest(BaseModel):
    text: str
    format: str = "audio"  # "audio" or "base64"
    session_id: str | None = None


def _resolve_effective_voice(db, session_id: str | None, owner: str | None) -> str | None:
    """Return the tts_voice of the persona bound to session_id, or None if
    there's no session_id/owner, no binding, a dangling binding, or an
    empty override -- fail-open, never raises. Caller owns closing db."""
    if not session_id or not owner:
        return None
    try:
        from src.crew_helpers import resolve_crew_binding
        crew = resolve_crew_binding(db, session_id, owner)
        return crew.tts_voice if crew and crew.tts_voice else None
    except Exception:
        return None


def setup_tts_routes(tts_service):
    """Setup TTS routes with the provided TTS service"""
    router = APIRouter(prefix="/api/tts", tags=["tts"])

    @router.get("/stats")
    async def get_tts_stats(request: Request, session_id: str | None = None):
        """Get TTS service statistics"""
        try:
            voice_override = None
            if session_id:
                from core.database import SessionLocal
                owner = effective_user(request)
                db = SessionLocal()
                try:
                    voice_override = _resolve_effective_voice(db, session_id, owner)
                finally:
                    db.close()
            return tts_service.get_stats(voice_override=voice_override)
        except Exception as e:
            logger.error(f"Failed to get TTS stats: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/synthesize")
    async def synthesize_speech(request_body: TTSRequest, request: Request):
        """Synthesize speech from text"""
        try:
            if not tts_service.available:
                raise HTTPException(
                    status_code=503,
                    detail={"message": "TTS service not available"}
                )

            voice_override = None
            if request_body.session_id:
                from core.database import SessionLocal
                owner = effective_user(request)
                db = SessionLocal()
                try:
                    voice_override = _resolve_effective_voice(db, request_body.session_id, owner)
                finally:
                    db.close()

            if request_body.format == "base64":
                audio_b64 = tts_service.synthesize_to_base64(request_body.text)
                if not audio_b64:
                    raise HTTPException(
                        status_code=500,
                        detail={"message": "Synthesis failed"}
                    )
                return {"audio": audio_b64}

            else:  # audio format
                audio_data = tts_service.synthesize(request_body.text, voice_override=voice_override)
                if not audio_data:
                    raise HTTPException(
                        status_code=500,
                        detail={"message": "Synthesis failed"}
                    )
                
                # Detect format from magic bytes (MP3: ID3 tag or sync word ff e0+)
                is_mp3 = audio_data[:3] == b'ID3' or (len(audio_data) >= 2 and audio_data[0] == 0xff and (audio_data[1] & 0xe0) == 0xe0)
                mime = "audio/mpeg" if is_mp3 else "audio/wav"
                return Response(
                    content=audio_data,
                    media_type=mime,
                    headers={
                        "Content-Disposition": "inline; filename=speech.mp3" if "mpeg" in mime else "inline; filename=speech.wav"
                    }
                )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Synthesis error: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail={"message": f"Synthesis failed: {str(e)}"}
            )

    @router.post("/clear-cache")
    async def clear_tts_cache():
        """Clear TTS cache"""
        try:
            tts_service.clear_cache()
            return {"success": True, "message": "Cache cleared"}
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return router
