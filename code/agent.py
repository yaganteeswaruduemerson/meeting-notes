

class Response:
    def __init__(self, **kwargs):
        self.status_code = 200
        self._data = kwargs
    def json(self):
        return self._data
try:
    from observability.observability_wrapper import (
        trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
    )
    from observability.instrumentation import initialize_tracer
except ImportError:
    from contextlib import contextmanager as _obs_cm, asynccontextmanager as _obs_acm
    def trace_agent(*_a, **_kw):
        def _deco(fn): return fn
        return _deco
    class _ObsHandle:
        output_summary = None
        def capture(self, *a, **kw): pass
    @_obs_acm
    async def trace_step(*_a, **_kw):
        yield _ObsHandle()
    @_obs_cm
    def trace_step_sync(*_a, **_kw):
        yield _ObsHandle()
    def trace_model_call(*_a, **_kw): pass
    def trace_tool_call(*_a, **_kw): pass
    def initialize_tracer(*_a, **_kw): pass

import asyncio as _asyncio

import time as _time
from config import settings as _obs_settings

import logging as _obs_startup_log
from contextlib import asynccontextmanager

_obs_startup_logger = _obs_startup_log.getLogger(__name__)

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {
    'content_safety_enabled': True,
    'runtime_enabled': True,
    'content_safety_severity_threshold': 3,
    'check_toxicity': True,
    'check_jailbreak': True,
    'check_pii_input': False,
    'check_credentials_output': True,
    'check_output': True,
    'check_toxic_code_output': True,
    'sanitize_pii': False
}

import logging
import json
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator, field_validator
from config import Config

import openai
import requests

VALIDATION_CONFIG_PATH = Config.VALIDATION_CONFIG_PATH or str(Path(__file__).parent / "validation_config.json")

SYSTEM_PROMPT = (
    "You are a professional Meeting Notes Summarizer Agent. Your role is to process raw meeting transcripts, chat exports, or notes and generate a structured, email-ready summary. Follow these instructions:\n\n"
    "1. Accept input as pasted text, uploaded TXT/DOCX file, or chat export.\n"
    "2. Generate a summary with these sections:\n"
    "   - Meeting Overview\n"
    "   - Key Discussion Points\n"
    "   - Decisions Made\n"
    "   - Action Items (with owner and due date for each; if missing, use Owner TBD or Due Not specified)\n"
    "   - Next Steps\n"
    "3. Extract all action items, assign owners and deadlines if mentioned, and tag each\n"
    "   with priority (High/Medium/Low) based on urgency language.\n"
    "4. Identify all attendees from the transcript.\n"
    "5. Allow the user to select summary length: one-liner, paragraph, or full detailed summary.\n"
    "6. Format the output as a clean, professional email body.\n"
    "7. Support follow-up questions such as what a person agreed to do or what was decided\n"
    "   about a topic by referencing the transcript context.\n"
    "8. Never infer or fabricate action items or decisions not explicitly stated.\n"
    "9. If an action item lacks an owner or deadline, clearly label as Owner TBD or Due Not specified.\n"
    "10. Always ask for user confirmation before sending the summary email to participants.\n"
    "11. Keep summaries concise and use bullet points by default.\n"
    "12. Ensure all processing is in-memory only, with no permanent storage or third-party\n"
    "    sharing, and comply with GDPR session handling.\n"
    "13. If information is missing or unclear, prompt the user for clarification.\n\n"
    "Output must be clear, actionable, and suitable for direct email distribution."
)
OUTPUT_FORMAT = (
    "- Structured summary with required sections (Meeting Overview, Key Discussion Points,\n"
    "  Decisions Made, Action Items, Next Steps)\n"
    "- Action items listed with owner, due date, and priority\n"
    "- Attendees listed\n"
    "- Email-ready formatting (subject, greeting, body, closing)\n"
    "- Bullet points preferred for clarity\n"
    "- Option for one-liner, paragraph, or detailed summary"
)
FALLBACK_RESPONSE = (
    "Unable to generate a complete summary due to insufficient or unclear meeting content.\n\n"
    "Please provide additional details or clarify your request."
)

# ==========================
# LLM Output Sanitizer
# ==========================
import re as _re

_FENCE_RE = _re.compile(r"```(?:\w+)?\s*\n(.*?)```", _re.DOTALL)
_LONE_FENCE_START_RE = _re.compile(r"^```\w*$")
_WRAPPER_RE = _re.compile(
    r"^(?:"
    r"Here(?:'s| is)(?: the)? (?:the |your |a )?(?:code|solution|implementation|result|explanation|answer)[^:]*:\s*"
    r"|Sure[!,.]?\s*"
    r"|Certainly[!,.]?\s*"
    r"|Below is [^:]*:\s*"
    r")",
    _re.IGNORECASE,
)
_SIGNOFF_RE = _re.compile(
    r"^(?:Let me know|Feel free|Hope this|This code|Note:|Happy coding|If you)",
    _re.IGNORECASE,
)
_BLANK_COLLAPSE_RE = _re.compile(r"\n{3,}")


def _strip_fences(text: str, content_type: str) -> str:
    """Extract content from Markdown code fences."""
    fence_matches = _FENCE_RE.findall(text)
    if fence_matches:
        if content_type == "code":
            return "\n\n".join(block.strip() for block in fence_matches)
        for match in fence_matches:
            fenced_block = _FENCE_RE.search(text)
            if fenced_block:
                text = text[:fenced_block.start()] + match.strip() + text[fenced_block.end():]
        return text
    lines = text.splitlines()
    if lines and _LONE_FENCE_START_RE.match(lines[0].strip()):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _strip_trailing_signoffs(text: str) -> str:
    """Remove conversational sign-off lines from the end of code output."""
    lines = text.splitlines()
    while lines and _SIGNOFF_RE.match(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).rstrip()


@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_llm_output(raw: str, content_type: str = "code") -> str:
    """
    Generic post-processor that cleans common LLM output artefacts.
    Args:
        raw: Raw text returned by the LLM.
        content_type: 'code' | 'text' | 'markdown'.
    Returns:
        Cleaned string ready for validation, formatting, or direct return.
    """
    if not raw:
        return ""
    text = _strip_fences(raw.strip(), content_type)
    text = _WRAPPER_RE.sub("", text, count=1).strip()
    if content_type == "code":
        text = _strip_trailing_signoffs(text)
    return _BLANK_COLLAPSE_RE.sub("\n\n", text).strip()

# ==========================
# Pydantic Models
# ==========================
class MeetingNotesRequest(BaseModel):
    input_type: str = Field(..., description="Type of input: 'text', 'file', or 'chat_export'")
    input_value: str = Field(..., description="Raw transcript text, file content (base64), or chat export text")
    summary_length: Optional[str] = Field("detailed", description="Summary length: one-liner, paragraph, or detailed")
    user_email: Optional[str] = Field(None, description="User's email address")
    participant_emails: Optional[List[str]] = Field(None, description="List of participant email addresses")
    user_consent: Optional[bool] = Field(False, description="User consent to send summary email")

    @field_validator("input_type")
    def validate_input_type(cls, v):
        allowed = {"text", "file", "chat_export"}
        if v not in allowed:
            raise ValueError(f"input_type must be one of {allowed}")
        return v

    @field_validator("input_value")
    def validate_input_value(cls, v):
        if not v or not isinstance(v, str) or len(v.strip()) == 0:
            raise ValueError("input_value cannot be empty")
        if len(v) > 50000:
            raise ValueError("input_value exceeds maximum allowed size (50,000 chars)")
        return v.strip()

    @field_validator("summary_length")
    def validate_summary_length(cls, v):
        allowed = {"one-liner", "paragraph", "detailed"}
        if v not in allowed:
            raise ValueError(f"summary_length must be one of {allowed}")
        return v

    @field_validator("user_email")
    def validate_user_email(cls, v):
        if v is not None and len(v.strip()) == 0:
            raise ValueError("user_email cannot be empty")
        return v

    @field_validator("participant_emails")
    def validate_participant_emails(cls, v):
        if v is not None and not isinstance(v, list):
            raise ValueError("participant_emails must be a list")
        return v

class MeetingNotesResponse(BaseModel):
    success: bool = Field(..., description="Whether the operation succeeded")
    summary: Optional[str] = Field(None, description="Structured meeting summary")
    action_items: Optional[List[Dict[str, Any]]] = Field(None, description="List of action items")
    attendees: Optional[List[str]] = Field(None, description="List of attendees")
    email_status: Optional[str] = Field(None, description="Email sending status")
    error: Optional[str] = Field(None, description="Error message if any")

class FollowUpQueryRequest(BaseModel):
    query_text: str = Field(..., description="Follow-up question about meeting responsibilities or decisions")
    transcript_context: str = Field(..., description="Transcript context for reference")

    @field_validator("query_text")
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def validate_query_text(cls, v):
        if not v or not isinstance(v, str) or len(v.strip()) == 0:
            raise ValueError("query_text cannot be empty")
        if len(v) > 50000:
            raise ValueError("query_text exceeds maximum allowed size (50,000 chars)")
        return v.strip()

    @field_validator("transcript_context")
    def validate_transcript_context(cls, v):
        if not v or not isinstance(v, str) or len(v.strip()) == 0:
            raise ValueError("transcript_context cannot be empty")
        if len(v) > 50000:
            raise ValueError("transcript_context exceeds maximum allowed size (50,000 chars)")
        return v.strip()

class FollowUpQueryResponse(BaseModel):
    success: bool = Field(..., description="Whether the operation succeeded")
    answer: Optional[str] = Field(None, description="LLM answer to follow-up question")
    error: Optional[str] = Field(None, description="Error message if any")

# ==========================
# Service Classes
# ==========================
class InputHandler:
    """Service for accepting and normalizing input (text/file/chat export)."""

    def receive_input(self, input_type: str, input_value: str) -> str:
        """Accepts and parses user input."""
        try:
            if input_type == "text":
                return input_value.strip()
            elif input_type == "file":
                # For demo: assume input_value is plain text (not base64/file parsing)
                return input_value.strip()
            elif input_type == "chat_export":
                return input_value.strip()
            else:
                raise ValueError("Unsupported input_type")
        except Exception as e:
            logging.warning(f"InputHandler.receive_input error: {e}")
            return ""

class Preprocessor:
    """Service for cleaning and standardizing transcript text."""

    def normalize_text(self, raw_text: str) -> str:
        """Cleans and standardizes transcript text."""
        try:
            text = raw_text.replace("\r\n", "\n").replace("\t", " ")
            text = _re.sub(r"\s{2,}", " ", text)
            text = text.strip()
            return text
        except Exception as e:
            logging.warning(f"Preprocessor.normalize_text error: {e}")
            return raw_text

class LLMService:
    """Service for LLM summarization and extraction."""

    def __init__(self):
        self._client = None

    @with_content_safety(config=GUARDRAILS_CONFIG)
    def get_llm_client(self):
        """Lazy Azure OpenAI client initialization."""
        api_key = Config.AZURE_OPENAI_API_KEY
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY not configured")
        if self._client is None:
            self._client = openai.AsyncAzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            )
        return self._client

    async def generate_summary(self, transcript_text: str, summary_length: str) -> str:
        """Calls LLM to generate structured summary."""
        system_message = SYSTEM_PROMPT + "\n\nOutput Format: " + OUTPUT_FORMAT
        user_message = (
            f"Transcript:\n{transcript_text}\n\n"
            f"Summary length: {summary_length}\n"
            "Please generate the structured summary as per instructions."
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        _t0 = _time.time()
        try:
            async with trace_step(
                "generate_summary",
                step_type="llm_call",
                decision_summary="Generate structured meeting summary",
                output_fn=lambda r: f"summary={r[:100]}",
            ) as step:
                response = await self.get_llm_client().chat.completions.create(
                    model=Config.LLM_MODEL or "gpt-4.1",
                    messages=messages,
                    **Config.get_llm_kwargs()
                )
                content = response.choices[0].message.content
                step.capture(content)
                try:
                    trace_model_call(
                        provider="azure",
                        model_name=Config.LLM_MODEL or "gpt-4.1",
                        prompt_tokens=getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
                        completion_tokens=getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0,
                        latency_ms=int((_time.time() - _t0) * 1000),
                        response_summary=content[:200] if content else "",
                    )
                except Exception:
                    pass
                return sanitize_llm_output(content, content_type="text")
        except Exception as e:
            logging.warning(f"LLMService.generate_summary error: {e}")
            return FALLBACK_RESPONSE

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def extract_action_items(self, transcript_text: str) -> List[Dict[str, Any]]:
        """Calls LLM to extract action items, owners, deadlines, priorities."""
        system_message = (
            SYSTEM_PROMPT
            + "\n\nOutput Format: List all action items with owner, due date, and priority. "
            "Return as a JSON array of objects: [{\"action\":..., \"owner\":..., \"due_date\":..., \"priority\":...}]"
        )
        user_message = (
            f"Transcript:\n{transcript_text}\n\n"
            "Extract all action items as per instructions."
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        _t0 = _time.time()
        try:
            async with trace_step(
                "extract_action_items",
                step_type="llm_call",
                decision_summary="Extract action items from transcript",
                output_fn=lambda r: f"action_items={str(r)[:100]}",
            ) as step:
                response = await self.get_llm_client().chat.completions.create(
                    model=Config.LLM_MODEL or "gpt-4.1",
                    messages=messages,
                    **Config.get_llm_kwargs()
                )
                content = response.choices[0].message.content
                step.capture(content)
                try:
                    trace_model_call(
                        provider="azure",
                        model_name=Config.LLM_MODEL or "gpt-4.1",
                        prompt_tokens=getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
                        completion_tokens=getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0,
                        latency_ms=int((_time.time() - _t0) * 1000),
                        response_summary=content[:200] if content else "",
                    )
                except Exception:
                    pass
                json_str = sanitize_llm_output(content, content_type="code")
                try:
                    action_items = json.loads(json_str)
                    if isinstance(action_items, list):
                        return action_items
                    else:
                        return []
                except Exception:
                    logging.warning("LLMService.extract_action_items: JSON parse failed")
                    return []
        except Exception as e:
            logging.warning(f"LLMService.extract_action_items error: {e}")
            return []

    async def answer_follow_up(self, query_text: str, transcript_context: str) -> str:
        """Handles user follow-up questions about responsibilities or decisions."""
        system_message = SYSTEM_PROMPT + "\n\nOutput Format: " + OUTPUT_FORMAT
        user_message = (
            f"Transcript:\n{transcript_context}\n\n"
            f"Follow-up question: {query_text}\n"
            "Please answer referencing the transcript context."
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        _t0 = _time.time()
        try:
            async with trace_step(
                "answer_follow_up",
                step_type="llm_call",
                decision_summary="Answer follow-up question from transcript",
                output_fn=lambda r: f"answer={str(r)[:100]}",
            ) as step:
                response = await self.get_llm_client().chat.completions.create(
                    model=Config.LLM_MODEL or "gpt-4.1",
                    messages=messages,
                    **Config.get_llm_kwargs()
                )
                content = response.choices[0].message.content
                step.capture(content)
                try:
                    trace_model_call(
                        provider="azure",
                        model_name=Config.LLM_MODEL or "gpt-4.1",
                        prompt_tokens=getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
                        completion_tokens=getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0,
                        latency_ms=int((_time.time() - _t0) * 1000),
                        response_summary=content[:200] if content else "",
                    )
                except Exception:
                    pass
                return sanitize_llm_output(content, content_type="text")
        except Exception as e:
            logging.warning(f"LLMService.answer_follow_up error: {e}")
            return FALLBACK_RESPONSE

class SummaryFormatter:
    """Service for formatting summary and action items for email."""

    def format_summary(self, summary: str, action_items: List[Dict[str, Any]], attendees: List[str], summary_length: str) -> str:
        """Formats summary and action items for email."""
        try:
            email_body = f"Meeting Summary ({summary_length.title()}):\n\n"
            email_body += summary.strip() + "\n\n"
            if action_items:
                email_body += "Action Items:\n"
                for item in action_items:
                    action = item.get("action", "")
                    owner = item.get("owner", "TBD")
                    due_date = item.get("due_date", "Not specified")
                    priority = item.get("priority", "Medium")
                    email_body += f"- {action} (Owner: {owner}, Due: {due_date}, Priority: {priority})\n"
            if attendees:
                email_body += "\nAttendees:\n"
                email_body += ", ".join(attendees) + "\n"
            email_body += "\nBest regards,\nMeeting Notes Summarizer Agent"
            return email_body
        except Exception as e:
            logging.warning(f"SummaryFormatter.format_summary error: {e}")
            return summary

class ConsentManager:
    """Service for managing user consent for email distribution."""

    def request_consent(self, user_email: str) -> bool:
        """Prompts user for email distribution consent."""
        # For demo: always return True if user_consent is True
        return True

    def validate_consent(self, user_consent: bool) -> bool:
        """Validates user consent."""
        return bool(user_consent)

class EmailSender:
    """Integration for sending summary email to participants."""

    def send_email(self, email_content: str, participant_emails: List[str], user_email: str) -> bool:
        """Sends summary email to participants."""
        try:
            # For demo: simulate email sending
            logging.info(f"Sending email to: {participant_emails} from {user_email}")
            # In production, integrate with SMTP or email API
            return True
        except Exception as e:
            logging.warning(f"EmailSender.send_email error: {e}")
            return False

class ErrorHandler:
    """Service for centralized error handling."""

    ERROR_MAP = {
        "NO_ACTION_ITEMS_FOUND": "No action items were found in the transcript.",
        "EMAIL_CONSENT_NOT_GIVEN": "Email consent not given. Summary email will not be sent.",
        "GENERIC_ERROR": FALLBACK_RESPONSE,
    }

    def handle_error(self, error_code: str, context: Optional[str] = None) -> str:
        """Handles errors and returns fallback responses."""
        msg = self.ERROR_MAP.get(error_code, FALLBACK_RESPONSE)
        if context:
            msg += f"\nContext: {context}"
        logging.warning(f"ErrorHandler.handle_error: {error_code} - {msg}")
        return msg

    def fallback_response(self) -> str:
        return FALLBACK_RESPONSE

class AuditLogger:
    """Service for logging actions/events for compliance."""

    def log_event(self, event_type: str, event_data: Any) -> None:
        """Logs actions/events for compliance."""
        try:
            logging.info(f"AuditLogger: {event_type} - {event_data}")
        except Exception as e:
            logging.warning(f"AuditLogger.log_event error: {e}")

# ==========================
# Main Agent Class
# ==========================
class MeetingNotesSummarizerAgent:
    """Orchestrator for meeting notes summarization and distribution."""

    def __init__(self):
        self.input_handler = InputHandler()
        self.preprocessor = Preprocessor()
        self.llm_service = LLMService()
        self.summary_formatter = SummaryFormatter()
        self.consent_manager = ConsentManager()
        self.email_sender = EmailSender()
        self.error_handler = ErrorHandler()
        self.audit_logger = AuditLogger()
        self.guardrails_config = GUARDRAILS_CONFIG

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_meeting_notes(
        self,
        input_data: MeetingNotesRequest
    ) -> MeetingNotesResponse:
        """Orchestrates the end-to-end summarization and distribution flow."""
        async with trace_step(
            "process_meeting_notes",
            step_type="process",
            decision_summary="Process meeting notes and distribute summary",
            output_fn=lambda r: f"summary={str(r.summary)[:100]}",
        ) as step:
            try:
                # Input normalization
                transcript = self.input_handler.receive_input(input_data.input_type, input_data.input_value)
                if not transcript:
                    error_msg = self.error_handler.handle_error("GENERIC_ERROR", "Input normalization failed")
                    self.audit_logger.log_event("input_error", error_msg)
                    return MeetingNotesResponse(success=False, error=error_msg)

                normalized_text = self.preprocessor.normalize_text(transcript)
                if not normalized_text:
                    error_msg = self.error_handler.handle_error("GENERIC_ERROR", "Text normalization failed")
                    self.audit_logger.log_event("preprocessing_error", error_msg)
                    return MeetingNotesResponse(success=False, error=error_msg)

                # LLM summary generation
                summary = await self.llm_service.generate_summary(normalized_text, input_data.summary_length)
                summary = sanitize_llm_output(summary, content_type="text")
                if not summary or summary == FALLBACK_RESPONSE:
                    error_msg = self.error_handler.handle_error("GENERIC_ERROR", "Summary generation failed")
                    self.audit_logger.log_event("summary_error", error_msg)
                    return MeetingNotesResponse(success=False, error=error_msg)

                # Action item extraction
                action_items = await self.llm_service.extract_action_items(normalized_text)
                if not action_items:
                    error_msg = self.error_handler.handle_error("NO_ACTION_ITEMS_FOUND")
                    self.audit_logger.log_event("action_item_error", error_msg)

                # Attendee extraction (LLM can be used, but for demo, extract from summary)
                attendees = []
                attendee_match = _re.findall(r"Attendees:\s*(.*)", summary)
                if attendee_match:
                    attendees = [a.strip() for a in attendee_match[0].split(",") if a.strip()]

                # Formatting
                formatted_summary = self.summary_formatter.format_summary(
                    summary, action_items, attendees, input_data.summary_length
                )

                # Consent check
                consent_given = self.consent_manager.validate_consent(input_data.user_consent)
                email_status = None
                if consent_given and input_data.participant_emails and input_data.user_email:
                    email_sent = self.email_sender.send_email(
                        formatted_summary, input_data.participant_emails, input_data.user_email
                    )
                    email_status = "sent" if email_sent else "failed"
                    self.audit_logger.log_event("email_sent", {
                        "status": email_status,
                        "recipients": input_data.participant_emails,
                        "sender": input_data.user_email
                    })
                else:
                    email_status = "not_sent"
                    self.audit_logger.log_event("email_not_sent", {
                        "reason": "Consent not given or missing participant emails/user email"
                    })

                step.capture({
                    "summary": formatted_summary,
                    "action_items": action_items,
                    "attendees": attendees,
                    "email_status": email_status
                })

                return MeetingNotesResponse(
                    success=True,
                    summary=formatted_summary,
                    action_items=action_items,
                    attendees=attendees,
                    email_status=email_status
                )
            except Exception as e:
                error_msg = self.error_handler.handle_error("GENERIC_ERROR", str(e))
                self.audit_logger.log_event("agent_error", error_msg)
                return MeetingNotesResponse(success=False, error=error_msg)

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def answer_follow_up_query(
        self,
        query_text: str,
        transcript_context: str
    ) -> FollowUpQueryResponse:
        """Handles user follow-up questions about responsibilities or decisions."""
        async with trace_step(
            "answer_follow_up_query",
            step_type="llm_call",
            decision_summary="Answer follow-up question",
            output_fn=lambda r: f"answer={str(r)[:100]}",
        ) as step:
            try:
                answer = await self.llm_service.answer_follow_up(query_text, transcript_context)
                answer = sanitize_llm_output(answer, content_type="text")
                step.capture(answer)
                if not answer or answer == FALLBACK_RESPONSE:
                    error_msg = self.error_handler.handle_error("GENERIC_ERROR", "Follow-up answer not found")
                    self.audit_logger.log_event("follow_up_error", error_msg)
                    return FollowUpQueryResponse(success=False, error=error_msg)
                return FollowUpQueryResponse(success=True, answer=answer)
            except Exception as e:
                error_msg = self.error_handler.handle_error("GENERIC_ERROR", str(e))
                self.audit_logger.log_event("agent_error", error_msg)
                return FollowUpQueryResponse(success=False, error=error_msg)

# ==========================
# Observability Lifespan
# ==========================
@asynccontextmanager
async def _obs_lifespan(application):
    """Initialise observability on startup, clean up on shutdown."""
    try:
        _obs_startup_logger.info('')
        _obs_startup_logger.info('========== Agent Configuration Summary ==========')
        _obs_startup_logger.info(f'Environment: {getattr(Config, "ENVIRONMENT", "N/A")}')
        _obs_startup_logger.info(f'Agent: {getattr(Config, "AGENT_NAME", "N/A")}')
        _obs_startup_logger.info(f'Project: {getattr(Config, "PROJECT_NAME", "N/A")}')
        _obs_startup_logger.info(f'LLM Provider: {getattr(Config, "MODEL_PROVIDER", "N/A")}')
        _obs_startup_logger.info(f'LLM Model: {getattr(Config, "LLM_MODEL", "N/A")}')
        _cs_endpoint = getattr(Config, 'AZURE_CONTENT_SAFETY_ENDPOINT', None)
        _cs_key = getattr(Config, 'AZURE_CONTENT_SAFETY_KEY', None)
        if _cs_endpoint and _cs_key:
            _obs_startup_logger.info('Content Safety: Enabled (Azure Content Safety)')
            _obs_startup_logger.info(f'Content Safety Endpoint: {_cs_endpoint}')
        else:
            _obs_startup_logger.info('Content Safety: Not Configured')
        _obs_startup_logger.info('Observability Database: Azure SQL')
        _obs_startup_logger.info(f'Database Server: {getattr(Config, "OBS_AZURE_SQL_SERVER", "N/A")}')
        _obs_startup_logger.info(f'Database Name: {getattr(Config, "OBS_AZURE_SQL_DATABASE", "N/A")}')
        _obs_startup_logger.info('===============================================')
        _obs_startup_logger.info('')
    except Exception as _e:
        _obs_startup_logger.warning('Config summary failed: %s', _e)

    _obs_startup_logger.info('')
    _obs_startup_logger.info('========== Content Safety & Guardrails ==========')
    if GUARDRAILS_CONFIG.get('content_safety_enabled'):
        _obs_startup_logger.info('Content Safety: Enabled')
        _obs_startup_logger.info(f'  - Severity Threshold: {GUARDRAILS_CONFIG.get("content_safety_severity_threshold", "N/A")}')
        _obs_startup_logger.info(f'  - Check Toxicity: {GUARDRAILS_CONFIG.get("check_toxicity", False)}')
        _obs_startup_logger.info(f'  - Check Jailbreak: {GUARDRAILS_CONFIG.get("check_jailbreak", False)}')
        _obs_startup_logger.info(f'  - Check PII Input: {GUARDRAILS_CONFIG.get("check_pii_input", False)}')
        _obs_startup_logger.info(f'  - Check Credentials Output: {GUARDRAILS_CONFIG.get("check_credentials_output", False)}')
    else:
        _obs_startup_logger.info('Content Safety: Disabled')
    _obs_startup_logger.info('===============================================')
    _obs_startup_logger.info('')

    _obs_startup_logger.info('========== Initializing Agent Services ==========')
    # 1. Observability DB schema (imports are inside function — only needed at startup)
    try:
        _t = initialize_tracer()
        if _t is not None:
            _obs_startup_logger.info('✓ Telemetry monitoring enabled')
        else:
            _obs_startup_logger.warning('✗ Telemetry monitoring disabled')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Telemetry monitoring failed to initialize')
    _obs_startup_logger.info('=================================================')
    _obs_startup_logger.info('')
    yield

# ==========================
# FastAPI App
# ==========================
app = FastAPI(lifespan=_obs_lifespan,

    title="Meeting Notes Summarizer Agent",
    description="Automatically processes meeting transcripts or notes, produces a clean structured summary, extracts action items with assigned owners and due dates, identifies key decisions made, and distributes the summary to all participants.",
    version=Config.SERVICE_VERSION if hasattr(Config, "SERVICE_VERSION") else "1.0.0",
    # SYNTAX-FIX: lifespan=_obs_lifespan
)

agent = MeetingNotesSummarizerAgent()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/summarize", response_model=MeetingNotesResponse)
async def summarize_endpoint(req: MeetingNotesRequest):
    """Endpoint for meeting notes summarization and distribution."""
    try:
        result = await agent.process_meeting_notes(req)
        return result
    except Exception as e:
        logging.warning(f"/summarize endpoint error: {e}")
        return MeetingNotesResponse(success=False, error=FALLBACK_RESPONSE)

@app.post("/followup", response_model=FollowUpQueryResponse)
async def followup_endpoint(req: FollowUpQueryRequest):
    """Endpoint for follow-up questions about meeting responsibilities or decisions."""
    try:
        result = await agent.answer_follow_up_query(req.query_text, req.transcript_context)
        return result
    except Exception as e:
        logging.warning(f"/followup endpoint error: {e}")
        return FollowUpQueryResponse(success=False, error=FALLBACK_RESPONSE)

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    """Generic exception handler for malformed JSON and other errors."""
    logging.warning(f"Exception handler: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": f"Malformed request or internal error: {str(exc)}",
            "tips": "Check your JSON formatting, field names, and ensure input text is under 50,000 characters."
        }
    )

# ==========================
# Entrypoint
# ==========================
async def _run_agent():
    """Entrypoint: runs the agent with observability (trace collection only)."""
    try:
        import uvicorn
    except ImportError:
        pass
    # Unified logging config — routes uvicorn, agent, and observability through
    # the same handler so all telemetry appears in a single consistent stream.
    _LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(name)s: %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn":        {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error":  {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            "agent":          {"handlers": ["default"], "level": "INFO", "propagate": False},
            "__main__":       {"handlers": ["default"], "level": "INFO", "propagate": False},
            "observability": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "config": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "azure":   {"handlers": ["default"], "level": "WARNING", "propagate": False},
            "urllib3": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    }

    config = uvicorn.Config(
        "agent:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
        log_config=_LOG_CONFIG,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    _asyncio.run(_run_agent())
# __agent_sanitized_for_testing__