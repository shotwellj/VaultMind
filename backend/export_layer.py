"""
Export and Integration Layer for VaultMind.
Converts VaultMind outputs to professional formats and manages webhooks.
"""

import json
import os
import uuid
import hashlib
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime


class ExportFormat(Enum):
    """Supported export formats."""
    MARKDOWN = "markdown"
    JSON = "json"
    CSV = "csv"
    PLAIN_TEXT = "plain_text"


class EventType(Enum):
    """Webhook event types."""
    NEW_DOCUMENT = "new_document"
    QUERY_COMPLETE = "query_complete"
    ALERT = "alert"
    EXPORT_COMPLETE = "export_complete"


@dataclass
class ExportTemplate:
    """Represents an export template."""
    template_id: str
    name: str
    format: ExportFormat
    content: str
    created_at: str
    updated_at: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class WebhookConfig:
    """Configuration for a webhook endpoint."""
    webhook_id: str
    url: str
    event_types: List[EventType]
    is_active: bool
    created_at: str
    secret_key: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class ExportResult:
    """Result of an export operation."""
    export_id: str
    format: ExportFormat
    content: str
    file_path: Optional[str]
    created_at: str
    source_data_id: str
    metadata: Dict = field(default_factory=dict)


class ExportManager:
    """Manages exports and integrations for VaultMind."""

    def __init__(self, base_path: Optional[str] = None):
        """Initialize export manager."""
        if base_path is None:
            home = os.path.expanduser("~")
            base_path = os.path.join(home, ".vaultmind")

        self.base_path = base_path
        self.templates_path = os.path.join(base_path, "templates")
        self.webhooks_path = os.path.join(base_path, "webhooks")
        self.exports_path = os.path.join(base_path, "exports")
        self.api_keys_path = os.path.join(base_path, "api_keys")

        os.makedirs(self.templates_path, exist_ok=True)
        os.makedirs(self.webhooks_path, exist_ok=True)
        os.makedirs(self.exports_path, exist_ok=True)
        os.makedirs(self.api_keys_path, exist_ok=True)

        self.webhooks: Dict[str, WebhookConfig] = {}
        self._load_webhooks()

    def _load_webhooks(self) -> None:
        """Load webhooks from disk."""
        webhook_files = [f for f in os.listdir(self.webhooks_path) if f.endswith('.json')]

        for filename in webhook_files:
            filepath = os.path.join(self.webhooks_path, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    webhook = WebhookConfig(
                        webhook_id=data['webhook_id'],
                        url=data['url'],
                        event_types=[EventType(et) for et in data['event_types']],
                        is_active=data['is_active'],
                        created_at=data['created_at'],
                        secret_key=data['secret_key'],
                        metadata=data.get('metadata', {})
                    )
                    self.webhooks[webhook.webhook_id] = webhook
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

    def export_chat_to_memo(
        self,
        chat_content: str,
        title: str,
        participants: Optional[List[str]] = None
    ) -> ExportResult:
        """Export chat conversation to structured memo."""
        if participants is None:
            participants = []

        export_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()

        memo_lines = []
        memo_lines.append(f"# MEMO: {title}")
        memo_lines.append("")
        memo_lines.append(f"**Date:** {created_at}")
        memo_lines.append(f"**Participants:** {', '.join(participants) if participants else 'N/A'}")
        memo_lines.append("")
        memo_lines.append("## Content")
        memo_lines.append("")
        memo_lines.append(chat_content)

        content = '\n'.join(memo_lines)

        file_path = os.path.join(
            self.exports_path,
            f"memo_{export_id}.md"
        )

        with open(file_path, 'w') as f:
            f.write(content)

        return ExportResult(
            export_id=export_id,
            format=ExportFormat.MARKDOWN,
            content=content,
            file_path=file_path,
            created_at=created_at,
            source_data_id="chat",
            metadata={"title": title, "participants": participants}
        )

    def export_analysis_to_report(
        self,
        analysis_data: Dict,
        title: str,
        sections: Optional[List[str]] = None
    ) -> ExportResult:
        """Export analysis data to structured report."""
        if sections is None:
            sections = []

        export_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()

        report_lines = []
        report_lines.append(f"# Report: {title}")
        report_lines.append("")
        report_lines.append(f"**Generated:** {created_at}")
        report_lines.append("")

        for section in sections:
            section_title = section.replace('_', ' ').title()
            section_data = analysis_data.get(section, {})

            report_lines.append(f"## {section_title}")
            report_lines.append("")

            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    formatted_key = key.replace('_', ' ').title()
                    report_lines.append(f"**{formatted_key}:** {value}")
            elif isinstance(section_data, list):
                for item in section_data:
                    report_lines.append(f"- {item}")
            else:
                report_lines.append(str(section_data))

            report_lines.append("")

        content = '\n'.join(report_lines)

        file_path = os.path.join(
            self.exports_path,
            f"report_{export_id}.md"
        )

        with open(file_path, 'w') as f:
            f.write(content)

        return ExportResult(
            export_id=export_id,
            format=ExportFormat.MARKDOWN,
            content=content,
            file_path=file_path,
            created_at=created_at,
            source_data_id="analysis",
            metadata={"title": title, "sections": sections}
        )

    def export_research_to_brief(
        self,
        research_findings: Dict,
        title: str,
        executive_summary: Optional[str] = None
    ) -> ExportResult:
        """Export research findings to brief template."""
        export_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()

        brief_lines = []
        brief_lines.append(f"# Research Brief: {title}")
        brief_lines.append("")
        brief_lines.append(f"**Date:** {created_at}")
        brief_lines.append("")

        if executive_summary:
            brief_lines.append("## Executive Summary")
            brief_lines.append("")
            brief_lines.append(executive_summary)
            brief_lines.append("")

        if "key_findings" in research_findings:
            brief_lines.append("## Key Findings")
            brief_lines.append("")
            for finding in research_findings.get("key_findings", []):
                brief_lines.append(f"- {finding}")
            brief_lines.append("")

        if "sources" in research_findings:
            brief_lines.append("## Sources")
            brief_lines.append("")
            for source in research_findings.get("sources", []):
                brief_lines.append(f"- {source}")
            brief_lines.append("")

        if "recommendations" in research_findings:
            brief_lines.append("## Recommendations")
            brief_lines.append("")
            for rec in research_findings.get("recommendations", []):
                brief_lines.append(f"- {rec}")
            brief_lines.append("")

        content = '\n'.join(brief_lines)

        file_path = os.path.join(
            self.exports_path,
            f"brief_{export_id}.md"
        )

        with open(file_path, 'w') as f:
            f.write(content)

        return ExportResult(
            export_id=export_id,
            format=ExportFormat.MARKDOWN,
            content=content,
            file_path=file_path,
            created_at=created_at,
            source_data_id="research",
            metadata={"title": title}
        )

    def register_webhook(
        self,
        url: str,
        event_types: List[EventType]
    ) -> WebhookConfig:
        """Register a new webhook endpoint."""
        webhook_id = str(uuid.uuid4())
        secret_key = hashlib.sha256(
            (webhook_id + str(datetime.utcnow())).encode()
        ).hexdigest()
        created_at = datetime.utcnow().isoformat()

        webhook = WebhookConfig(
            webhook_id=webhook_id,
            url=url,
            event_types=event_types,
            is_active=True,
            created_at=created_at,
            secret_key=secret_key
        )

        webhook_data = {
            'webhook_id': webhook.webhook_id,
            'url': webhook.url,
            'event_types': [et.value for et in webhook.event_types],
            'is_active': webhook.is_active,
            'created_at': webhook.created_at,
            'secret_key': webhook.secret_key,
            'metadata': webhook.metadata
        }

        webhook_file = os.path.join(
            self.webhooks_path,
            f"{webhook_id}.json"
        )

        with open(webhook_file, 'w') as f:
            json.dump(webhook_data, f, indent=2)

        self.webhooks[webhook_id] = webhook

        return webhook

    def fire_webhook(
        self,
        event_type: EventType,
        event_data: Dict
    ) -> Dict[str, bool]:
        """Fire webhook event to all registered endpoints."""
        results = {}

        for webhook_id, webhook in self.webhooks.items():
            if not webhook.is_active:
                continue

            if event_type not in webhook.event_types:
                continue

            payload = {
                'event_type': event_type.value,
                'timestamp': datetime.utcnow().isoformat(),
                'data': event_data
            }

            try:
                import urllib.request
                import json as json_lib

                req = urllib.request.Request(
                    webhook.url,
                    data=json_lib.dumps(payload).encode('utf-8'),
                    headers={'Content-Type': 'application/json'}
                )

                with urllib.request.urlopen(req, timeout=10) as response:
                    results[webhook_id] = response.status == 200
            except Exception:
                results[webhook_id] = False

        return results

    def generate_api_key(self, name: str, description: Optional[str] = None) -> str:
        """Generate a new API key for external integrations."""
        api_key_id = str(uuid.uuid4())
        secret = hashlib.sha256(
            (api_key_id + str(datetime.utcnow())).encode()
        ).hexdigest()

        api_key_data = {
            'api_key_id': api_key_id,
            'name': name,
            'description': description or '',
            'secret': secret,
            'created_at': datetime.utcnow().isoformat(),
            'is_active': True
        }

        api_key_file = os.path.join(
            self.api_keys_path,
            f"{api_key_id}.json"
        )

        with open(api_key_file, 'w') as f:
            json.dump(api_key_data, f, indent=2)

        return secret

    def list_templates(self) -> List[ExportTemplate]:
        """List all available export templates."""
        templates = []

        template_files = [f for f in os.listdir(self.templates_path) if f.endswith('.json')]

        for filename in template_files:
            filepath = os.path.join(self.templates_path, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    template = ExportTemplate(
                        template_id=data['template_id'],
                        name=data['name'],
                        format=ExportFormat(data['format']),
                        content=data['content'],
                        created_at=data['created_at'],
                        updated_at=data['updated_at'],
                        metadata=data.get('metadata', {})
                    )
                    templates.append(template)
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        return templates

    def load_template(self, template_id: str) -> Optional[ExportTemplate]:
        """Load a specific export template."""
        templates = self.list_templates()
        for template in templates:
            if template.template_id == template_id:
                return template
        return None

    def save_template(
        self,
        name: str,
        format_type: ExportFormat,
        content: str,
        metadata: Optional[Dict] = None
    ) -> ExportTemplate:
        """Save a new export template."""
        if metadata is None:
            metadata = {}

        template_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        template = ExportTemplate(
            template_id=template_id,
            name=name,
            format=format_type,
            content=content,
            created_at=now,
            updated_at=now,
            metadata=metadata
        )

        template_data = {
            'template_id': template.template_id,
            'name': template.name,
            'format': template.format.value,
            'content': template.content,
            'created_at': template.created_at,
            'updated_at': template.updated_at,
            'metadata': template.metadata
        }

        template_file = os.path.join(
            self.templates_path,
            f"{template_id}.json"
        )

        with open(template_file, 'w') as f:
            json.dump(template_data, f, indent=2)

        return template

    def export_to_format(
        self,
        data: Dict,
        format_type: ExportFormat,
        title: str
    ) -> str:
        """Export data to specified format."""
        if format_type == ExportFormat.JSON:
            return json.dumps(data, indent=2)

        elif format_type == ExportFormat.CSV:
            import csv
            from io import StringIO

            output = StringIO()
            writer = csv.writer(output)

            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                headers = list(data[0].keys())
                writer.writerow(headers)
                for row in data:
                    writer.writerow([row.get(h, '') for h in headers])

            return output.getvalue()

        elif format_type == ExportFormat.MARKDOWN:
            lines = [f"# {title}", ""]
            for key, value in data.items():
                lines.append(f"## {key}")
                if isinstance(value, dict):
                    for k, v in value.items():
                        lines.append(f"- **{k}:** {v}")
                elif isinstance(value, list):
                    for item in value:
                        lines.append(f"- {item}")
                else:
                    lines.append(str(value))
                lines.append("")
            return '\n'.join(lines)

        else:
            lines = []
            for key, value in data.items():
                lines.append(f"{key.upper()}")
                if isinstance(value, dict):
                    for k, v in value.items():
                        lines.append(f"  {k}: {v}")
                elif isinstance(value, list):
                    for item in value:
                        lines.append(f"  - {item}")
                else:
                    lines.append(f"  {value}")
            return '\n'.join(lines)

    def list_api_keys(self) -> List[Dict]:
        """List all registered API keys."""
        api_keys = []

        key_files = [f for f in os.listdir(self.api_keys_path) if f.endswith('.json')]

        for filename in key_files:
            filepath = os.path.join(self.api_keys_path, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    api_keys.append({
                        'api_key_id': data['api_key_id'],
                        'name': data['name'],
                        'description': data['description'],
                        'created_at': data['created_at'],
                        'is_active': data['is_active']
                    })
            except (json.JSONDecodeError, KeyError):
                pass

        return api_keys


# Module-level convenience wrapper functions
_export_manager_instance = None


def _get_export_manager():
    """Get or create singleton ExportManager instance."""
    global _export_manager_instance
    if _export_manager_instance is None:
        _export_manager_instance = ExportManager()
    return _export_manager_instance


def export_chat_to_memo(messages, title="", author="VaultMind"):
    """Export chat messages to structured memo.

    Args:
        messages: Chat content (string or list)
        title: Memo title
        author: Author name

    Returns:
        dict with export result
    """
    if isinstance(messages, list):
        messages = "\n".join([str(m) for m in messages])
    result = _get_export_manager().export_chat_to_memo(messages, title)
    return {
        'export_id': result.export_id,
        'format': result.format.value,
        'content': result.content,
        'file_path': result.file_path,
        'created_at': result.created_at,
        'metadata': result.metadata
    }


def export_analysis_to_report(analysis, title="", sources=[]):
    """Export analysis data to structured report.

    Args:
        analysis: Analysis data dict
        title: Report title
        sources: List of sources

    Returns:
        dict with export result
    """
    result = _get_export_manager().export_analysis_to_report(analysis, title, sources)
    return {
        'export_id': result.export_id,
        'format': result.format.value,
        'content': result.content,
        'file_path': result.file_path,
        'created_at': result.created_at,
        'metadata': result.metadata
    }


def export_research_to_brief(question, answer, sources=[]):
    """Export research findings to brief.

    Args:
        question: Research question
        answer: Research answer/findings
        sources: List of sources

    Returns:
        dict with export result
    """
    research_findings = {
        'key_findings': [answer] if isinstance(answer, str) else answer,
        'sources': sources
    }
    result = _get_export_manager().export_research_to_brief(research_findings, question)
    return {
        'export_id': result.export_id,
        'format': result.format.value,
        'content': result.content,
        'file_path': result.file_path,
        'created_at': result.created_at,
        'metadata': result.metadata
    }


def list_templates():
    """List all available export templates.

    Returns:
        list of template dicts
    """
    templates = _get_export_manager().list_templates()
    return [
        {
            'template_id': t.template_id,
            'name': t.name,
            'format': t.format.value,
            'created_at': t.created_at,
            'updated_at': t.updated_at
        }
        for t in templates
    ]


def register_webhook(url, events=[], secret=""):
    """Register a webhook endpoint.

    Args:
        url: Webhook URL
        events: List of event types to listen for
        secret: Optional secret key

    Returns:
        dict with webhook config
    """
    event_types = [EventType(e) if isinstance(e, str) else e for e in events]
    webhook = _get_export_manager().register_webhook(url, event_types)
    return {
        'webhook_id': webhook.webhook_id,
        'url': webhook.url,
        'event_types': [e.value for e in webhook.event_types],
        'is_active': webhook.is_active,
        'created_at': webhook.created_at,
        'secret_key': webhook.secret_key
    }


def fire_webhook(event, data):
    """Fire webhook event.

    Args:
        event: Event type string
        data: Event data dict

    Returns:
        dict with results
    """
    event_type = EventType(event) if isinstance(event, str) else event
    results = _get_export_manager().fire_webhook(event_type, data)
    return {
        'fired': len(results),
        'results': results
    }


def generate_api_key(label, permissions):
    """Generate a new API key.

    Args:
        label: Key label/name
        permissions: List of permissions

    Returns:
        dict with API key info
    """
    secret = _get_export_manager().generate_api_key(label)
    return {
        'label': label,
        'secret': secret,
        'permissions': permissions
    }
