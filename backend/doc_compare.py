"""
Document Comparison Engine for VaultMind.
Compares documents and produces structured diffs with AI commentary.
"""

import difflib
import re
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ChangeType(Enum):
    """Type of change in a section."""
    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"
    UNCHANGED = "unchanged"


@dataclass
class DiffSection:
    """Represents a section in a document diff."""
    section_id: str
    section_title: str
    change_type: ChangeType
    original_text: str
    modified_text: str
    diff_lines: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Complete result of document comparison."""
    document_a_name: str
    document_b_name: str
    sections: List[DiffSection]
    summary: Dict
    risk_summary: List[str]
    similarity_score: float

    def to_dict(self) -> Dict:
        """Convert comparison result to dictionary."""
        return {
            'document_a_name': self.document_a_name,
            'document_b_name': self.document_b_name,
            'sections': [
                {
                    'section_id': s.section_id,
                    'section_title': s.section_title,
                    'change_type': s.change_type.value,
                    'risk_flags': s.risk_flags
                }
                for s in self.sections
            ],
            'summary': self.summary,
            'risk_summary': self.risk_summary,
            'similarity_score': self.similarity_score
        }


class DocumentComparator:
    """Compares two documents and produces structured diffs."""

    def __init__(self):
        """Initialize document comparator."""
        self.matcher = None

    def compare_documents(
        self,
        doc_a: str,
        doc_b: str,
        doc_a_name: str = "Document A",
        doc_b_name: str = "Document B"
    ) -> ComparisonResult:
        """Compare two complete documents and return structured diff."""
        sections_a = self._split_into_sections(doc_a)
        sections_b = self._split_into_sections(doc_b)

        diff_sections = self._compare_sections(sections_a, sections_b)
        summary = self._generate_summary(diff_sections)
        risk_summary = self._extract_risk_summary(diff_sections)
        similarity = self._calculate_similarity(doc_a, doc_b)

        return ComparisonResult(
            document_a_name=doc_a_name,
            document_b_name=doc_b_name,
            sections=diff_sections,
            summary=summary,
            risk_summary=risk_summary,
            similarity_score=similarity
        )

    def _split_into_sections(self, doc: str) -> Dict[str, str]:
        """Split document into sections based on headings."""
        sections = {}
        current_section = "PREAMBLE"
        current_text = []

        lines = doc.split('\n')

        for line in lines:
            if re.match(r'^#+\s+', line):
                if current_section and current_text:
                    sections[current_section] = '\n'.join(current_text).strip()

                current_section = line.strip()
                current_text = []
            else:
                current_text.append(line)

        if current_section and current_text:
            sections[current_section] = '\n'.join(current_text).strip()

        return sections if sections else {"FULL_DOCUMENT": doc}

    def _compare_sections(
        self,
        sections_a: Dict[str, str],
        sections_b: Dict[str, str]
    ) -> List[DiffSection]:
        """Compare two sets of sections and return diff results."""
        diff_sections = []
        all_keys = set(sections_a.keys()) | set(sections_b.keys())

        for idx, key in enumerate(sorted(all_keys)):
            text_a = sections_a.get(key, "")
            text_b = sections_b.get(key, "")

            if text_a == text_b:
                change_type = ChangeType.UNCHANGED
            elif not text_a:
                change_type = ChangeType.ADDED
            elif not text_b:
                change_type = ChangeType.REMOVED
            else:
                change_type = ChangeType.CHANGED

            diff_lines = list(difflib.unified_diff(
                text_a.split('\n'),
                text_b.split('\n'),
                lineterm=''
            ))

            risk_flags = self._detect_risk_clauses(text_a, text_b)

            diff_section = DiffSection(
                section_id=str(idx),
                section_title=key,
                change_type=change_type,
                original_text=text_a,
                modified_text=text_b,
                diff_lines=diff_lines,
                risk_flags=risk_flags
            )

            diff_sections.append(diff_section)

        return diff_sections

    def _detect_risk_clauses(self, text_a: str, text_b: str) -> List[str]:
        """Detect risk-related clauses in text."""
        risk_keywords = [
            r'indemnif',
            r'liability',
            r'waive',
            r'terminate',
            r'force\s+majeure',
            r'breach',
            r'confidential',
            r'exclusive',
            r'non-compete',
            r'non-solicitation',
            r'arbitration',
            r'governing\s+law',
        ]

        flags = []

        for keyword in risk_keywords:
            if re.search(keyword, text_b, re.IGNORECASE) and not \
               re.search(keyword, text_a, re.IGNORECASE):
                flags.append(f"New risk keyword found: {keyword}")
            elif re.search(keyword, text_b, re.IGNORECASE) and \
                 re.search(keyword, text_a, re.IGNORECASE):
                if text_a != text_b:
                    flags.append(f"Modified risk clause: {keyword}")

        return flags

    def compare_clauses(self, text_a: str, text_b: str) -> Dict:
        """Compare individual clauses for contracts."""
        sentences_a = re.split(r'(?<=[.!?])\s+', text_a)
        sentences_b = re.split(r'(?<=[.!?])\s+', text_b)

        matcher = difflib.SequenceMatcher(None, sentences_a, sentences_b)
        matching_blocks = matcher.get_matching_blocks()

        added_clauses = []
        removed_clauses = []
        modified_clauses = []

        prev_j = 0
        for block in matching_blocks:
            if block.i > 0:
                for idx in range(prev_j, block.j):
                    if idx < len(sentences_b):
                        added_clauses.append(sentences_b[idx].strip())

            prev_j = block.j + block.size

        for idx in range(block.i, len(sentences_a)):
            if sentences_a[idx].strip():
                removed_clauses.append(sentences_a[idx].strip())

        return {
            "added_clauses": added_clauses,
            "removed_clauses": removed_clauses,
            "modified_clauses": modified_clauses,
            "total_changes": len(added_clauses) + len(removed_clauses)
        }

    def detect_risk_clauses(self, text: str) -> Dict:
        """Detect risk-related clauses in document text."""
        risk_patterns = {
            "indemnification": r'(?:indemnif|hold\s+harmless)',
            "liability_cap": r'(?:limit.*liability|cap.*liability|maximum.*liability)',
            "termination": r'(?:terminate|termination|cancel)',
            "non_standard_terms": r'(?:unusual|non-standard|custom)',
            "missing_insurance": r'(?:insurance|indemnit)',
            "jurisdiction": r'(?:governing\s+law|jurisdiction)',
        }

        found_risks = {}

        for risk_type, pattern in risk_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            matches_list = [m.group() for m in matches]

            if matches_list:
                found_risks[risk_type] = matches_list

        return found_risks

    def _generate_summary(self, diff_sections: List[DiffSection]) -> Dict:
        """Generate summary statistics of changes."""
        added_count = sum(1 for s in diff_sections if s.change_type == ChangeType.ADDED)
        removed_count = sum(1 for s in diff_sections if s.change_type == ChangeType.REMOVED)
        changed_count = sum(1 for s in diff_sections if s.change_type == ChangeType.CHANGED)
        unchanged_count = sum(1 for s in diff_sections if s.change_type == ChangeType.UNCHANGED)

        return {
            "total_sections": len(diff_sections),
            "added_sections": added_count,
            "removed_sections": removed_count,
            "changed_sections": changed_count,
            "unchanged_sections": unchanged_count,
            "sections_with_changes": added_count + removed_count + changed_count
        }

    def _extract_risk_summary(self, diff_sections: List[DiffSection]) -> List[str]:
        """Extract summary of risk flags from all sections."""
        all_risks = []

        for section in diff_sections:
            all_risks.extend(section.risk_flags)

        return list(set(all_risks))

    def _calculate_similarity(self, doc_a: str, doc_b: str) -> float:
        """Calculate similarity score between documents."""
        matcher = difflib.SequenceMatcher(None, doc_a, doc_b)
        return matcher.ratio()

    def export_comparison_markdown(self, result: ComparisonResult) -> str:
        """Export comparison result as markdown."""
        lines = []
        lines.append(f"# Document Comparison Report")
        lines.append(f"\n## Metadata")
        lines.append(f"- Document A: {result.document_a_name}")
        lines.append(f"- Document B: {result.document_b_name}")
        lines.append(f"- Similarity Score: {result.similarity_score:.1%}")
        lines.append(f"\n## Summary")
        lines.append(f"- Total Sections: {result.summary['total_sections']}")
        lines.append(f"- Added: {result.summary['added_sections']}")
        lines.append(f"- Removed: {result.summary['removed_sections']}")
        lines.append(f"- Changed: {result.summary['changed_sections']}")

        if result.risk_summary:
            lines.append(f"\n## Risk Flags")
            for risk in result.risk_summary:
                lines.append(f"- {risk}")

        lines.append(f"\n## Detailed Changes")

        for section in result.sections:
            lines.append(f"\n### {section.section_title}")
            lines.append(f"Status: {section.change_type.value.upper()}")

            if section.risk_flags:
                lines.append(f"Risks: {', '.join(section.risk_flags)}")

            if section.diff_lines:
                lines.append("```diff")
                lines.extend(section.diff_lines)
                lines.append("```")

        return '\n'.join(lines)

    def generate_diff_summary(self, result: ComparisonResult) -> Dict:
        """Generate human-readable summary of key changes."""
        key_changes = []

        for section in result.sections:
            if section.change_type == ChangeType.ADDED:
                key_changes.append({
                    "type": "added_section",
                    "section": section.section_title,
                    "preview": section.modified_text[:100] + "..."
                })
            elif section.change_type == ChangeType.REMOVED:
                key_changes.append({
                    "type": "removed_section",
                    "section": section.section_title,
                    "preview": section.original_text[:100] + "..."
                })
            elif section.change_type == ChangeType.CHANGED and section.risk_flags:
                key_changes.append({
                    "type": "modified_with_risks",
                    "section": section.section_title,
                    "risks": section.risk_flags
                })

        return {
            "key_changes": key_changes,
            "total_key_changes": len(key_changes),
            "summary": result.summary,
            "risks": result.risk_summary
        }

    def export_comparison_dict(self, result: ComparisonResult) -> Dict:
        """Export comparison as structured dictionary."""
        sections_data = []

        for section in result.sections:
            sections_data.append({
                "section_id": section.section_id,
                "title": section.section_title,
                "change_type": section.change_type.value,
                "original_length": len(section.original_text),
                "modified_length": len(section.modified_text),
                "risk_flags": section.risk_flags,
                "line_diff_count": len(section.diff_lines)
            })

        return {
            "documents": {
                "a": result.document_a_name,
                "b": result.document_b_name
            },
            "similarity_score": result.similarity_score,
            "summary": result.summary,
            "risk_summary": result.risk_summary,
            "sections": sections_data
        }

    def export_comparison_plain_text(self, result: ComparisonResult) -> str:
        """Export comparison as plain text."""
        lines = []
        lines.append("=" * 80)
        lines.append("DOCUMENT COMPARISON REPORT")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Document A: {result.document_a_name}")
        lines.append(f"Document B: {result.document_b_name}")
        lines.append(f"Similarity: {result.similarity_score:.1%}")
        lines.append("")
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Total Sections: {result.summary['total_sections']}")
        lines.append(f"Added: {result.summary['added_sections']}")
        lines.append(f"Removed: {result.summary['removed_sections']}")
        lines.append(f"Changed: {result.summary['changed_sections']}")
        lines.append("")

        if result.risk_summary:
            lines.append("RISK FLAGS")
            lines.append("-" * 40)
            for risk in result.risk_summary:
                lines.append(f"  - {risk}")
            lines.append("")

        lines.append("DETAILED CHANGES")
        lines.append("-" * 40)

        for section in result.sections:
            lines.append(f"\n[{section.change_type.value.upper()}] {section.section_title}")

            if section.risk_flags:
                for flag in section.risk_flags:
                    lines.append(f"  RISK: {flag}")

        return '\n'.join(lines)


# Module-level convenience wrapper functions
_doc_comparator_instance = None


def _get_comparator():
    """Get or create singleton DocumentComparator instance."""
    global _doc_comparator_instance
    if _doc_comparator_instance is None:
        _doc_comparator_instance = DocumentComparator()
    return _doc_comparator_instance


def compare_documents(text_a, text_b, label_a="Document A", label_b="Document B"):
    """Compare two documents and return comparison result.

    Args:
        text_a: First document text
        text_b: Second document text
        label_a: Label for first document
        label_b: Label for second document

    Returns:
        ComparisonResult with comparison data
    """
    return _get_comparator().compare_documents(text_a, text_b, label_a, label_b)


def export_comparison_markdown(result):
    """Export comparison result as markdown string.

    Args:
        result: ComparisonResult object

    Returns:
        Markdown-formatted string
    """
    return _get_comparator().export_comparison_markdown(result)
