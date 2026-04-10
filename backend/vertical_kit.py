"""
Multi-Vertical Adaptation Kit for VaultMind.
Configuration system to adapt VaultMind for different industries.
"""

import json
import os
import re
from typing import Optional, Dict, List
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime


class VerticalType(Enum):
    """Supported industry verticals."""
    LEGAL = "legal"
    ACCOUNTING = "accounting"
    CONSULTING = "consulting"
    MEDICAL = "medical"
    COMPLIANCE = "compliance"
    GENERAL = "general"


@dataclass
class DomainConfig:
    """Domain-specific configuration for a vertical."""
    vertical_type: VerticalType
    terminology: Dict[str, str]
    prompt_templates: Dict[str, str]
    risk_keywords: List[str]
    document_types: List[str]
    citation_format: str
    entity_patterns: Dict[str, str]
    metadata: Dict = field(default_factory=dict)


@dataclass
class VerticalProfile:
    """Complete profile for a vertical configuration."""
    profile_id: str
    name: str
    vertical_type: VerticalType
    domain_config: DomainConfig
    created_at: str
    updated_at: str
    is_active: bool
    custom: bool


class VerticalManager:
    """Manages vertical profiles and domain configurations."""

    def __init__(self, base_path: Optional[str] = None):
        """Initialize vertical manager."""
        if base_path is None:
            home = os.path.expanduser("~")
            base_path = os.path.join(home, ".vaultmind")

        self.base_path = base_path
        self.verticals_path = os.path.join(base_path, "verticals")
        os.makedirs(self.verticals_path, exist_ok=True)

        self.active_profile: Optional[VerticalProfile] = None
        self._init_default_profiles()

    def _init_default_profiles(self) -> None:
        """Initialize default profiles if they don't exist."""
        if not os.path.exists(os.path.join(self.verticals_path, "legal.json")):
            self._create_legal_profile()
        if not os.path.exists(os.path.join(self.verticals_path, "accounting.json")):
            self._create_accounting_profile()
        if not os.path.exists(os.path.join(self.verticals_path, "consulting.json")):
            self._create_consulting_profile()
        if not os.path.exists(os.path.join(self.verticals_path, "medical.json")):
            self._create_medical_profile()
        if not os.path.exists(os.path.join(self.verticals_path, "compliance.json")):
            self._create_compliance_profile()
        if not os.path.exists(os.path.join(self.verticals_path, "general.json")):
            self._create_general_profile()

    def _create_legal_profile(self) -> None:
        """Create legal domain profile."""
        domain_config = DomainConfig(
            vertical_type=VerticalType.LEGAL,
            terminology={
                "contract": "Agreement",
                "clause": "Provision",
                "liability": "Legal Obligation",
                "breach": "Material Breach",
                "indemnity": "Indemnification",
                "force_majeure": "Force Majeure Event"
            },
            prompt_templates={
                "clause_analysis": "Analyze this legal clause for risk factors and obligations",
                "contract_review": "Review this contract for missing provisions and non-standard terms",
                "compliance_check": "Check this document for regulatory compliance requirements"
            },
            risk_keywords=[
                "indemnification",
                "unlimited liability",
                "non-compete",
                "non-solicitation",
                "confidentiality",
                "breach",
                "termination for convenience",
                "force majeure",
                "governing law",
                "arbitration",
                "exclusive remedy",
                "waiver"
            ],
            document_types=[
                "Contract",
                "Lease",
                "NDA",
                "Employment Agreement",
                "Service Agreement",
                "Purchase Agreement",
                "License Agreement",
                "Settlement Agreement"
            ],
            citation_format="[Case Name, Volume Reporter Page (Court Year)]",
            entity_patterns={
                "case_citation": r'(\w+\s+v\.\s+\w+),\s+(\d+)\s+(\w+)\s+(\d+)\s+\((\w+)\s+(\d{4})\)',
                "statute": r'(\d+\s+U\.S\.C\.\s+\d+)',
                "contract_date": r'(?:dated|effective|as of)\s+([A-Za-z]+\s+\d+,\s+\d{4})',
                "party_names": r'(?:between|by and between|this\s+\w+\s+(?:made|entered into)\s+by\s+and\s+between)\s+([^,]+)\s+and\s+([^,]+)'
            }
        )

        profile = VerticalProfile(
            profile_id="legal",
            name="Legal Domain",
            vertical_type=VerticalType.LEGAL,
            domain_config=domain_config,
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            is_active=False,
            custom=False
        )

        self._save_profile(profile)

    def _create_accounting_profile(self) -> None:
        """Create accounting domain profile."""
        domain_config = DomainConfig(
            vertical_type=VerticalType.ACCOUNTING,
            terminology={
                "revenue": "Total Revenue",
                "expense": "Operating Expense",
                "asset": "Asset Account",
                "liability": "Liability Account",
                "equity": "Shareholders' Equity",
                "cash_flow": "Cash Flow Statement"
            },
            prompt_templates={
                "financial_analysis": "Analyze these financial statements for key metrics and trends",
                "audit_assessment": "Review this document for audit findings and recommendations",
                "compliance_review": "Check this document for accounting standards compliance"
            },
            risk_keywords=[
                "impairment",
                "contingent liability",
                "related party transaction",
                "revenue recognition",
                "going concern",
                "subsequent event",
                "restatement",
                "internal control weakness"
            ],
            document_types=[
                "Balance Sheet",
                "Income Statement",
                "Cash Flow Statement",
                "Trial Balance",
                "General Ledger",
                "Audit Report",
                "Tax Return",
                "Financial Statement Notes"
            ],
            citation_format="GAAP / IFRS / IRS Section",
            entity_patterns={
                "account_number": r'\b\d{3}-\d{2}-\d{4}\b',
                "financial_amount": r'\$\s*[\d,]+(?:\.\d{2})?',
                "tax_id": r'(?:EIN|Tax\s+ID):\s*\d{2}-\d{7}',
                "fiscal_period": r'(?:Q[1-4]|FY)\s*\d{4}'
            }
        )

        profile = VerticalProfile(
            profile_id="accounting",
            name="Accounting Domain",
            vertical_type=VerticalType.ACCOUNTING,
            domain_config=domain_config,
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            is_active=False,
            custom=False
        )

        self._save_profile(profile)

    def _create_consulting_profile(self) -> None:
        """Create consulting domain profile."""
        domain_config = DomainConfig(
            vertical_type=VerticalType.CONSULTING,
            terminology={
                "strategy": "Strategic Initiative",
                "recommendation": "Recommended Action",
                "metric": "Key Performance Indicator",
                "benchmark": "Industry Benchmark",
                "gap": "Performance Gap",
                "process": "Business Process"
            },
            prompt_templates={
                "strategy_analysis": "Analyze this strategy document for alignment and viability",
                "recommendations_review": "Review these recommendations for feasibility and impact",
                "benchmark_comparison": "Compare these metrics against industry benchmarks"
            },
            risk_keywords=[
                "execution risk",
                "resource constraint",
                "stakeholder resistance",
                "market volatility",
                "competitive threat",
                "implementation timeline",
                "change management",
                "budget overrun"
            ],
            document_types=[
                "Strategy Document",
                "Business Case",
                "Project Charter",
                "Process Flow",
                "Benchmark Report",
                "Executive Summary",
                "Recommendation Memo",
                "Change Plan"
            ],
            citation_format="Source / Reference / Appendix",
            entity_patterns={
                "metric_value": r'(\d+(?:\.\d+)?)\s*%|(\d+(?:\.\d+)?)\s*(?:K|M|B)',
                "timeframe": r'(?:Q[1-4]|Year)\s+\d{4}|(\d+)\s+(?:months?|quarters?|years?)',
                "department": r'(?:Sales|Marketing|Operations|Finance|HR|IT|R&D)',
                "kpi_label": r'(?:ROI|NPV|IRR|CAGR|CAC|LTV|NPS)'
            }
        )

        profile = VerticalProfile(
            profile_id="consulting",
            name="Consulting Domain",
            vertical_type=VerticalType.CONSULTING,
            domain_config=domain_config,
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            is_active=False,
            custom=False
        )

        self._save_profile(profile)

    def _create_medical_profile(self) -> None:
        """Create medical domain profile."""
        domain_config = DomainConfig(
            vertical_type=VerticalType.MEDICAL,
            terminology={
                "patient": "Patient",
                "diagnosis": "Clinical Diagnosis",
                "treatment": "Treatment Plan",
                "medication": "Pharmaceutical Agent",
                "symptom": "Clinical Symptom",
                "procedure": "Medical Procedure"
            },
            prompt_templates={
                "clinical_review": "Review this clinical document for accuracy and completeness",
                "compliance_check": "Check this document for HIPAA and regulatory compliance",
                "care_assessment": "Assess this care plan for appropriateness and safety"
            },
            risk_keywords=[
                "contraindication",
                "adverse event",
                "drug interaction",
                "allergic reaction",
                "dosage error",
                "patient safety",
                "informed consent",
                "medical malpractice"
            ],
            document_types=[
                "Clinical Notes",
                "Discharge Summary",
                "Medication List",
                "Lab Results",
                "Imaging Report",
                "Operative Report",
                "Pathology Report",
                "Vital Signs Record"
            ],
            citation_format="ICD-10 / CPT / SNOMED CT",
            entity_patterns={
                "icd_code": r'\b[A-Z]\d{2}(?:\.\d{1,2})?\b',
                "cpt_code": r'\b\d{5}[A-Z]?\b',
                "date_of_birth": r'(?:DOB|Date of Birth):\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
                "vital_sign": r'(?:BP|HR|RR|Temp|O2\s+Sat):\s*[\d.]+(?:\s*[a-zA-Z/%]+)?'
            }
        )

        profile = VerticalProfile(
            profile_id="medical",
            name="Medical Domain",
            vertical_type=VerticalType.MEDICAL,
            domain_config=domain_config,
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            is_active=False,
            custom=False
        )

        self._save_profile(profile)

    def _create_compliance_profile(self) -> None:
        """Create compliance domain profile."""
        domain_config = DomainConfig(
            vertical_type=VerticalType.COMPLIANCE,
            terminology={
                "control": "Control Procedure",
                "regulation": "Regulatory Requirement",
                "violation": "Compliance Violation",
                "audit": "Compliance Audit",
                "risk": "Compliance Risk",
                "policy": "Compliance Policy"
            },
            prompt_templates={
                "compliance_assessment": "Assess this document for regulatory compliance",
                "control_evaluation": "Evaluate the effectiveness of these control procedures",
                "risk_identification": "Identify and assess compliance risks in this document"
            },
            risk_keywords=[
                "non-compliance",
                "regulatory violation",
                "audit finding",
                "control deficiency",
                "remediation",
                "enforcement action",
                "sanctions",
                "regulatory change"
            ],
            document_types=[
                "Compliance Policy",
                "Audit Report",
                "Risk Assessment",
                "Control Documentation",
                "Regulatory Notice",
                "Remediation Plan",
                "Training Record",
                "Incident Report"
            ],
            citation_format="Regulation / CFR / Statute",
            entity_patterns={
                "cfr_citation": r'\b(\d{1,2})\s+CFR\s+(\d+(?:\.\d+)*)',
                "regulation": r'(?:Regulation|Rule)\s+[A-Z]{1,3}[-\d.]*',
                "effective_date": r'(?:effective|as of)\s+([A-Za-z]+\s+\d+,\s+\d{4})',
                "deadline": r'(?:deadline|due date):\s*([A-Za-z]+\s+\d+,\s+\d{4})'
            }
        )

        profile = VerticalProfile(
            profile_id="compliance",
            name="Compliance Domain",
            vertical_type=VerticalType.COMPLIANCE,
            domain_config=domain_config,
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            is_active=False,
            custom=False
        )

        self._save_profile(profile)

    def _create_general_profile(self) -> None:
        """Create general domain profile."""
        domain_config = DomainConfig(
            vertical_type=VerticalType.GENERAL,
            terminology={
                "document": "Document",
                "information": "Information",
                "requirement": "Requirement",
                "change": "Change",
                "risk": "Risk",
                "recommendation": "Recommendation"
            },
            prompt_templates={
                "analysis": "Analyze this document",
                "summary": "Summarize the key points in this document",
                "comparison": "Compare these documents"
            },
            risk_keywords=[
                "risk",
                "warning",
                "critical",
                "important",
                "attention",
                "review",
                "verify",
                "confirm"
            ],
            document_types=[
                "Document",
                "Report",
                "Memo",
                "Email",
                "Spreadsheet",
                "Presentation",
                "Form",
                "Other"
            ],
            citation_format="Source / Reference",
            entity_patterns={
                "date": r'\d{1,2}[/-]\d{1,2}[/-]\d{4}',
                "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
                "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                "url": r'https?://[^\s]+'
            }
        )

        profile = VerticalProfile(
            profile_id="general",
            name="General Domain",
            vertical_type=VerticalType.GENERAL,
            domain_config=domain_config,
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            is_active=True,
            custom=False
        )

        self._save_profile(profile)

    def _save_profile(self, profile: VerticalProfile) -> None:
        """Save profile to disk."""
        profile_data = {
            'profile_id': profile.profile_id,
            'name': profile.name,
            'vertical_type': profile.vertical_type.value,
            'domain_config': {
                'vertical_type': profile.domain_config.vertical_type.value,
                'terminology': profile.domain_config.terminology,
                'prompt_templates': profile.domain_config.prompt_templates,
                'risk_keywords': profile.domain_config.risk_keywords,
                'document_types': profile.domain_config.document_types,
                'citation_format': profile.domain_config.citation_format,
                'entity_patterns': profile.domain_config.entity_patterns,
                'metadata': profile.domain_config.metadata
            },
            'created_at': profile.created_at,
            'updated_at': profile.updated_at,
            'is_active': profile.is_active,
            'custom': profile.custom
        }

        filepath = os.path.join(
            self.verticals_path,
            f"{profile.profile_id}.json"
        )

        with open(filepath, 'w') as f:
            json.dump(profile_data, f, indent=2)

    def load_profile(self, profile_id: str) -> Optional[VerticalProfile]:
        """Load a vertical profile by ID."""
        filepath = os.path.join(self.verticals_path, f"{profile_id}.json")

        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            domain_config = DomainConfig(
                vertical_type=VerticalType(data['domain_config']['vertical_type']),
                terminology=data['domain_config']['terminology'],
                prompt_templates=data['domain_config']['prompt_templates'],
                risk_keywords=data['domain_config']['risk_keywords'],
                document_types=data['domain_config']['document_types'],
                citation_format=data['domain_config']['citation_format'],
                entity_patterns=data['domain_config']['entity_patterns'],
                metadata=data['domain_config'].get('metadata', {})
            )

            profile = VerticalProfile(
                profile_id=data['profile_id'],
                name=data['name'],
                vertical_type=VerticalType(data['vertical_type']),
                domain_config=domain_config,
                created_at=data['created_at'],
                updated_at=data['updated_at'],
                is_active=data['is_active'],
                custom=data['custom']
            )

            return profile
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def get_active_profile(self) -> VerticalProfile:
        """Get the currently active profile."""
        if self.active_profile:
            return self.active_profile

        all_profiles = self.list_profiles()
        for profile in all_profiles:
            if profile.is_active:
                self.active_profile = profile
                return profile

        return self.load_profile("general")

    def set_active_profile(self, profile_id: str) -> bool:
        """Set the active profile."""
        profile = self.load_profile(profile_id)

        if not profile:
            return False

        all_profiles = self.list_profiles()
        for p in all_profiles:
            if p.is_active:
                p.is_active = False
                self._save_profile(p)

        profile.is_active = True
        self._save_profile(profile)
        self.active_profile = profile

        return True

    def list_profiles(self) -> List[VerticalProfile]:
        """List all available profiles."""
        profiles = []

        profile_files = [f for f in os.listdir(self.verticals_path) if f.endswith('.json')]

        for filename in profile_files:
            profile_id = filename.replace('.json', '')
            profile = self.load_profile(profile_id)
            if profile:
                profiles.append(profile)

        return profiles

    def create_custom_profile(
        self,
        name: str,
        base_vertical: VerticalType,
        custom_config: Dict
    ) -> VerticalProfile:
        """Create a custom profile based on a base vertical."""
        base_profile = self.load_profile(base_vertical.value)

        if not base_profile:
            raise ValueError(f"Base vertical {base_vertical.value} not found")

        import uuid
        profile_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        domain_config = DomainConfig(
            vertical_type=base_profile.domain_config.vertical_type,
            terminology={**base_profile.domain_config.terminology, **custom_config.get('terminology', {})},
            prompt_templates={**base_profile.domain_config.prompt_templates, **custom_config.get('prompt_templates', {})},
            risk_keywords=custom_config.get('risk_keywords', base_profile.domain_config.risk_keywords),
            document_types=custom_config.get('document_types', base_profile.domain_config.document_types),
            citation_format=custom_config.get('citation_format', base_profile.domain_config.citation_format),
            entity_patterns={**base_profile.domain_config.entity_patterns, **custom_config.get('entity_patterns', {})},
            metadata=custom_config.get('metadata', {})
        )

        profile = VerticalProfile(
            profile_id=profile_id,
            name=name,
            vertical_type=base_vertical,
            domain_config=domain_config,
            created_at=now,
            updated_at=now,
            is_active=False,
            custom=True
        )

        self._save_profile(profile)

        return profile

    def get_domain_entities(self, text: str, profile_id: Optional[str] = None) -> Dict[str, List[str]]:
        """Extract domain-specific entities from text."""
        if not profile_id:
            profile = self.get_active_profile()
        else:
            profile = self.load_profile(profile_id)

        if not profile:
            return {}

        entities = {}

        for entity_type, pattern in profile.domain_config.entity_patterns.items():
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)

            if matches:
                if isinstance(matches[0], tuple):
                    entities[entity_type] = [m[0] if m[0] else m[1] if len(m) > 1 else '' for m in matches if any(m)]
                else:
                    entities[entity_type] = matches

        return entities


# Module-level convenience wrapper functions
_vertical_manager_instance = None


def _get_vertical_manager():
    """Get or create singleton VerticalManager instance."""
    global _vertical_manager_instance
    if _vertical_manager_instance is None:
        _vertical_manager_instance = VerticalManager()
    return _vertical_manager_instance


def load_profile(name):
    """Load a vertical profile by name.

    Args:
        name: Profile name/ID

    Returns:
        dict with profile data or None
    """
    profile = _get_vertical_manager().load_profile(name)
    if profile:
        return {
            'profile_id': profile.profile_id,
            'name': profile.name,
            'vertical_type': profile.vertical_type.value,
            'is_active': profile.is_active,
            'custom': profile.custom,
            'created_at': profile.created_at
        }
    return None


def get_active_profile():
    """Get the currently active vertical profile.

    Returns:
        dict with active profile data
    """
    profile = _get_vertical_manager().get_active_profile()
    return {
        'profile_id': profile.profile_id,
        'name': profile.name,
        'vertical_type': profile.vertical_type.value,
        'is_active': profile.is_active,
        'custom': profile.custom,
        'created_at': profile.created_at
    }


def set_active_profile(name):
    """Set the active vertical profile.

    Args:
        name: Profile name/ID

    Returns:
        None
    """
    _get_vertical_manager().set_active_profile(name)


def list_profiles():
    """List all available vertical profiles.

    Returns:
        list of profile dicts
    """
    profiles = _get_vertical_manager().list_profiles()
    return [
        {
            'profile_id': p.profile_id,
            'name': p.name,
            'vertical_type': p.vertical_type.value,
            'is_active': p.is_active,
            'custom': p.custom,
            'created_at': p.created_at
        }
        for p in profiles
    ]


def create_custom_profile(name, base="general", overrides={}):
    """Create a custom vertical profile.

    Args:
        name: Profile name
        base: Base vertical type
        overrides: Config overrides dict

    Returns:
        dict with new profile data
    """
    base_vertical = VerticalType(base)
    profile = _get_vertical_manager().create_custom_profile(name, base_vertical, overrides)
    return {
        'profile_id': profile.profile_id,
        'name': profile.name,
        'vertical_type': profile.vertical_type.value,
        'is_active': profile.is_active,
        'custom': profile.custom,
        'created_at': profile.created_at
    }
