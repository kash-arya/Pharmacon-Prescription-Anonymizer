"""
Core Entity Definitions and PII Anchor Lexicons for Medical Prescriptions.
"""

import re
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Normalized or pixel bounding box (x0, y0, x1, y1)."""
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    def expand(self, dx_ratio: float = 0.15, dy_ratio: float = 0.15) -> "BoundingBox":
        """Expand bounding box by given ratios for safety margin."""
        w = self.width
        h = self.height
        return BoundingBox(
            x0=max(0.0, self.x0 - w * dx_ratio),
            y0=max(0.0, self.y0 - h * dy_ratio),
            x1=self.x1 + w * dx_ratio,
            y1=self.y1 + h * dy_ratio,
        )


class RedactionTarget(BaseModel):
    """Identified PII entity with spatial coordinates on a specific page."""
    page_number: int
    category: str = Field(description="e.g. PATIENT_NAME, PHONE, UHID, DOCTOR_NAME, ADDRESS")
    raw_text: Optional[str] = None
    bbox: BoundingBox
    confidence: float = 1.0
    engine: str = Field(default="vector_rule", description="Detector engine: vector_rule, ocr_anchor, vlm")


# ---------------------------------------------------------------------------
# Multi-lingual & Standard Demographic Anchor Patterns
# ---------------------------------------------------------------------------

# Normalized Label Anchors for Form Key-Value Matching
KEY_VALUE_ANCHORS = {
    "PATIENT_NAME": [
        r"\bPatient\s+Information\b",
        r"\bPatient\s*Name\b",
        r"\bPt(?:\s*Name)?\b",
        r"\bPatient\b(?!\s+(?:ID|Information|Type|should|must|to))",
        r"\b‘?Name\b(?!\s+(?:of\s+(?:Insurance|Company|Hospital|Clinic)))",
        r"\bSurname\b",
        r"\bForename(?:s)?\b",
        r"\bVorname\b",
        r"\bNom\b",
        r"\bPrénom\b",
        r"\bApellidos\b",
        r"\bNombre\b",
        r"\bNombre\s+del\s+Paciente\b",
        r"\bNombre(?:\(s\))?\s+y\s+Apellidos(?:\s+del\s+Paciente)?\b",
        r"\bFamilienname(?:\(n\))?\b",
        r"\bPatient(?:/in)?\b",
        r"\bVersicherte(?:/r)?\b",
        r"\bCaregiver\b",
        r"\bIn\s+Cas[ce]\s+of\s+Emergency\b",
        r"\bEmergency\s+Contact\b",
    ],
    "IDENTIFIER": [
        r"\bUHID(?:\s*No\.?)?\b",
        r"\bOP(?:\s*No\.?)?\b",
        r"\bOPD(?:\s*No\.?)?\b(?!\s*(?:Timings|Hours))",
        r"\bIPD(?:\s*No\.?)?\b",
        r"\bIP(?:\s*No\.?)?\b",
        r"\bMRN\b",
        r"\bPatient\s*ID\b",
        r"\bPractice\s*Code\b",
        r"\bPrescriber\s*No\.?\b",
        r"\bReg(?:istration)?\.?\s*(?:No\.?)?\b",
        r"\bToken(?:\s*No\.?)?\b",
        r"\bPolicy\s*(?:No\.?|Number)?\b",
        r"\bBill\s*No\.?\b",
        r"\bNHS\s*(?:No\.?|Number)?\b",
        r"\bDEA\s*(?:No\.?|Number)?\b",
        r"\bNPI\s*(?:No\.?)?\b",
        r"\bMedicare\s*(?:No\.?)?\b",
        r"\bPBS\s*(?:Prescriber)?(?:\s*No\.?)?\b",
        r"\bAHPRA\s*(?:No\.?)?\b",
        r"\bVersicherten-Nr\.?\b",
        r"\bKrankenkasse\b",
        r"\bKassen-Nr\.?\b",
        r"\bBSNR\b",
        r"\bLANR\b",
        r"\bABHA(?:\s*ID|\s*Number)?\b",
        r"\bRID\s*#?\b",
        r"\bHistoria\s+Cl[ií]nica\b",
        r"\bCarn[eé]\s+de\s+Identidad\b",
        r"\bMitglieds-Nr\.?\b",
        r"\bVersicherungsnummer\b",
        r"\bLicense\s*#?\b",
        r"\bLicence\s*#?\b",
        r"\bKassen-Nr\.?\b",
        r"\bStatus\b",
        r"\bLicence(?:\s*No\.?)?\b",
        r"\bLicense(?:\s*No\.?)?\b",
        r"\bVertragsarztstempel\b",
    ],
    "INSURANCE": [
        r"\bInsurance(?:\s+Provider|\s+Plan|\s+Company)?\b",
        r"\bKrankenkasse(?:\s+bzw\.\s+Kostenträger)?\b",
        r"\bKostenträger\b",
        r"\bEntitlement\b",
        r"\bCRN\b",
    ],
    "DEMOGRAPHIC": [
        r"\bAge(?:\s*/\s*Sex|\s*/\s*Gender)?\b",
        r"\bSex\b",
        r"\bGender\b",
        r"\bDOB\b",
        r"\bD\.?O\.?B\.?\b",
        r"\bDate\s+of\s+Birth\b",
        r"\bGeb(?:urtsdatum)?\.?\s*(?:-Jahr)?\b",
        r"\bFecha\s+de\s+nacimiento\b",
        r"\bDate\s+de\s+naissance\b",
        r"\bPat\.?\s*Type\b",
    ],
    "CONTACT": [
        r"\bMobile(?:\s*No\.?)?\b",
        r"\bPhone(?:\s*No\.?)?\b",
        r"\bContact(?:\s*No\.?)?\b",
        r"\bHome\s+phone\b",
        r"\bWork\s+phone\b",
        r"\bTel(?:\s*No\.?)?\b",
        r"\bTelefon\b",
        r"\bE-?mail\b",
        r"\bFax(?:\s*No\.?)?\b",
        r"\bWeb(?:site)?\b",
        r"\bURL\b",
    ],
    "DOCTOR_INFO": [
        r"\bConsultant\b",
        r"\bDoctor(?:\s+Name)?\b",
        r"\bRef(?:erred)?[.,]?\s*By\b",
        r"\bRadiologist\b",
        r"\bPhysician\b",
        r"\bSurgeon\b",
        r"\bPractitioner\b",
        r"\bCertifying\s+Practitioner\b",
        r"\bTyped\s*By\b",
        r"\bCreated\s*By\b",
        r"\bPrinted\s*By\b",
        r"\bPharmacist(?:'s|’s)?\s*Signature\b",
        r"\bDoctor(?:'s|’s)?\s*Signature\b",
        r"\bPrescriber(?:'s|’s)?\s*Signature\b",
        r"\bPhysician(?:'s|’s)?\s*Signature\b",
        r"\bSignature\s+of\s+Prescriber\b",
        r"\bFor\s+Dispenser\b",
        r"\bArzt-Nr\.?\b",
        r"\bBetriebsstätten-Nr\.?\b",
        r"\bFirma\s+(?:del\s+)?médico\b",
        r"\bUnterschrift\s+(?:des\s+)?Arztes\b",
        r"\bGMC\s*(?:No\.?|Number)?\b",
        r"\bM[eé]dico\b",
        r"\bFirma\b",
        r"\bCu[ñn]o\b",
    ],
    "ADDRESS": [
        r"\bAddress\b",
        r"\bAddr\b",
        r"\bPractice\s*(?:/\s*Surgery\s*)?Address\b",
        r"\bSurgery\s+Address\b",
        r"\bCity\b",
        r"\bState\b",
        r"\bPin(?:\s*code)?\b",
        r"\bWohnanschrift\b",
        r"\bDirección\b",
        r"\bAnschrift\b",
        r"\bDirecci[oó]n(?:\s+Particular)?\b",
        r"\bDienstgeber(?:/in)?\b",
        r"\bDienstort\b",
    ],
    "DATE": [
        r"\bDate\b",
        r"\bDispense\s+Date\b",
        r"\bOP\s*Date\b",
        r"\bReg\.?\s*Date\b",
        r"\bReport\s*Date(?:\s*/\s*Time)?\b",
        r"\bPrint\s*Date(?:\s*/\s*Time)?\b",
        r"\bCollected\s*Date(?:\s*/\s*Time)?\b",
        r"\bReceived\s*Date(?:\s*/\s*Time)?\b",
        r"\bVisit\s*Date\b",
        r"\bExpiry\s*Date\b",
        r"\bBirth\s*Date\b",
        r"\bAusstellungsdatum\b",
        r"\bFecha\b",
    ],
}

PATIENT_NAME_ANCHORS = [
    r"\bPatient\s+Name\s*[:\-=~+]",
    r"\bPt(?:\s+Name)?\s*[:\-=~+]",
    r"\b‘?Name\s*[:\-=~+]",
    r"\bPatient\s*[:\-=~+]",
    r"\bPatient\s+Surname\s*[:\-=~+]?\s*[A-Za-zÄÖÜäöüßÁÉÍÓÚÑáéíóúñ\-]+",
    r"\bForename(?:\(s\))?\s*[:\-=~+]?\s*[A-Za-zÄÖÜäöüßÁÉÍÓÚÑáéíóúñ\-]+",
    r"\b(?:Mr|Mrs|Ms|Miss|Master|Baby|Shri|Smt|Herr|Frau|Fr\.|M\.|Mme|Mlle|Monsieur|Madame|Mademoiselle|Sr\.|Sra\.|Srta\.|Señor|Señora|Señorita|Sig\.|Sig\.ra|Signor|Signora|Dott\.|Dott\.ssa)\s+[A-Za-zÄÖÜäöüßÁÉÍÓÚÑáéíóúñ]+(?:\s+[A-Za-zÄÖÜäöüßÁÉÍÓÚÑáéíóúñ]+)*",
]

DEMOGRAPHIC_ANCHORS = [
    r"\bAge(?:\s*/\s*Sex|\s*/\s*Gender)?\s*[:\-=~+]?\s*[\d\s]+(?:\s*(?:Y|Yrs|Years))?(?:\s*/\s*(?:Male|Female|M|F))?",
    r"\bSex\s*[:\-=~+]?\s*(?:Male|Female|M|F)\b",
    r"\bGender\s*[:\-=~+]?\s*(?:Male|Female|M|F)\b",
    r"\bDOB\s*[:\-=~+]?\s*[0-9\-/]+",
    r"\bDate(?:\s+of\s+Birth)?\s*[:\-=~+]?\s*[0-9]{1,2}[/\-.][0-9]{1,2}[/\-.][0-9]{2,4}",
    r"\b\d{1,3}\s*(?:Y|Yrs|Years)\s*/\s*(?:Male|Female|M|F)\b",
]

IDENTIFIER_ANCHORS = [
    r"\bUHID(?:\s*No\.?)?\s*[:\-=~+]?\s*[A-Za-z0-9\-/]+",
    r"\bOP(?:\s*No\.?)?\s*[:\-=~+]?\s*[A-Za-z0-9\-/]+",
    r"\bOPD\s*(?:No\.?)?\s*[:\-=~+]?\s*[A-Za-z0-9\-/]+(?!\s*(?:Timings|Hours))",
    r"\bIPD\s*(?:No\.?)?\s*[:\-=~+]?\s*[A-Za-z0-9\-/]+",
    r"\bMRN\s*[:\-=~+]?\s*[A-Za-z0-9\-/]+",
    r"\b(?:Reg\.?\s*No\.?|Registration\s*(?:No\.?)?)\s*[:\-=~+]?\s*[A-Za-z0-9\-/]+",
    r"\bPatient\s*ID\s*[:\-=~+]?\s*[A-Za-z0-9\-/]+",
    r"\bToken(?:\s*No\.?)?\s*[:\-=~+]?\s*[A-Za-z0-9\-.\s]+",
    r"\bPolicy\s*(?:No\.?|Number)?\s*[:\-=~+]?\s*[A-Za-z0-9\-/]+",
]

CONTACT_ANCHORS = [
    r"\b(?:Phone|Mobile|Contact|Tel)(?:\s*No\.?)?\s*[:\-=~+]?\s*\+?[0-9\s\-()]{7,20}",
    r"\b(?:Tel(?:efon)?|Phone|Mobile|Kontakt)\.?\s*[:\-=~+]?\s*[\d\-\s/()]{5,18}",
    r"\b\d{3}[\-\s.]\d{3}[\-\s.]\d{4}\b",
    r"\b(?:\+?91[\-\s]?)?[6-9]\d{4}\s*\d{5}\b",
    r"\b(?:\+?91[\-\s]?)?[6-9]\d{9}\b",
    r"\b0\d{2,4}[\-\s]?\d{6,8}\b",
    r"\b\(\d{3}\)[\s\-]?\d{3}[\-\s]?\d{4}\b",
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
]

DOCTOR_ANCHORS = [
    r"\b(?:Dr\.|Dr|Prof\.|Prof|Herr\s+Dr\.|Frau\s+Dr\.|Prof\.\s*Dr\.|Dr\.\s*med\.|Dott\.|Dott\.ssa)\s+(?:med\.\s+)?[A-Za-zÄÖÜäöüßÁÉÍÓÚÑáéíóúñ]+(?:\s+[A-ZÄÖÜÁÉÍÓÚÑ]\b\.?)?(?:\s+[A-Za-zÄÖÜäöüßÁÉÍÓÚÑáéíóúñ]+)*(?:\s*\([A-Za-z\s,.]+\))?",
    r"\bConsultant\s*[:\-=~+]?\s*(?:Dr\.|Dr)?\s*[A-Za-zÄÖÜäöüßÁÉÍÓÚÑáéíóúñ]+(?:\s+[A-ZÄÖÜÁÉÍÓÚÑ]\b\.?)?(?:\s+[A-Za-zÄÖÜäöüßÁÉÍÓÚÑáéíóúñ]+)*",
    r"\bRef(?:erred)?[.,]?\s*By\s*[:\-=~+]?\s*[A-Za-z0-9\s.]+",
    r"\bTyped\s*By\s*[:\-=~+]?\s*[A-Za-z0-9\s.]+",
    r"\bCreated\s*By\s*[:\-=~+]?\s*[A-Za-z0-9\s.]+",
    r"\bPrinted\s*By\s*[:\-=~+]?\s*[A-Za-z0-9\s.]+",
    r"\b(?:MBBS|M\.B\.B\.S|MD|M\.D|MS|M\.S|DMRD|D\.M\.R\.D|DNB|D\.N\.B|FRCR|FRCS|FRACP)\b",
]

# Statutory National Identifiers (US, UK, AU, EU, IN)
STATUTORY_ID_PATTERNS = [
    # US Identifiers
    (r"\bDEA(?:\s*(?:REG|Reg|Registration)\.?)?(?:\s*(?:No|Number)\.?)?\s*[:\-=~+]?\s*([A-Z]{2}[0-9]{7})\b", "IDENTIFIER_DEA"),
    (r"\bNPI(?:\s*(?:No|Number)\.?)?\s*[:\-=~+]?\s*(\d{10})\b", "IDENTIFIER_NPI"),
    (r"\b(?:SSN|Social\s*Security)\s*[:\-=~+]?\s*(\d{3}-\d{2}-\d{4})\b", "IDENTIFIER_SSN"),
    (r"\b(?:(?:State|[A-Z]{2})?\s*(?:Medical\s*)?Lic(?:en[cs]e)?\.?\s*(?:No\.?)?)\s*[:\-=~+]?\s*([A-Z0-9]{2,4}[-\s]?[A-Z0-9]{4,10})\b", "IDENTIFIER_US_STATE_LIC"),
    # UK Identifiers
    (r"\bNHS(?:\s*(?:No|Number)\.?)?\s*[:\-=~+]?\s*(\d{3}\s*\d{3}\s*\d{4}|\d{10})\b", "IDENTIFIER_NHS"),
    (r"\bGMC(?:\s*(?:No|Number)\.?)?\s*[:\-=~+]?\s*(\d{7})\b", "IDENTIFIER_GMC"),
    # Australia Identifiers
    (r"\bMedicare(?:\s*(?:No|Number)\.?)?\s*[:\-=~+]?\s*(\d{4}\s*\d{5}\s*\d{1}|\d{10}(?:\s*\d)?)\b", "IDENTIFIER_MEDICARE"),
    (r"\b(?:PBS\s+Prescriber|PBS|Prescriber)(?:\s*(?:No|Number)\.?)?\s*[:\-=~+]?\s*(\d{6,7}[A-Za-z]?)\b", "IDENTIFIER_PBS"),
    (r"\bAHPRA(?:\s*(?:REG|Reg|Registration)\.?)?(?:\s*(?:No|Number)\.?)?\s*[:\-=~+]?\s*(MED\d{10})\b", "IDENTIFIER_AHPRA"),
    # EU / German Identifiers
    (r"\bVersicherten-Nr\.?\s*[:\-=~+.]?\s*([A-Z0-9]{8,12})\b", "IDENTIFIER_VERSICHERTEN"),
    (r"\b(?:BSNR|Betriebsstätten-Nr\.?)(?:\s*\(BSNR\))?\s*[:\-=~+.]?\s*(\d{7,9})\b", "IDENTIFIER_BSNR"),
    (r"\b(?:LANR|Arzt-Nr\.?)(?:\s*\(LANR\))?\s*[:\-=~+.]?\s*(\d{7,9})\b", "IDENTIFIER_LANR"),
    (r"\b(?:Kassen-Nr\.?|Kostentr[aä]gerkennung|IK-Nr\.?)\s*[:\-=~+.]?\s*(\d{6,9})\b", "IDENTIFIER_KASSEN_NR"),
    # India Identifiers
    (r"\bABHA(?:\s*(?:ID|Number)\.?)?\s*[:\-=~+]?\s*(\d{2}-\d{4}-\d{4}-\d{4})\b", "IDENTIFIER_ABHA"),
    (r"\b(?:MCI|NMC|KMC|DMC|TMC)[-\s]?\d{4,6}\b", "IDENTIFIER_MCI"),
]

ADDRESS_ANCHORS = [
    r"\b(?:Address|Addr|Wohnanschrift|Dirección)\s*[:\-=~+]?\s*[^\n\r]+",
    r"\b(?:Surgery\s+Address|Practice\s+Stamp|Practice\s+Address)\s*[:\-=~+]?\s*[^\n\r]+",
    r"\b(?:SEC|Sector)\s*\d+[\s,]+[A-Za-z]+",
    r"\b\d+[A-Za-z0-9\-/]?[ \t]+[A-Za-z0-9\-.,]+(?:[ \t]+[A-Za-z0-9\-.,]+){0,5}[ \t]+(?:Street|St|Avenue|Ave|Drive|Dr|Road|Rd|Boulevard|Blvd|Court|Ct|Lane|Ln|Terrace|Colony|Market|Chowk|Way|Square|Sq|Place|Pl|Straße|Str|Weg|Platz|Allee|Gasse|Rue|Boulevard|Bd|Avenue|Av|Calle|Paseo|Via|Piazza)\b",
    r"\b[A-Z][a-zA-Z\s]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?\b",
    r"\b[A-ZÄÖÜÁÉÍÓÚÑ][a-zäöüßáéíóúñ]+(?:straße|str\.|strasse|weg|platz|allee|gasse)\s+\d+[a-zA-Z]?\b",
    r"\b(?:Rue|Boulevard|Bd|Avenue|Av\.|Calle|Paseo|Via|Piazza)\s+(?:de\s+|del\s+|della\s+|di\s+)?[A-ZÄÖÜÁÉÍÓÚÑa-zäöüßáéíóúñ\s\-]+(?:\d+[a-zA-Z]?)?\b",
    r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b",
    r"\b(?:NSW|VIC|QLD|WA|SA|TAS|ACT|NT)\s+\d{4}\b",
    r"\b\d{5}\s+[A-ZÄÖÜ][a-zäöüß]+\b",
    r"\b\d{5}(?:-\d{4})?\b",
    r"\b(?:PIN|PIN\s*Code|Postal\s*Code)?\s*[:\-=~+]?\s*(?:110\d{3}|[1-9]\d{2}\s*\d{3})\b",
    r"\b(?:City|State|Country|District|Town|Location)\s*[:\-=~+]?\s*[A-Za-z\s]+",
]

DATE_ANCHORS = [
    r"\b\d{1,2}[/\-.][0-9]{1,2}[/\-.][0-9]{2,4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s*(?:AM|PM))?)?",
    r"\b\d{1,2}\.\s*\d{1,2}\.\s*\d{2,4}\b",
    r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Januar|Februar|März|Mai|Juni|Juli|Oktober|Dezember|Enero|Febrero|Marzo|Abril|Mayo|Junio|Julio|Agosto|Septiembre|Octubre|Noviembre|Diciembre|Janvier|Février|Mars|Avril|Mai|Juin|Juillet|Août|Septembre|Octobre|Novembre|Décembre)[a-z]*[\s\-_]\d{2,4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s*(?:AM|PM))?)?",
    r"\b\d{1,2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Januar|Februar|März|Mai|Juni|Juli|Oktober|Dezember|Enero|Febrero|Marzo|Abril|Mayo|Junio|Julio|Agosto|Septiembre|Octubre|Noviembre|Diciembre|Janvier|Février|Mars|Avril|Mai|Juin|Juillet|Août|Septembre|Octobre|Novembre|Décembre)[a-z]*-\d{2,4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s*(?:AM|PM))?)?",
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Januar|Februar|März|Mai|Juni|Juli|Oktober|Dezember|Enero|Febrero|Marzo|Abril|Mayo|Junio|Julio|Agosto|Septiembre|Octubre|Noviembre|Diciembre|Janvier|Février|Mars|Avril|Mai|Juin|Juillet|Août|Septembre|Octobre|Novembre|Décembre)[a-z]*\s+\d{1,2},?\s+\d{4}\b",
]

CLINICAL_WHITELIST_PATTERNS = [
    # German/EU/US clinical & prescription terms
    r"\b(?:Medikament|Medikamente|Tabl\.?|Tablette|Tabletten|Kapsel|Kapseln|tägl\.?|täglich|Gültig\s+bis|Gebührenfrei|Aut\s+idem|Disp\.?|Sig\.?|Generic|gt|gtt|drops|eye\s+infection|taper|SHAKE|Acne|Supply|Quantity|Pack|Endorsement|Directions|Take|Inhal|Expiration|Opening|Qty|Dispensed|Rp\.?)\b",
    r"\b\d+(?:\.\d+)?\s*(?:g|mg|ml|mcg)\b",
    # General morphological medical / pharmacology suffixes (covers thousands of FDA/EMA drugs)
    r"\b[A-Za-z]{3,}(?:cillin|mycin|prazole|sartan|dipine|statin|olol|pril|zole|mab|nib|tide|profen|parin|glitazone|gliptin|floxacin|cycline|zodone|pam|lam|sone|olone|vir|triptan|setron|caine|dronate|mide|zide)\b",
    # Specific clinical drug substances & reference agents
    r"\b(?:Prednisone|Lialda|Omeprazole|Telmisartan|Amlodipine|Paracetamol|Triptans|Ibuprofen|Metformin|Amoxicillin|Azithromycin|Pantoprazole|Cefixime|Lipitor|Atorvastatin|Lisinopril|Salbutamol|Esomeprazole|Metoprolol|Metoprolol\s+Tartrate|Ramipril|Pantoprazol|Glimepiride|Rosuvastatin)\b",
    # Dosage units and pharmaceutical forms
    r"\b(?:mg|mcg|micrograms|ml|gm|gram|tablet|tablets|tab|tabs|capsule|capsules|cap|caps|syrup|syp|suspension|drops|injection|inj|ointment|gel|cream|patch|inhaler)\b",
    # Administration instructions and clinical frequencies
    r"\b(?:Refill|Refil|daily|once\s+daily|twice\s+daily|thrice\s+daily|SOS|stat|PRN|meals|breakfast|bedtime|hs|tid|bid|od|qid|three\s+months|for\s+three\s+months|tablets\s+daily|Dispense\s+as\s+Written|DAW)\b",
    r"\b(?:Rx|Rp\.?|Prescription|Verordnungen|Advice|Treatment|ADVICE\s*/\s*TREATMENT|Directions|Instructions|INVESTIGATIONS|General\s+Local/\s*Systemic\s+Examination)\b",
    # Clinical Departments & Hospital Specialties
    r"\b(?:Department|GENERAL\s+SURGERY|General\s+Surgery|Surgery|Internal\s+Medicine|Orthopedics|Radiology|Pediatrics|Cardiology)\b",
    # Clinical Diagnoses, Vitals & Findings
    r"\b(?:Chief\s+Complaints|DIAGNOSIS|Diagnosis|VITALS|Vitals|BP|Temp|sO2|SpO2|Pulse|Heart\s+Rate|Respiratory\s+Rate|Weight|Height|BMI)\b",
    r"\b(?:Hypertension|Essential\s+Hypertension|Blood\s+Pressure|Blood\s+Group|Diabetes|Fever|Cough|Pain|Headache|Migraine|Allergy|Allergies)\b",
    r"\b(?:Varicella|Chicken\s+Pox|Measles|IMMUNE|Hepatitis|Peanuts|Medical\s+Problems|General\s+Medical\s+History|Medical\s+History|medication\s+taken)\b",
    r"\b(?:Do\s+you\s+have\s+medical\s+insurance\?|Medical\s+Insurance\s+Details|Normal|mm\s+of\s+Hg|°F|%)\b",
    r"\b(?:Lifestyle\s+Modification|Dietary\s+Advice|disease\s+process|plan\s+of\s+care)\b",
    # Diagnostic Imaging / Ultrasound Anatomy & Findings
    r"\b(?:ULTRASOUND|WHOLE\s*ABDOMEN|HOLE\s*ABDOMEN|Ultrasound|Abdomen|USG|color\s+doppler|Samsung\s+RS80|transducer|cine\s+scans|elastography)\b",
    r"\b(?:Liver|Gall\s*bladder|Gall|Bladder|CBD|Pancreas|Kidney|Right\s+kidney|Left\s+kidney|Prostate|Spleen|Urinary\s+bladder|Aorta|IVC|peritoneal\s+cavity|Appendix|Bile|Biliary|Duct|Ducts|Lumen|Caliber)\b",
    r"\b(?:echotexture|echogenecity|echogenicity|echogenic|dilatation|calculus|calculi|hydronephrosis|focal\s+lesion|space-occupying\s+lesion|free\s+fluid|calyx|Calyx|calices|pelvis|ureter)\b",
    r"\b(?:normal\s+in\s+size|normal\s+in\s+course|distended|wall\s+thickness|upper\s+limit\s+normal|differentiation\s+is\s+maintained|normal|norma)\b",
    r"\b(?:lymph\s+node|enlarged|mass|portal\s+vein|peritoneal|parenchyma|parenchymal|cortical|medullary|cortico-medullary|corticomedullary|renal\s+non\s+obstructive\s+calculi|calculus\s*/\s*mass)\b",
    r"\b(?:Size|Dimension|Dimensions|Diameter|Volume|Thickness|Measurement|Measurements|IHBR|E/O)\b",
    # German/EU/international clinical formulation and dosage words
    r"\b(?:Filmtabletten|Filmtablette|magensaftresistente|Dosierung|unzerkaut|morgens|abends|mittags|Frühstück|Essen|Stück|Packung|N1|N2|N3)\b",
    # General medical condition suffixes and ICD-10 diagnosis codes
    r"\b[A-Za-z]{3,}(?:itis|osis|emia|pathy|algia|rrhea|pnea|trophy|plasia|megaly|sclerosis|stenosis)\b",
    r"\b(?:Acute|Chronic|Essential|Benign|Malignant|Primary|Secondary|Clinical\s+Indications|ICD-10|Indications|Substitutions|Permitted|Refills?|Dispense|oral|capsule|tablet|mouth|management)\b",
    r"\([A-Z]\d{2}(?:\.\d{1,2})?\)",
]

_COMPILED_WHITELIST = [re.compile(pat, re.IGNORECASE) for pat in CLINICAL_WHITELIST_PATTERNS]


def is_whitelisted(text: str) -> bool:
    """Determines if a candidate text string contains protected clinical pharmacology or findings."""
    if not text:
        return False
    clean = text.strip()
    return any(rx.search(clean) for rx in _COMPILED_WHITELIST)


is_clinical_term = is_whitelisted



