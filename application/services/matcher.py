import re
from typing import Optional


class LabelMatcher:
    GOV_WARNING_ANCHORS = [
        "government warning",
        "surgeon general",
        "women should not drink alcoholic beverages during pregnancy",
        "consumption of alcoholic beverages impairs your ability to drive a car or operate machinery",
        "may cause health problems",
    ]

    GOV_WARNING_REQUIRED_GROUPS = {
        "header": [
            "government warning",
        ],
        "authority": [
            "surgeon general",
        ],
        "pregnancy": [
            "women should not drink alcoholic beverages during pregnancy",
            "during pregnancy",
        ],
        "impairment": [
            "impairs your ability to drive a car or operate machinery",
            "operate machinery",
            "drive a car or operate machinery",
        ],
    }

    @staticmethod
    def normalize_text(s: Optional[str]) -> str:
        if not s:
            return ""

        s = s.lower()

        replacements = {
            "waming": "warning",
            "govemment": "government",
            "surgcon": "surgeon",
            "genera!": "general",
            "pregnaney": "pregnancy",
            "bevarages": "beverages",
            "machincry": "machinery",
            "750mi": "750ml",
            "50mi": "50ml",
            "11 5%": "11.5%",
            "11,5%": "11.5%",
            "fl. oz": "fl oz",
            "fl oz.": "fl oz",
            "fluid ounces": "fl oz",
            "fluid ounce": "fl oz",
            "alc by vol.": "alc by vol",
            "alc/ vol": "alc/vol",
            "alc /vol": "alc/vol",
            "alc / vol": "alc/vol",
            "by vol.": "by vol",
            "acar": "a car",
            "bev- erages": "beverages",
        }
        for bad, good in replacements.items():
            s = s.replace(bad, good)

        s = re.sub(r"[^a-z0-9%\s./()\-,:;]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @staticmethod
    def _normalize_numeric_string(qty: str) -> str:
        try:
            n = float(qty)
            if n.is_integer():
                return str(int(n))
            return str(n).rstrip("0").rstrip(".")
        except ValueError:
            return qty.strip()

    @classmethod
    def _normalize_net_contents_candidate(cls, qty: str, unit: str) -> str:
        qty = cls._normalize_numeric_string(qty)
        unit = cls.normalize_text(unit)

        if "millil" in unit or unit == "ml":
            return f"{qty}ml"
        if "liter" in unit or unit == "l":
            return f"{qty}l"
        if unit == "cl":
            return f"{qty}cl"
        return f"{qty}oz"

    @classmethod
    def normalize_expected(cls, field_name: str, value):
        if value is None:
            return None

        text = str(value).strip()

        if field_name == "alcohol_content":
            m = re.search(r"(\d+(?:\.\d+)?)", text)
            return float(m.group(1)) if m else None

        if field_name == "net_contents":
            t = cls.normalize_text(text)
            m = re.search(
                r"(\d+(?:\.\d+)?)\s*(ml|milliliters?|milli?liters?|l|liters?|cl|oz|fl\s*oz)",
                t,
            )
            if not m:
                return t
            return cls._normalize_net_contents_candidate(m.group(1), m.group(2))

        return cls.normalize_text(text)

    @classmethod
    def token_overlap_score(cls, a: str, b: str) -> float:
        """Recall-oriented: fraction of *expected* (a) tokens found in *ocr* (b)."""
        a_tokens = set(cls.normalize_text(a).split())
        b_tokens = set(cls.normalize_text(b).split())

        if not a_tokens or not b_tokens:
            return 0.0

        return len(a_tokens & b_tokens) / len(a_tokens)

    @classmethod
    def match_expected_string(cls, field_name: str, expected_value, normalized_ocr_text: str) -> dict:
        if expected_value is None:
            return {
                "raw_value": None,
                "value": None,
                "normalized_value": None,
                "confidence": None,
                "status": "missing",
            }

        expected_norm = cls.normalize_expected(field_name, expected_value)

        if expected_norm is None:
            return {
                "raw_value": None,
                "value": expected_value,
                "normalized_value": None,
                "confidence": 0.0,
                "status": "missing",
            }

        if field_name == "alcohol_content":
            matches = list(re.finditer(
                r"(\d+(?:\.\d+)?)\s*%\s*(?:alc/?\s*vol|alc\s+by\s+vol|alcohol\s+by\s+volume|by\s+vol(?:ume)?|abv|alcohol)?",
                normalized_ocr_text,
            ))

            for m in matches:
                found = float(m.group(1))
                if abs(found - expected_norm) < 0.01:
                    return {
                        "raw_value": m.group(0),
                        "value": found,
                        "normalized_value": found,
                        "confidence": 0.99,
                        "status": "found",
                    }

            if matches:
                first = matches[0]
                found = float(first.group(1))
                return {
                    "raw_value": first.group(0),
                    "value": found,
                    "normalized_value": found,
                    "confidence": 0.0,
                    "status": "missing",
                }

            return {
                "raw_value": None,
                "value": expected_norm,
                "normalized_value": expected_norm,
                "confidence": 0.0,
                "status": "missing",
            }

        if field_name == "net_contents":
            matches = list(re.finditer(
                r"(\d+(?:\.\d+)?)\s*(ml|milliliters?|milli?liters?|l|liters?|cl|oz|fl\.?\s*oz)",
                normalized_ocr_text,
            ))

            candidates = []
            for m in matches:
                raw_value = m.group(0)
                found = cls._normalize_net_contents_candidate(m.group(1), m.group(2))
                candidates.append((raw_value, found))

            for raw_value, found in candidates:
                if found == expected_norm:
                    return {
                        "raw_value": raw_value,
                        "value": found,
                        "normalized_value": found,
                        "confidence": 0.99,
                        "status": "found",
                    }

            if candidates:
                raw_value, found = candidates[0]
                return {
                    "raw_value": raw_value,
                    "value": found,
                    "normalized_value": found,
                    "confidence": 0.0,
                    "status": "missing",
                }

            return {
                "raw_value": None,
                "value": expected_norm,
                "normalized_value": expected_norm,
                "confidence": 0.0,
                "status": "missing",
            }

        expected_norm_str = str(expected_norm)

        if field_name == "brand_name":
            return cls._match_brand_name(expected_value, expected_norm_str, normalized_ocr_text)

        if expected_norm_str in normalized_ocr_text:
            return {
                "raw_value": expected_value,
                "value": expected_value,
                "normalized_value": expected_norm_str,
                "confidence": 0.99,
                "status": "found",
            }

        overlap = cls.token_overlap_score(expected_norm_str, normalized_ocr_text)
        if overlap >= 0.5:
            return {
                "raw_value": expected_value,
                "value": expected_value,
                "normalized_value": expected_norm_str,
                "confidence": round(overlap, 3),
                "status": "partial",
            }

        return {
            "raw_value": None,
            "value": expected_value,
            "normalized_value": expected_norm_str,
            "confidence": 0.0,
            "status": "missing",
        }

    @classmethod
    def _match_brand_name(cls, expected_value, expected_norm_str: str, normalized_ocr_text: str) -> dict:
        tokens = expected_norm_str.split()

        # Short single-word brands: require word-boundary match to avoid false positives
        if len(tokens) == 1 and len(expected_norm_str) < 5:
            pattern = r'\b' + re.escape(expected_norm_str) + r'\b'
            if re.search(pattern, normalized_ocr_text):
                return {
                    "raw_value": expected_value,
                    "value": expected_value,
                    "normalized_value": expected_norm_str,
                    "confidence": 0.99,
                    "status": "found",
                }
        elif expected_norm_str in normalized_ocr_text:
            return {
                "raw_value": expected_value,
                "value": expected_value,
                "normalized_value": expected_norm_str,
                "confidence": 0.99,
                "status": "found",
            }

        overlap = cls.token_overlap_score(expected_norm_str, normalized_ocr_text)
        if overlap >= 0.8:
            return {
                "raw_value": expected_value,
                "value": expected_value,
                "normalized_value": expected_norm_str,
                "confidence": round(overlap, 3),
                "status": "partial",
            }

        return {
            "raw_value": None,
            "value": expected_value,
            "normalized_value": expected_norm_str,
            "confidence": 0.0,
            "status": "missing",
        }

    @classmethod
    def detect_government_warning(cls, normalized_ocr_text: str) -> dict:
        if not normalized_ocr_text:
            return {
                "status": "missing",
                "confidence": 0.0,
                "matched_groups": {},
                "matched_anchors": [],
            }

        matched_groups = {}
        matched_anchor_phrases = []

        for group_name, phrases in cls.GOV_WARNING_REQUIRED_GROUPS.items():
            best = 0.0
            best_phrase = None

            for phrase in phrases:
                if phrase in normalized_ocr_text:
                    best = 1.0
                    best_phrase = phrase
                    break

                score = cls.token_overlap_score(phrase, normalized_ocr_text)
                if score > best:
                    best = score
                    best_phrase = phrase

            matched_groups[group_name] = {
                "best_phrase": best_phrase,
                "score": round(best, 3),
            }

        for anchor in cls.GOV_WARNING_ANCHORS:
            if anchor in normalized_ocr_text or cls.token_overlap_score(anchor, normalized_ocr_text) >= 0.6:
                matched_anchor_phrases.append(anchor)

        group_scores = [v["score"] for v in matched_groups.values()]
        avg_score = sum(group_scores) / len(group_scores) if group_scores else 0.0

        strong_groups = sum(1 for s in group_scores if s >= 0.75)
        partial_groups = sum(1 for s in group_scores if s >= 0.5)

        if strong_groups >= 3 or (strong_groups >= 2 and len(matched_anchor_phrases) >= 2):
            status = "found"
            confidence = max(0.85, round(avg_score, 3))
        elif partial_groups >= 2:
            status = "partial"
            confidence = round(avg_score, 3)
        else:
            status = "missing"
            confidence = round(avg_score, 3)

        return {
            "status": status,
            "confidence": confidence,
            "matched_groups": matched_groups,
            "matched_anchors": matched_anchor_phrases,
        }

    @classmethod
    def evaluate_text(
        cls,
        raw_ocr_text: str,
        ttb_id: Optional[str] = None,
        brand_name: Optional[str] = None,
        alcohol_content: Optional[str] = None,
        net_contents: Optional[str] = None,
        require_government_warning: bool = True,
    ) -> dict:
        normalized_ocr_text = cls.normalize_text(raw_ocr_text)

        fields = {
            "brand_name": cls.match_expected_string("brand_name", brand_name, normalized_ocr_text),
            "alcohol_content": cls.match_expected_string("alcohol_content", alcohol_content, normalized_ocr_text),
            "net_contents": cls.match_expected_string("net_contents", net_contents, normalized_ocr_text),
        }

        government_warning = cls.detect_government_warning(normalized_ocr_text)

        warnings = []
        for field_name, field in fields.items():
            if field["status"] == "missing":
                warnings.append(f"{field_name}_missing")
            elif field["status"] == "partial":
                warnings.append(f"{field_name}_partial_match")
            elif field["confidence"] is not None and field["confidence"] < 0.75:
                warnings.append(f"{field_name}_low_confidence")

        if require_government_warning:
            if government_warning["status"] == "missing":
                warnings.append("government_warning_missing")
            elif government_warning["status"] == "partial":
                warnings.append("government_warning_partial")

        return {
            "id": ttb_id,
            "raw_ocr_text": raw_ocr_text,
            "normalized_ocr_text": normalized_ocr_text,
            "fields": fields,
            "government_warning": government_warning,
            "warnings": warnings,
        }
