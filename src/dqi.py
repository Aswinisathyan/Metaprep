class DataQualityIndex:
    """
    Computes multi-dimensional Data Quality Index (DQI)
    for multimodal datasets (Text + Image).

    - Automatically adapts to modality presence
    - Dynamically normalizes weights
    - Prevents artificial inflation from missing modalities
    """

    def __init__(self, text_profile: dict, image_profile: dict = None):
        self.text_profile = text_profile or {}
        self.image_profile = image_profile or {}

        # Base weights (used if modality present)
        self.base_weights = {
            "C": 0.30,     # Completeness
            "S": 0.20,     # Consistency
            "TQ": 0.25,    # Text Quality
            "IQ": 0.25     # Image Quality
        }

    # ----------------------------------
    # TEXT SUB-METRIC EXTRACTION
    # ----------------------------------
    def compute_text_quality(self):

        P = self.text_profile.get("punctuation_score", 1.0)
        R = self.text_profile.get("repetition_score", 1.0)
        E = self.text_profile.get("extraneous_char_score", 1.0)
        L = self.text_profile.get("length_stability_score", 1.0)

        TQ = (0.30 * P) + (0.30 * R) + (0.20 * E) + (0.20 * L)

        return round(min(max(TQ, 0), 1), 3)

    # ----------------------------------
    # IMAGE QUALITY
    # ----------------------------------
    def compute_image_quality(self):

        blur_score = self.image_profile.get("blur_score")
        noise_score = self.image_profile.get("noise_score")
        resolution_score = self.image_profile.get("resolution_score")

        # If no real image metrics exist → return None
        if blur_score is None and noise_score is None and resolution_score is None:
            return None

        blur_score = blur_score if blur_score is not None else 1.0
        noise_score = noise_score if noise_score is not None else 1.0
        resolution_score = resolution_score if resolution_score is not None else 1.0

        IQ = (
            0.40 * blur_score +
            0.30 * noise_score +
            0.30 * resolution_score
        )

        return round(min(max(IQ, 0), 1), 3)

    # ----------------------------------
    # FINAL DQI COMPUTATION
    # ----------------------------------
    def compute_dqi(self):

        C = self.text_profile.get("C", 1.0)
        S = self.text_profile.get("S", 1.0)

        TQ = self.compute_text_quality()
        IQ = self.compute_image_quality()

        metrics = {
            "C": C,
            "S": S,
            "TQ": TQ
        }

        # Include IQ only if available
        if IQ is not None:
            metrics["IQ"] = IQ

        # Select active weights
        active_weights = {
            k: self.base_weights[k]
            for k in metrics.keys()
        }

        # Normalize weights dynamically
        weight_sum = sum(active_weights.values())
        normalized_weights = {
            k: v / weight_sum
            for k, v in active_weights.items()
        }

        # Weighted sum
        DQI = sum(
            normalized_weights[k] * metrics[k]
            for k in metrics
        )

        return {
            "C": round(C, 3),
            "S": round(S, 3),
            "TQ": TQ,
            "IQ": round(IQ, 3) if IQ is not None else None,
            "DQI": round(min(max(DQI, 0), 1), 3)
        }

    # ----------------------------------
    # IMPROVEMENT ANALYSIS
    # ----------------------------------
    @staticmethod
    def compare(before_metrics: dict, after_metrics: dict):

        before_dqi = before_metrics.get("DQI", 0)
        after_dqi = after_metrics.get("DQI", 0)

        improvement = round(after_dqi - before_dqi, 3)

        if improvement > 0.05:
            impact = "Significant Improvement"
        elif improvement > 0.01:
            impact = "Moderate Improvement"
        elif improvement > 0:
            impact = "Minor Improvement"
        else:
            impact = "No Improvement"

        return {
            "DQI_before": before_dqi,
            "DQI_after": after_dqi,
            "Improvement": improvement,
            "Impact_Level": impact
        }