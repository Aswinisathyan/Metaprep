# dqi.py

class DataQualityIndex:
    """
    Computes multi-dimensional Data Quality Index (DQI)
    for multimodal datasets (Text + Image).
    """

    def __init__(self, text_profile: dict, image_profile: dict):
        self.text_profile = text_profile
        self.image_profile = image_profile

        # Default weights (can be tuned)
        self.weights = {
            "C": 0.30,     # Completeness
            "S": 0.20,     # Consistency
            "TQ": 0.25,    # Text Quality
            "IQ": 0.25     # Image Quality
        }

    # ----------------------------------
    # TEXT SUB-METRIC EXTRACTION
    # ----------------------------------
    def compute_text_quality(self):

        # Individual sub-scores (expected normalized 0-1)
        P = self.text_profile.get("punctuation_score", 1.0)
        R = self.text_profile.get("repetition_score", 1.0)
        E = self.text_profile.get("extraneous_char_score", 1.0)
        L = self.text_profile.get("length_stability_score", 1.0)

        # Composite Text Quality
        TQ = (0.30 * P) + (0.30 * R) + (0.20 * E) + (0.20 * L)

        return round(min(max(TQ, 0), 1), 3)

    # ----------------------------------
    # IMAGE QUALITY
    # ----------------------------------
    def compute_image_quality(self):

        blur_score = self.image_profile.get("blur_score", 1.0)
        noise_score = self.image_profile.get("noise_score", 1.0)
        resolution_score = self.image_profile.get("resolution_score", 1.0)

        IQ = (0.40 * blur_score) + (0.30 * noise_score) + (0.30 * resolution_score)

        return round(min(max(IQ, 0), 1), 3)

    # ----------------------------------
    # FINAL DQI COMPUTATION
    # ----------------------------------
    def compute_dqi(self):

        C = self.text_profile.get("C", 1.0)
        S = self.text_profile.get("S", 1.0)

        TQ = self.compute_text_quality()
        IQ = self.compute_image_quality()

        DQI = (
            self.weights["C"] * C +
            self.weights["S"] * S +
            self.weights["TQ"] * TQ +
            self.weights["IQ"] * IQ
        )

        return {
            "C": round(C, 3),
            "S": round(S, 3),
            "TQ": TQ,
            "IQ": IQ,
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
