class AdaptiveStrategyEngine:

    def __init__(self, text_profile, image_profile):
        self.text_profile = text_profile
        self.image_profile = image_profile
        self.strategy = {}
        self.confidence = 0.0

        # Ideal quality vector
        self.ideal_vector = [1.0, 1.0, 1.0, 1.0]

    # -----------------------
    # TEXT EVALUATION
    # -----------------------
    def evaluate_text_quality(self):

        C = self.text_profile.get("C", 1.0)
        S = self.text_profile.get("S", 1.0)
        TQ = self.text_profile.get("TQ", 1.0)

        if C < 0.90:
            self.strategy["fill_missing"] = True

        if S < 0.95:
            self.strategy["remove_duplicates"] = True

        if TQ < 0.90:
            self.strategy["advanced_text_cleaning"] = True

    # -----------------------
    # IMAGE EVALUATION
    # -----------------------
    def evaluate_image_quality(self):

        IQ = self.image_profile.get("IQ", 1.0)

        if IQ < 0.80:
            self.strategy["apply_esrgan"] = True

    # -----------------------
    # MATHEMATICAL CONFIDENCE
    # -----------------------
    def compute_confidence(self):

        C = self.text_profile.get("C", 1.0)
        S = self.text_profile.get("S", 1.0)
        TQ = self.text_profile.get("TQ", 1.0)
        IQ = self.image_profile.get("IQ", 1.0)

        actual_vector = [C, S, TQ, IQ]

        deviation = sum(abs(i - j) for i, j in zip(self.ideal_vector, actual_vector)) / 4
        self.confidence = round(1 - deviation, 2)

    # -----------------------
    # MAIN PREDICT
    # -----------------------
    def predict(self):

        self.evaluate_text_quality()
        self.evaluate_image_quality()
        self.compute_confidence()

        if not self.strategy:
            self.strategy["already_clean"] = True

        return self.strategy, self.confidence
