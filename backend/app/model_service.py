from dataclasses import dataclass
from pathlib import Path

import xgboost as xgb

from app.features import FEATURE_LABELS, booking_feature_frame
from app.schemas import BookingAssessmentRequest


ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


@dataclass(frozen=True)
class ModelReason:
    code: str
    label: str
    contribution: float


@dataclass(frozen=True)
class ModelPrediction:
    score: float
    reasons: list[ModelReason]


class ModelStore:
    def __init__(self) -> None:
        self._payment: xgb.XGBClassifier | None = None
        self._inventory: xgb.XGBClassifier | None = None

    @property
    def available(self) -> bool:
        return all(
            path.exists()
            for path in (
                ARTIFACT_DIR / "payment_fraud_model.json",
                ARTIFACT_DIR / "inventory_abuse_model.json",
            )
        )

    @property
    def mode(self) -> str:
        return "hybrid_xgboost_rules" if self.available else "rules_only_untrained"

    def _load(self) -> None:
        if not self.available:
            return
        if self._payment is None:
            self._payment = xgb.XGBClassifier()
            self._payment.load_model(ARTIFACT_DIR / "payment_fraud_model.json")
        if self._inventory is None:
            self._inventory = xgb.XGBClassifier()
            self._inventory.load_model(ARTIFACT_DIR / "inventory_abuse_model.json")

    def predict(self, booking: BookingAssessmentRequest) -> tuple[ModelPrediction, ModelPrediction] | None:
        self._load()
        if self._payment is None or self._inventory is None:
            return None
        frame = booking_feature_frame(booking)
        return self._predict_one(self._payment, frame), self._predict_one(self._inventory, frame)

    @staticmethod
    def _predict_one(model: xgb.XGBClassifier, frame) -> ModelPrediction:
        score = float(model.predict_proba(frame)[0, 1])
        contributions = model.get_booster().predict(xgb.DMatrix(frame), pred_contribs=True)[0][:-1]
        positive = [
            (feature, float(value))
            for feature, value in zip(frame.columns, contributions, strict=True)
            if value > 0
        ]
        positive.sort(key=lambda item: item[1], reverse=True)
        total = sum(value for _, value in positive[:4]) or 1.0
        reasons = [
            ModelReason(
                code=f"model_{feature}",
                label=FEATURE_LABELS[feature],
                contribution=round(value / total, 3),
            )
            for feature, value in positive[:4]
        ]
        return ModelPrediction(score=round(score, 3), reasons=reasons)


model_store = ModelStore()
