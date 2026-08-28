from dataclasses import dataclass

from app.model_service import model_store
from app.schemas import (
    BookingAssessmentRequest,
    BookingAssessmentResponse,
    ComponentScore,
    Decision,
    RiskReason,
)


POLICY_VERSION = "2026.08-demo.1"


@dataclass(frozen=True)
class Signal:
    code: str
    label: str
    weight: float
    source: str = "rule"


def _bounded_sum(signals: list[Signal]) -> float:
    return round(min(sum(signal.weight for signal in signals), 1.0), 3)


def _component(signals: list[Signal]) -> ComponentScore:
    ordered = sorted(signals, key=lambda signal: signal.weight, reverse=True)
    return ComponentScore(
        score=_bounded_sum(ordered),
        reasons=[
            RiskReason(
                code=signal.code,
                label=signal.label,
                contribution=signal.weight,
                source=signal.source,
            )
            for signal in ordered
        ],
    )


def _payment_signals(booking: BookingAssessmentRequest) -> list[Signal]:
    signals: list[Signal] = []
    chargeback_rate = booking.chargebacks_90d / max(booking.total_bookings_90d, 1)

    if booking.card_on_blocklist:
        signals.append(Signal("card_blocklist", "Payment token is on the demo blocklist", 0.95))
    if booking.device_linked_to_fraud:
        signals.append(Signal("fraud_device", "Device is linked to a previously confirmed case", 0.55))
    if booking.ip_country != booking.card_country:
        signals.append(Signal("country_mismatch", "IP and card countries differ", 0.16))
    if booking.hours_until_departure < 2:
        signals.append(Signal("imminent_departure", "Departure is less than two hours away", 0.18))
    if booking.account_age_days < 7:
        signals.append(Signal("new_account", "Agent account is less than seven days old", 0.18))
    if booking.payment_attempts_10m >= 5:
        signals.append(Signal("payment_velocity", "Five or more payment attempts occurred in ten minutes", 0.28))
    if chargeback_rate >= 0.05 and booking.total_bookings_90d >= 10:
        signals.append(Signal("chargeback_history", "Recent chargeback rate is at least five percent", 0.38))

    return signals


def _inventory_signals(booking: BookingAssessmentRequest) -> list[Signal]:
    signals: list[Signal] = []
    cancellation_rate = booking.cancellations_90d / max(booking.total_bookings_90d, 1)

    if booking.seats_requested >= 20:
        signals.append(Signal("bulk_quantity", "Booking requests at least twenty seats", 0.34))
    elif booking.seats_requested >= 8:
        signals.append(Signal("large_quantity", "Booking requests at least eight seats", 0.16))
    if booking.recent_holds_24h >= 10:
        signals.append(Signal("hold_velocity", "Agent created at least ten holds in 24 hours", 0.33))
    if booking.recent_late_cancellations_90d >= 3:
        signals.append(Signal("late_cancellations", "Agent has at least three recent late cancellations", 0.34))
    if cancellation_rate >= 0.4 and booking.total_bookings_90d >= 10:
        signals.append(Signal("cancellation_rate", "Recent cancellation rate is at least forty percent", 0.28))
    if booking.bookings_24h >= 25:
        signals.append(Signal("booking_velocity", "Agent created at least twenty-five bookings in 24 hours", 0.24))

    return signals


def _bot_signals(booking: BookingAssessmentRequest) -> list[Signal]:
    signals: list[Signal] = []

    if booking.interaction_duration_seconds < 8:
        signals.append(Signal("fast_completion", "Form was completed in under eight seconds", 0.36, "telemetry"))
    if booking.fields_pasted >= 6:
        signals.append(Signal("paste_pattern", "Six or more fields were filled by paste", 0.24, "telemetry"))
    if booking.pointer_events < 3:
        signals.append(Signal("low_pointer_activity", "Fewer than three pointer events were observed", 0.20, "telemetry"))
    if booking.bookings_24h >= 50:
        signals.append(Signal("automation_velocity", "Account created at least fifty bookings in 24 hours", 0.28, "telemetry"))

    return signals


def _summary(decision: Decision, reasons: list[RiskReason]) -> str:
    if not reasons:
        return "Approved because no configured risk signal was triggered."
    evidence = "; ".join(reason.label.lower() for reason in reasons[:3])
    return f"{decision.value.title()} recommended because {evidence}."


def assess_booking(booking: BookingAssessmentRequest) -> BookingAssessmentResponse:
    payment = _component(_payment_signals(booking))
    inventory = _component(_inventory_signals(booking))
    bot = _component(_bot_signals(booking))
    model_predictions = model_store.predict(booking)
    if model_predictions is not None:
        payment_prediction, inventory_prediction = model_predictions
        payment.score = max(payment.score, payment_prediction.score)
        inventory.score = max(inventory.score, inventory_prediction.score)
        payment.reasons.extend(
            RiskReason(
                code=reason.code,
                label=reason.label,
                contribution=reason.contribution,
                source="xgboost_shap",
            )
            for reason in payment_prediction.reasons
        )
        inventory.reasons.extend(
            RiskReason(
                code=reason.code,
                label=reason.label,
                contribution=reason.contribution,
                source="xgboost_shap",
            )
            for reason in inventory_prediction.reasons
        )
    overall = round(max(payment.score, inventory.score, bot.score * 0.6), 3)

    if booking.card_on_blocklist or overall >= 0.8:
        decision = Decision.BLOCK
    elif overall >= 0.35:
        decision = Decision.REVIEW
    else:
        decision = Decision.APPROVE

    reasons = sorted(
        [*payment.reasons, *inventory.reasons, *bot.reasons],
        key=lambda reason: reason.contribution,
        reverse=True,
    )
    return BookingAssessmentResponse(
        booking_id=booking.booking_id,
        decision=decision,
        overall_score=overall,
        payment_fraud=payment,
        inventory_abuse=inventory,
        bot_likelihood=bot,
        engine_mode=model_store.mode,
        policy_version=POLICY_VERSION,
        summary=_summary(decision, reasons),
    )
