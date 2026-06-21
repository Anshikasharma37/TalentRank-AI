
from datetime import date, datetime
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import ACTIVITY_THRESHOLDS


def normalize_signals(signals: dict, reference_date: date = None) -> dict:
   
    if reference_date is None:
        reference_date = date.today()

    # Activity: days since last login
    last_active_raw = signals.get("last_active_date", "")
    days_since_active = _days_since(last_active_raw, reference_date)

    if days_since_active <= ACTIVITY_THRESHOLDS["very_active"]:
        activity_score = 1.0
    elif days_since_active <= ACTIVITY_THRESHOLDS["active"]:
        activity_score = 0.75
    elif days_since_active <= ACTIVITY_THRESHOLDS["somewhat"]:
        activity_score = 0.4
    else:
        activity_score = 0.1   

    #  Availability: open to work + short notice period 
    open_to_work   = 1.0 if signals.get("open_to_work_flag", False) else 0.3
    notice_days    = signals.get("notice_period_days", 60)
    # Score notice period: 0 days = 1.0, 30 days = 0.8, 60 days = 0.6, 90+ days = 0.3
    if notice_days == 0:
        notice_score = 1.0
    elif notice_days <= 30:
        notice_score = 0.8
    elif notice_days <= 60:
        notice_score = 0.6
    elif notice_days <= 90:
        notice_score = 0.4
    else:
        notice_score = 0.3
    availability_score = (open_to_work * 0.7) + (notice_score * 0.3)

    # GitHub: code activity score
    github_raw = signals.get("github_activity_score", -1)
    if github_raw == -1:
        github_score = 0.0   # no GitHub linked — neutral
    else:
        github_score = min(github_raw / 100.0, 1.0)

    # Recruiter response rate 
    response_rate = float(signals.get("recruiter_response_rate", 0.0))
    response_score = min(response_rate, 1.0)

    #  Reliability: interview completion + offer acceptance
    interview_rate = float(signals.get("interview_completion_rate", 0.0))
    offer_rate     = signals.get("offer_acceptance_rate", -1)
    if offer_rate == -1:
        offer_score = 0.5   # no history → neutral
    else:
        offer_score = max(0.0, min(float(offer_rate), 1.0))
    reliability_score = (interview_rate * 0.6) + (offer_score * 0.4)

    # Profile quality 
    completeness    = float(signals.get("profile_completeness_score", 50)) / 100.0
    verified_email  = 1.0 if signals.get("verified_email", False) else 0.0
    verified_phone  = 1.0 if signals.get("verified_phone", False) else 0.0
    linkedin        = 1.0 if signals.get("linkedin_connected", False) else 0.0
    profile_quality = (completeness * 0.5) + (verified_email * 0.2) + \
                      (verified_phone * 0.2) + (linkedin * 0.1)

    #  Platform engagement
    saved_by_30d   = min(signals.get("saved_by_recruiters_30d", 0) / 20.0, 1.0)

    return {
        "activity_score":     activity_score,
        "availability_score": availability_score,
        "github_score":       github_score,
        "response_score":     response_score,
        "reliability_score":  reliability_score,
        "profile_quality":    profile_quality,
        "saved_by_30d":       saved_by_30d,
        "days_since_active":  days_since_active,  # kept for disqualifier checks
        "open_to_work":       signals.get("open_to_work_flag", False),
        "notice_period_days": notice_days,
    }


def composite_behavioral_score(normalized: dict) -> float:
   
    return (
        normalized["activity_score"]     * 0.30 +
        normalized["response_score"]     * 0.25 +
        normalized["availability_score"] * 0.20 +
        normalized["reliability_score"]  * 0.15 +
        normalized["github_score"]       * 0.10
    )


def _days_since(date_str: str, reference: date) -> int:
   
    if not date_str:
        return 999
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return max(0, (reference - d).days)
    except ValueError:
        return 999


if __name__ == "__main__":
   
    test_signals = {
        "last_active_date": "2026-05-20",
        "open_to_work_flag": True,
        "notice_period_days": 30,
        "recruiter_response_rate": 0.75,
        "github_activity_score": 65,
        "interview_completion_rate": 0.9,
        "offer_acceptance_rate": 0.8,
        "profile_completeness_score": 90,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": True,
        "saved_by_recruiters_30d": 5,
    }
    norm = normalize_signals(test_signals, reference_date=date(2026, 6, 12))
    print("Normalized signals:")
    for k, v in norm.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\nComposite behavioral score: {composite_behavioral_score(norm):.3f}")
