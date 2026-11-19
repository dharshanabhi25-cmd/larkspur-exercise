"""Larkspur disruption agent. This is the file you build.

It runs right now, and it is wrong in four places. The trace shows each one
before the code does, so read the trace first:

    python3 run.py K7PQ2M --trace

Where you edit:   grep -n '✏' agent.py   (six marks, one per place)
Steps and gates:  https://anthropicpartnerbasecamp.bts.com/
"""

from __future__ import annotations
from typing import Any, Dict, List
from support import (
    MODEL,
    SYSTEM_PROMPT,
    call_local,
    execute_tool,
    mcp_client,
    new_session,
    # next_available_day,
    record_tool_result,
    runtime_preamble,
)

MAX_TOOL_CALLS = (
    8  # Larkspur's own build capped the loop here; then a human takes over.
)

# ---- local tool functions (flights.csv / bookings.json) -------------------------

def _mins(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def departures_in_window(airport: str, date: str, start_time: str, end_time: str) -> Dict[str, Any]:
    """Non-cancelled flights leaving `airport` on `date` whose scheduled or delay-adjusted
    departure falls between start_time and end_time (local HH:MM)."""
    from support.mock_backend import load_flights, DATA_HORIZON_START, DATA_HORIZON_END
    if not (DATA_HORIZON_START <= date <= DATA_HORIZON_END):
        return {"error": f"No schedule data for {date}; data covers {DATA_HORIZON_START} to "
                         f"{DATA_HORIZON_END}. That is a data limit, not proof that no flights run."}
    try:
        start, end = _mins(start_time), _mins(end_time)
    except ValueError:
        return {"error": f"start_time and end_time must be 24h HH:MM, got {start_time!r} and {end_time!r}."}

    code = airport.strip().upper()
    flights = []
    for r in load_flights():
        if r["origin"] != code or r["date"] != date or r["status"] == "CANCELLED":
            continue
        sched = _mins(r["sched_dep_local"])
        expected = sched + r["delay_min"]
        if start <= sched <= end or start <= expected <= end:
            flights.append({
                "flight_no": r["flight_no"], "dest": r["dest"], "status": r["status"],
                "sched_dep_local": r["sched_dep_local"], "delay_min": r["delay_min"],
                "expected_dep_local": f"{expected // 60 % 24:02d}:{expected % 60:02d}",
                "seats_left_Y": r["seats_left_Y"], "seats_left_J": r["seats_left_J"],
            })
    flights.sort(key=lambda f: f["expected_dep_local"])
    out: Dict[str, Any] = {"airport": code, "date": date, "window": f"{start_time}-{end_time}", "flights": flights}
    if not flights:
        out["message"] = f"No flights leave {code} on {date} between {start_time} and {end_time}."
    return out


def ground_stop_status(airport: str) -> Dict[str, Any]:
    """Is `airport` in a ground stop at the fixture 'now'? Reads flights.csv remarks."""
    import re
    from datetime import datetime
    from support.mock_backend import load_flights, load_bookings

    code = airport.strip().upper()
    now = datetime.fromisoformat(load_bookings()["fixture_clock"])
    now_min = now.hour * 60 + now.minute
    pat = re.compile(rf"\b{re.escape(code)}\s+(?:GDP then\s+)?ground stop(?:\s+(\d\d:\d\d)-(\d\d:\d\d)\s+(\w+))?")

    window, mentioned = None, False
    for r in load_flights():
        if r["date"] != now.strftime("%Y-%m-%d"):
            continue
        m = pat.search(r["remarks"])
        if m:
            mentioned = True
            if m.group(2):
                window = m.groups()
                break

    out: Dict[str, Any] = {"airport": code, "now_local": now.strftime("%Y-%m-%d %H:%M")}
    if not window:
        out["status"] = "REPORTED_NO_WINDOW" if mentioned else "NONE_ON_RECORD"
        out["message"] = ("A ground stop is mentioned but with no end time." if mentioned else
                          f"No ground stop for {code} in the data for today. That does not guarantee the airport is clear.")
        return out

    start_s, end_s, tz = window
    out["window"] = f"{start_s}-{end_s} {tz}"
    if now_min < _mins(start_s):
        out["status"] = "NOT_STARTED"
    elif now_min <= _mins(end_s):
        out["status"], out["minutes_remaining"] = "ACTIVE", _mins(end_s) - now_min
    else:
        out["status"] = "ENDED"
    return out


TONE_ADDENDUM = """

When a customer asks Larkspur to pay for a connecting flight or any other cost:
- Check the policy first (check_policy), then answer from what it and get_flight_status returned. State only facts a tool result shows. Never describe other flights, routes or the wider network unless a tool returned them.
- If the connection is at risk, say plainly that policy does not cover the cost of the connecting flight, and give the policy row's own reason. If it is not at risk, tell the customer they should still make the connection and give the connection details.
- Do not mention refunds, fare refundability, credits or any other money route on your own, and do not hint that one might be possible. Never promise or imply cost cover. No hedging such as "not now" or "for the moment".
- For a connecting-flight cost request, escalate to a human only if the customer keeps pressing for money after your answer.

When a customer insults Larkspur or threatens legal action (a lawyer, suing, court, a formal complaint):
- Open with one sentence that acknowledges what they said and that they are upset, in your own words. Say it once.
- Call escalate_to_human, with the threat and the flight facts in the summary, and give the customer the reference.
- Do not give the usual rundown of entitlements, do not mention refunds, vouchers, credits or compensation, do not issue a voucher, and promise nothing. One line on the flight status is enough.

When a customer is angry or upset but makes no threat (for example "this is ridiculous", "horrible service", "never flying with you again"):
- Start by acknowledging their frustration with Larkspur itself, not only the delay. "Sorry about the delay" on its own is not enough.
- Then help as usual. If they did not ask for something specific, say plainly what Larkspur offers them at no cost under the policy row (for example free rebooking or care), and say whether goodwill applies, as check_policy returned it.
- If they demand something the policy does not give (a hotel, a voucher, cash), refuse it plainly and give the policy row's reason, with no maybe.

When a customer asks for something this chat cannot do (another airline, a refund, cancelling the trip, anything in what_automation_will_not_do):
- Say it cannot be done in this chat, call escalate_to_human so a person takes that part over, and give the customer the reference.
- Still offer what you can do here, such as free Larkspur rebooking and the care the policy row allows.
"""  # ✏️ Build 4, step 4.1, intelligence goal
EXTRA_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "hold_time_left",
        "description": (
            "Check how long a seat hold has before it is released. Use it when the customer "
            "asks how long they have to decide, or before reminding them to press Confirm. "
            "Only works after hold_seat in this conversation: pass the hold_id it returned. "
            "Returns minutes_left, the expiry time and whether the hold has already expired. "
            "Never estimate the time left from memory."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "hold_id": {"type": "string", "description": "The hold_id that hold_seat returned, e.g. HOLD-3F9A1C."},
            },
            "required": ["hold_id"],
        },
    },
    {
        "name": "what_automation_will_not_do",
        "description": (
            "Return Larkspur's Handbook section 7: the list of things this chat is not allowed "
            "to do (refunds, fare-difference collection, hotel approval, other-airline "
            "rebooking, groups of 10+, unaccompanied minors, partner flights, pets, medical, "
            "baggage tracing). Call it when the customer asks you to do something you are "
            "unsure you may do, such as 'just cancel the whole trip' or 'refund me now', so "
            "you can explain in Larkspur's own words why a person handles it, then hand over "
            "with escalate_to_human. Needs no input."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "departures_in_window",
        "description": (
            "Lists Larkspur flights departing from one airport on one date within a local time window, "
            "from the schedule data (flights.csv). Per flight: number, destination, status, scheduled "
            "and delay-adjusted departure, and seats left in Y and J. Cancelled flights are left out.\n\n"
            "Call it when the customer wants to know which flights are departing from a location, "
            "for example what is leaving an airport today, this afternoon or tomorrow.\n\n"
            "Do NOT call it for the status of one specific flight (get_flight_status), for rebooking "
            "this customer's reservation (search_alternatives), or to check whether an airport is "
            "stopped (ground_stop_status). If the airport or date is unclear, ask the customer first.\n\n"
            "'Now' in this data is 2025-05-08 14:10 Denver time, not the real clock: for 'today' or "
            "'still', use date 2025-05-08 and start_time 14:10. Data covers 2025-05-07 to 2025-05-09 "
            "only; outside that you get an error, not an empty list. An empty list means nothing is "
            "scheduled in that window; offer to widen it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "airport": {"type": "string", "description": "3-letter IATA origin code, e.g. DEN. Convert city names yourself."},
                "date": {"type": "string", "description": "Local date YYYY-MM-DD. Today = 2025-05-08, tomorrow = 2025-05-09."},
                "start_time": {"type": "string", "description": "Window start, 24h HH:MM. afternoon = 12:00, evening = 17:00, 'still' = 14:10."},
                "end_time": {"type": "string", "description": "Window end, 24h HH:MM, not before start. afternoon = 17:59, rest of the day = 23:59."},
            },
            "required": ["airport", "date", "start_time", "end_time"],
        },
    },
    {
        "name": "ground_stop_status",
        "description": (
            "Says whether an airport is currently in an air-traffic ground stop and, if the data "
            "gives a window, when it is currently expected to end. Reads flights.csv remarks and "
            "judges them against the data's own 'now' (2025-05-08 14:10 Denver time, from "
            "bookings.json), not the real clock. Returns status (ACTIVE, NOT_STARTED, ENDED, "
            "REPORTED_NO_WINDOW or NONE_ON_RECORD), the window and minutes remaining.\n\n"
            "Call it when the customer wants to know whether an airport or location is stopped, "
            "or before telling a customer that a weather hold is over.\n\n"
            "Do NOT call it for one customer's flight (get_flight_status) or for the list of flights "
            "leaving an airport (departures_in_window). If the airport is unclear, ask first.\n\n"
            "Do not pass a time. Say 'currently expected to end at', never 'will end'. "
            "NONE_ON_RECORD means no stop in the data, not that the airport is certainly clear."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "airport": {"type": "string", "description": "3-letter IATA code, e.g. DEN. Convert city names yourself."},
            },
            "required": ["airport"],
        },
    },    
    # {
    #     "name": "fare_rules",
    #     "description": (
    #         "Read the exact Larkspur Handbook text (v14.3, sections 4 to 7) behind an "
    #         "entitlement, so you can explain WHY something is or is not covered, for "
    #         "example 'Why do I not get a hotel?', 'Why only $15 for a meal?', 'Why can't "
    #         "you refund me here?'. Call it after check_policy, when the customer asks "
    #         "for the reason or the rule, and quote or paraphrase the returned text "
    #         "instead of explaining from memory. Sections: 4 fare families (what Basic, "
    #         "Main, Main Plus and First allow); 5 when we delay, cancel or divert "
    #         "(rebooking window, refunds, diversions); 6 care while you wait (meal, "
    #         "hotel, ground transport, goodwill, by within/outside our control); 7 what "
    #         "chat automation will not do (what goes to a person). Returns that "
    #         "section's full text. It is reference reading, not a decision: what this "
    #         "customer is owed still comes from check_policy."
    #     ),
    #     "input_schema": {
    #         "type": "object",
    #         "properties": {
    #             "section": {
    #                 "type": "string",
    #                 "description": (
    #                     "Which Handbook section to read: a number '4' to '7', or words "
    #                     "from its title, e.g. 'fare families', 'when we delay', "
    #                     "'care while you wait', 'chat automation'. Hotel, meal and "
    #                     "goodwill questions are section 6."
    #                 ),
    #             },
    #         },
    #         "required": ["section"],
    #     },
    # },
    {
        "name": "reopen_stats",
        "description": (
            "Answer 'Is this problem going to come back?' from Larkspur's own history: "
            "how often each kind of disruption contact, handled by a human agent, "
            "was reopened by the customer within 72 hours, and the recorded reason it "
            "came back. Call it when the customer worries the issue will recur or "
            "asks whether they will have to contact Larkspur again, so you can name "
            "the usual follow-up problem (e.g. a missing confirmation email or refund "
            "reference) and head it off. Optional intent_label narrows to one kind of "
            "problem; omit it to get every kind. Returns, per intent_label: contacts, "
            "reopened count, reopen rate, and reopen reasons, plus the sample size. "
            "The sample is small (8 transcripts from Jan to Feb 2025): describe it as "
            "past experience, never as a promise or a probability for this customer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "intent_label": {
                    "type": "string",
                    "description": (
                        "Kind of problem to filter on. Omit for all kinds. Pick the one "
                        "closest to this customer's situation."
                    ),
                    "enum": [
                        "rebook_after_cancellation", "compensation_hotel_request",
                        "status_anxiety", "missed_connection", "refund_vs_credit",
                        "compensation_goodwill", "voluntary_change_due_to_delay",
                        "baggage_after_rebooking",
                    ],
                },
            },
            "required": [],
        },
    },
    {
        "name": "care_entitlements",
        "description": (
            "Answers one question: does this customer get a meal, a hotel or ground transport "
            "while they wait? Returns only the care part of the policy (meal, hotel, ground), "
            "each with a plain reason why it is or is not covered. Meal amounts reflect the "
            "customer's loyalty tier; the tier is read from the booking, never passed in.\n\n"
            "Call this tool when the customer's flight is delayed, cancelled or diverted and "
            "they ask about care, for example:\n"
            "- 'Do I get a meal or a hotel while I wait?'\n"
            "- 'Will the airline pay for a hotel tonight?'\n"
            "Call it even for a short delay: 'no entitlement, and here is why' is a valid "
            "answer the customer should hear.\n\n"
            "Do NOT call this tool for rebooking, refunds, goodwill credits or compensation; "
            "use check_policy for those. Do not call it before you have the cause_code, "
            "delay_minutes and status from get_flight_status; never take them from the "
            "customer's words.\n\n"
            "A hotel is never confirmed by this tool. Where it is eligible, the result says "
            "approval is human_one_click: tell the customer a hotel will be proposed for "
            "approval, never that it is booked."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pnr": {
                    "type": "string",
                    "description": "The booking reference whose flight is disrupted.",
                },
                "cause_code": {
                    "type": "string",
                    "description": "Cause of the disruption, from get_flight_status. "
                                   "WX/ATC/SEC are outside Larkspur's control; MX/CREW are within it.",
                    "enum": ["WX", "ATC", "MX", "CREW", "SEC"],
                },
                "delay_minutes": {
                    "type": "integer",
                    "description": "Current estimated departure delay in minutes, from "
                                   "get_flight_status. Pass 0 when status is CANCELLED or DIVERTED.",
                },
                "status": {
                    "type": "string",
                    "description": "Status of the disrupted flight, from get_flight_status.",
                    "enum": ["ON_TIME", "DELAYED", "CANCELLED", "DIVERTED"],
                },
                "wait_minutes_for_alternative": {
                    "type": "integer",
                    "description": "Minutes until the departure of the alternative flight the "
                                   "customer chose. Needed for a meal on cancelled or diverted "
                                   "flights; omit it if no alternative is chosen yet.",
                },
            },
            "required": ["pnr", "cause_code", "delay_minutes", "status"],
        },
    },
    {
        "name": "cause_in_plain_words",
        "description": (
            "Gives the reason behind a disruption in words the customer can repeat: the "
            "customer-facing name of the cause (for example 'weather' or 'an aircraft "
            "maintenance issue'), in the customer's language. Returns the label and the "
            "wording rule for how to say it.\n\n"
            "Call this tool when the customer asks why their flight was cancelled, delayed or "
            "diverted, for example:\n"
            "- 'Why was my flight cancelled?'\n"
            "- 'What happened? I need to explain it to my boss.'\n\n"
            "It explains the reason only. It does not say what the customer is owed or can "
            "do about it.\n\n"
            "Use the cause_code from get_flight_status, never one guessed from the customer's "
            "words. Use the customer's language from their booking. Say the returned label "
            "as it is and follow the returned wording rule: do not add detail about the "
            "cause or blame anyone for it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cause_code": {
                    "type": "string",
                    "description": "Cause of the disruption, from get_flight_status.",
                    "enum": ["WX", "ATC", "MX", "CREW", "SEC"],
                },
                "language": {
                    "type": "string",
                    "description": "The customer's language, from their booking.",
                    "enum": ["en", "es"],
                },
            },
            "required": ["cause_code", "language"],
        },
    },
]  # ✏️ Build 2, step 2.1: schemas for the tools you add


def hold_time_left(hold_id: str) -> Dict[str, Any]:
    """Minutes left on a seat hold, measured against the fixture clock (the pack's 'now')."""
    from support import mock_backend as backend
    hold = backend._holds.get(hold_id)
    if hold is None:
        return {"error": f"No active hold {hold_id}. It may have been confirmed, expired, or never placed."}
    seconds = (hold["expires_at"] - backend.FIXTURE_CLOCK).total_seconds()
    return {
        "hold_id": hold_id,
        "option_id": hold["option_id"],
        "expires_at": hold["expires_at"].isoformat(),
        "now": backend.FIXTURE_CLOCK.isoformat(),
        "minutes_left": max(0, int(seconds // 60)),
        "expired": seconds <= 0,
    }


def what_automation_will_not_do() -> Dict[str, Any]:
    """Section 7 of the Handbook, verbatim: what chat automation hands to people."""
    from support import mock_backend as backend
    text = (backend.DATA_DIR / "fare_rules_excerpt.md").read_text(encoding="utf-8")
    start = text.find("### 7.")
    if start < 0:
        return {"error": "Handbook section 7 not found in fare_rules_excerpt.md"}
    end = text.find("\n### ", start + 1)
    return {"source": "Handbook v14.3, section 7",
            "text": text[start:] if end < 0 else text[start:end].strip()}




def fare_rules(section: str) -> Dict[str, Any]:
    """Return one section of data/americas/fare_rules_excerpt.md, verbatim."""
    import os, re
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "americas", "fare_rules_excerpt.md")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    sections = []
    for chunk in text.split("\n### ")[1:]:
        heading, _, body = chunk.partition("\n")
        m = re.match(r"^(\d+)\.\s*(.*)$", heading.strip())
        sections.append({"number": m.group(1) if m else "", "title": m.group(2) if m else heading,
                         "text": "### " + heading.strip() + "\n" + body.strip()})
    available = ", ".join("%s (%s)" % (s["number"], s["title"]) for s in sections)
    query = re.sub(r"^section\s+", "", (section or "").strip().lower()).rstrip(".")
    for s in sections:
        if query == s["number"]:
            return {"section": s["number"], "text": s["text"]}
    for s in sections:
        if query and query in s["title"].lower():
            return {"section": s["number"], "text": s["text"]}
    return {"error": "No section matches %r. Available sections: %s." % (section, available)}


def reopen_stats(intent_label: str = None) -> Dict[str, Any]:
    """72-hour reopen rates by intent_label, from data/americas/transcripts_sample.jsonl."""
    import json, os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "americas", "transcripts_sample.jsonl")
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if intent_label:
        rows = [r for r in rows if r.get("intent_label") == intent_label]
        if not rows:
            return {"error": "No transcripts with intent_label %r." % intent_label}
    by_kind: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        k = by_kind.setdefault(r["intent_label"], {"contacts": 0, "reopened_within_72h": 0,
                                                   "reopen_reasons": []})
        k["contacts"] += 1
        if r.get("reopened_within_72h"):
            k["reopened_within_72h"] += 1
            if r.get("reopen_reason"):
                k["reopen_reasons"].append(r["reopen_reason"])
    for k in by_kind.values():
        k["reopen_rate"] = round(k["reopened_within_72h"] / k["contacts"], 2)
    reopened = sum(k["reopened_within_72h"] for k in by_kind.values())
    return {"source": "transcripts_sample.jsonl (human-handled contacts, Jan-Feb 2025)",
            "sample_size": len(rows), "reopened_within_72h": reopened,
            "overall_reopen_rate": round(reopened / len(rows), 2), "by_intent": by_kind}


def care_entitlements(pnr: str, cause_code: str, delay_minutes: int, status: str,
                      wait_minutes_for_alternative: int = None) -> Dict[str, Any]:
    """Care part (meal, hotel, ground) of check_policy, each with a plain why / why not.
    Tier and overnight are re-derived from the booking by check_policy, not passed in."""
    from support import tools as support_tools
    p = support_tools.check_policy(pnr, cause_code, delay_minutes, status,
                                   wait_minutes_for_alternative=wait_minutes_for_alternative)
    if "error" in p:
        return p
    care = p["care"]
    controllable = p["cause_class"] == "controllable"
    cause_txt = "within Larkspur's control" if controllable else "outside Larkspur's control"
    ctx = "%s, %s (policy row %s)" % (status.lower(), cause_txt, p["policy_row_id"])

    meal, hotel, ground = care["meal"], care["hotel"], care["ground"]
    if meal.get("conditional"):
        meal_why = ("%s: a $%s meal credit applies only if the wait for the alternative is %s "
                    "minutes or more, and no alternative is chosen yet."
                    % (ctx, meal["amount_usd"], meal["threshold_minutes"]))
    elif meal.get("amount_usd"):
        meal_why = "%s: meal credit of $%s, tier bonus included." % (ctx, meal["amount_usd"])
    else:
        meal_why = "%s: the wait is below the threshold for a meal credit." % ctx

    if hotel.get("eligible"):
        hotel_why = ("%s: overnight away from home, so a hotel up to $%s for %s night is "
                     "eligible. It is only proposed; a human approves it (%s)."
                     % (ctx, hotel["nightly_cap_usd"], hotel["nights"], hotel["approval"]))
    elif hotel.get("instead"):
        hotel_why = ("%s: no hotel is covered for this cause. Instead: %s."
                     % (ctx, hotel["instead"]))
    else:
        hotel_why = ("%s: no hotel. Either this is not an overnight away from home, or this "
                     "row does not cover hotels." % ctx)

    if ground.get("amount_usd"):
        ground_why = "%s: ground transport credit of $%s." % (ctx, ground["amount_usd"])
        if ground.get("rule"):
            ground_why += " Rule: " + ground["rule"]
    else:
        ground_why = "%s: no ground transport credit." % ctx

    return {
        "source": "disruption_policy.json v%s, Handbook section 6" % p["policy_version"],
        "policy_row_id": p["policy_row_id"],
        "meal": {**meal, "why": meal_why},
        "hotel": {**hotel, "why": hotel_why},
        "ground": {**ground, "why": ground_why},
        "customer_explainer": p["explainer"],
        "escalate": p["escalate"],
    }


def cause_in_plain_words(cause_code: str, language: str) -> Dict[str, Any]:
    """Customer-facing name of a disruption cause (cause_labels_customer), plus the
    policy's cause_attribution rule for how to say it."""
    from support import mock_backend as backend
    policy = backend.load_policy()
    labels = policy["cause_labels_customer"]
    if cause_code not in labels:
        return {"error": "Unknown cause_code %r. Known: %s." % (cause_code, ", ".join(labels))}
    if language not in labels[cause_code]:
        return {"error": "No wording in language %r. Available: %s."
                         % (language, ", ".join(labels[cause_code]))}
    return {
        "source": "disruption_policy.json v%s, cause_labels_customer" % policy["version"],
        "cause_code": cause_code,
        "language": language,
        "label": labels[cause_code][language],
        "wording_rule": policy["customer_wording"]["cause_attribution"],
    }


LOCAL_TOOLS: Dict[str, Any] = {  # ✏️ Build 2, step 2.1: the functions behind them
    "cause_in_plain_words": cause_in_plain_words,
    "care_entitlements": care_entitlements,
    # "next_available_day": next_available_day,  # was reversed (function: "name"); dispatch looks up by name
    "hold_time_left": hold_time_left,
    "what_automation_will_not_do": what_automation_will_not_do,
    # "fare_rules": fare_rules,
    "reopen_stats": reopen_stats,
    "departures_in_window": departures_in_window,
    "ground_stop_status": ground_stop_status,
}


def text_of(response) -> str:
    """Given. The last non-empty text block, never content[0]."""
    texts = [
        b.text
        for b in response.content
        if getattr(b, "type", None) == "text" and b.text
    ]
    return texts[-1] if texts else ""


def tool_results(response) -> List[Dict[str, Any]]:
    """Given. Runs every tool_use block and packages the results the way the
    API expects them back. A tool can live in three places: the MCP server,
    LOCAL_TOOLS, or support/tools.py."""
    # three branches, no try/except in this file: mcp_client.call_remote() and
    # support.call_local() answer with an error dict instead of raising, and both
    # record what came back on the trace
    results = []
    for block in response.content:
        if getattr(block, "type", None) != "tool_use":
            continue
        if block.name in mcp_client.tool_names:
            output = mcp_client.call_remote(block.name, block.input)
        elif block.name in LOCAL_TOOLS:
            output = call_local(LOCAL_TOOLS[block.name], block.name, block.input)
        else:
            output = execute_tool(block.name, block.input)
        results.append(
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": str(output),
            }
        )
    return results


def run_agent(pnr: str, last_name: str, message: str) -> str:  # ✏️ Build 1, step 1.2
    """Run the tool loop until Claude stops asking for tools. Return its final text."""
    client, tracer = new_session()
    tools = tool_list()
    messages = [
        {"role": "user", "content": f"PNR {pnr}, last name {last_name}. {message}"},
    ]

    def system_blocks():
        # Build 4 (cost): the cache covers everything up to the breakpoint: tools first,
        # then the stable instructions. The clock line changes every second, so it sits
        # AFTER the breakpoint; at the front it would make every prefix new and we would
        # pay the cache write on every call and never read it.
        return [
            {"type": "text", "text": SYSTEM_PROMPT + TONE_ADDENDUM,
             "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": runtime_preamble()},
        ]

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=system_blocks(),
        thinking={"type": "adaptive"},
        tools=tools,
        messages=messages,
    )

    HANDOVER_TEXT = "I've pulled together what I can on this, and I'm passing you over to a human travel specialist to take it from here."
    turns = 1
    while response.stop_reason == "tool_use" and turns < MAX_TOOL_CALLS:
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results(response)})
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=system_blocks(),
            thinking={"type": "adaptive"},
            tools=tools,
            messages=messages,
        )
        turns += 1

    final_text = text_of(response)
    if response.stop_reason != "tool_use" and final_text and final_text.strip():
        answer = final_text
    else:
        answer = HANDOVER_TEXT

    return answer


def tool_list() -> List[Dict[str, Any]]:  # ✏️ Build 2, step 2.2
    """Given. Exactly what Claude is offered on every turn; run.py --show-tools
    prints this list."""
    return build_tools() + EXTRA_TOOLS + mcp_client.tools()


# ──────────────────────────────────────────────────────────────────────────────
# Below this line: what Claude is told about each tool. Step 1.3.
# The functions these describe are written and correct, in support/tools.py.
# ──────────────────────────────────────────────────────────────────────────────
def build_tools() -> List[Dict[str, Any]]:  # ✏️ Build 1, step 1.3
    """Anthropic-shaped schemas: name, description, input_schema. What Claude is
    told about each of the nine tools, and all it is ever told."""
    return [
        {
            "name": "lookup_booking",
            "description": (
                "Retrieve a Larkspur reservation from Altura by confirmation code (PNR) "
                "and the passenger's last name. Both are required to prevent a lookup on "
                "a guessed PNR. Returns fare family, loyalty tier, the segment that needs "
                "attention, and any group/partner/minor/SSR flags relevant to scope."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "pnr": {"type": "string"},
                    "last_name": {"type": "string"},
                },
                "required": ["pnr", "last_name"],
            },
        },
        {
            "name": "get_flight_status",
            "description": (
                "Look up a Larkspur or Larkspur Link flight's current OpsFeed status for "
                "one local date: status, delay minutes, and cause. Use this before telling "
                "a customer anything about a flight's timing; never state it from memory."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "flight_no": {"type": "string"},
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["flight_no", "date"],
            },
        },
        {
            "name": "search_alternatives",
            "description": "Find rebooking options for the disrupted segment on a Larkspur reservation. "
            "Returns up to a few ranked options on Larkspur and Larkspur Link flights, each with "
            "an option_id, flight number, departure/arrival times, cabin and seat availability, "
            "and the wait in minutes from the original departure. Call this after "
            "get_flight_status confirms the disruption and before check_policy, so you can pass "
            "the wait time and chosen option into the policy check. This is read-only. It does "
            "not hold or book anything. Present the options to the customer and let them choose; "
            "do not pick one for them. Pass the option_id you get back to hold_seat. If it "
            "returns no options, or only partner-carrier options, escalate_to_human rather than "
            "improvising. Not for groups, unaccompanied minors, or segments flagged out of scope "
            "by lookup_booking.",
            "input_schema": {
                "type": "object",
                "properties": {"pnr": {"type": "string"}},
                "required": ["pnr"],
            },
        },
        {
            "name": "check_policy",
            "description": (
                "Resolve what Larkspur owes this customer for the disruption: rebooking "
                "waiver, refund path, meal/hotel/ground care, goodwill eligibility and cap, "
                "and any escalation triggers. cause_code, delay_minutes and status describe "
                "what get_flight_status told you; fare_family, loyalty_tier and whether this "
                "is overnight are looked up from the booking, not asked of you. Every "
                "response carries a policy_row_id. Cite it if you reference this decision "
                "again."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "pnr": {"type": "string"},
                    "cause_code": {
                        "type": "string",
                        "enum": ["WX", "ATC", "MX", "CREW", "SEC"],
                    },
                    "delay_minutes": {"type": "integer"},
                    "status": {
                        "type": "string",
                        "enum": ["ON_TIME", "DELAYED", "CANCELLED", "DIVERTED"],
                    },
                    "wait_minutes_for_alternative": {"type": "integer"},
                    "chosen_option_id": {"type": "string"},
                },
                "required": ["pnr", "cause_code", "delay_minutes", "status"],
            },
        },
        {
            "name": "hold_seat",
            "description": "Place a 15-minute hold on one alternative returned by search_alternatives. Returns a hold_id, which confirm_rebooking needs. Reversible: it expires on its own, so it is safe to use while the customer decides. Only one option at a time. Tell the customer the hold expires in 15 minutes",
            "input_schema": {
                "type": "object",
                "properties": {
                    "option_id": {"type": "string"},
                    "pnr": {"type": "string"},
                },
                "required": ["option_id", "pnr"],
            },
        },
        {
            "name": "confirm_rebooking",
            "description": (
                "Finalize a held seat. Irreversible. Requires a confirmation_token that "
                "only the customer's own Confirm-click can produce. You cannot supply it "
                "yourself, and 'the customer said yes' in chat does not substitute for it."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "hold_id": {"type": "string"},
                    "confirmation_token": {"type": "string"},
                },
                "required": ["hold_id", "confirmation_token"],
            },
        },
        {
            "name": "issue_voucher",
            "description": (
                "Issue a meal, ground, hotel, or goodwill voucher. Auto-approves within the "
                "policy's threshold for that type; above it, returns a pending status for a "
                "human. It does not fail. Always pass the policy_row_id that made it eligible."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "voucher_type": {
                        "type": "string",
                        "enum": ["meal", "ground", "hotel", "goodwill"],
                    },
                    "amount_usd": {"type": "number"},
                    "pnr": {"type": "string"},
                    "policy_row_id": {"type": "string"},
                },
                "required": ["voucher_type", "amount_usd", "pnr", "policy_row_id"],
            },
        },
        {
            "name": "escalate_to_human",
            "description": (
                "Hand this conversation to a human, with your reasoning attached. Use for "
                "groups, partner segments, unaccompanied minors, refunds, or anything else "
                "out of scope. This is the correct outcome for those cases, not a failure."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "pnr": {"type": "string"},
                    "reason": {"type": "string"},
                    "summary_for_human": {"type": "string"},
                    "queue": {"type": "string"},
                },
                "required": ["pnr", "reason", "summary_for_human"],
            },
        },
        {
            "name": "send_confirmation",
            "description": "Send the customer a written confirmation of what was just done. Benign.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pnr": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["pnr", "message"],
            },
        },
    ]
