"""Synthetic institute email generator with curriculum difficulty levels.

Every email is produced from a template together with its exact gold label, so
no manual labelling is needed. Templates carry an id; a quarter of them (and a
few courses / venues) are reserved for the *test* split only, so the synthetic
test set measures generalisation to unseen phrasings instead of memorisation.

Levels:
    L1  single event, explicit long-form date and time
    L2  varied date/time formats (05/11/2026, Nov 5th, 5 pm, EOD, noon, 10:00 hrs)
    L3  relative dates (today, tomorrow, this Friday, next Friday)
    L4  distractors (postponements, quoted reply chains) and 2-4 events per email
    L5  forwarded / noisy emails, partly Hinglish
About 15% of emails at every level contain no event at all.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Vocabulary. The last entries of COURSES / VENUES are test-only.
# ---------------------------------------------------------------------------

COURSES = [
    ("CSL3090", "Artificial Intelligence"),
    ("CSL3010", "Operating Systems"),
    ("CSL3020", "Computer Networks"),
    ("CSL7020", "Deep Reinforcement Learning"),
    ("EEL2010", "Signals and Systems"),
    ("MEL2010", "Fluid Mechanics"),
    ("CYL1010", "Engineering Chemistry"),
    ("MAL2010", "Linear Algebra and Optimization"),
    ("CHL3020", "Mass Transfer"),
    ("PHL1020", "Quantum Physics"),
    # test-only
    ("BBL2010", "Molecular Biology"),
    ("HSL4010", "Economics of Innovation"),
]
N_TEST_ONLY_COURSES = 2

SENDERS = [
    ("Prof. Anand Sharma", "anand.sharma@iitj.ac.in"),
    ("Prof. Ritu Verma", "ritu.verma@iitj.ac.in"),
    ("Dr. Vikram Singh", "vikram.singh@iitj.ac.in"),
    ("Dr. Neha Gupta", "neha.gupta@iitj.ac.in"),
    ("Dr. Arjun Mehta", "arjun.mehta@iitj.ac.in"),
    ("Academic Office", "acad-office@iitj.ac.in"),
    ("CDC Office", "cdc@iitj.ac.in"),
    ("Student Gymkhana", "gymkhana@iitj.ac.in"),
]

VENUES = [
    "LHC-101",
    "LHC-102",
    "LHC-201",
    "LHC-305",
    "Seminar Hall 1",
    "Dept Conference Room",
    "Lab 214",
    "Online (Google Meet)",
    "Auditorium",
    "Jodhpur Club",
    # test-only
    "LHC-108",
    "Seminar Hall 3",
]
N_TEST_ONLY_VENUES = 2

MONTH_NAMES = ["", "January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]
SHORT_MONTHS = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

GREETINGS = ["Dear Students,", "Dear All,", "Hello everyone,", "Dear Class,", "Hi all,", "Dear Institute Members,"]
SIGNOFFS = ["Regards,", "Best regards,", "Thanks,", "Best wishes,", "Sincerely,", "Warm regards,"]
NOISE_LINES = [
    "Attendance will be taken.",
    "Please find further details on the course page.",
    "For any queries, contact the TAs.",
    "Kindly treat this as important.",
    "This notice is issued on {received}.",
    "Please ignore if already done.",
    "Slides from the last lecture are on Google Classroom.",
]

# ---------------------------------------------------------------------------
# Event templates: (subject, sentence, title). Sentences use {when}, {venue},
# {course}, {cname}, {n}. Deadline templates never mention a venue.
# Indices >= N_TRAIN_TEMPLATES in each list are test-only.
# ---------------------------------------------------------------------------

N_TRAIN_TEMPLATES = 6

EVENT_TEMPLATES: dict[str, list[tuple[str, str, str]]] = {
    "deadline": [
        ("[{course}] Assignment {n} Submission Deadline",
         "Please remember that Assignment {n} is due {when}. Upload your code and report on the course portal.",
         "{course} Assignment {n} Submission"),
        ("[{course}] Lab Report {n} due",
         "Lab report {n} must be submitted {when} via Google Classroom.",
         "{course} Lab Report {n} Submission"),
        ("[{course}] Project proposal",
         "The project proposal (max 2 pages) has to be emailed to the TAs {when}.",
         "{course} Project Proposal Submission"),
        ("Course add/drop - last date",
         "The add/drop window for {cname} ({course}) closes {when}. Requests after this will not be entertained.",
         "{course} Add/Drop Deadline"),
        ("[{course}] Homework {n}",
         "Homework {n} is now live; the submission deadline is {when}.",
         "{course} Homework {n} Submission"),
        ("Semester fee payment reminder",
         "Students are reminded to complete the semester fee payment {when} to avoid a late fine.",
         "Semester Fee Payment Deadline"),
        # test-only
        ("[{course}] Re-evaluation requests",
         "Any re-evaluation requests for {cname} must reach me {when}.",
         "{course} Re-evaluation Request Deadline"),
        ("[{course}] Mini-project code freeze",
         "Final code for the mini-project must be pushed to GitHub {when}; commits after that will be ignored.",
         "{course} Mini-Project Code Submission"),
    ],
    "exam": [
        ("[{course}] Mid-Semester Examination Schedule",
         "The mid-semester examination for {cname} ({course}) is scheduled {when} in {venue}.",
         "{course} Mid-Semester Exam"),
        ("[{course}] Quiz {n}",
         "Quiz {n} will be conducted {when}. Venue: {venue}. Syllabus: everything up to last week's lecture.",
         "{course} Quiz {n}"),
        ("[{course}] End-Semester Exam",
         "The end-semester exam of {course} will be held {when} at {venue}. Bring your ID card.",
         "{course} End-Semester Exam"),
        ("[{course}] Lab Viva",
         "The lab viva for {cname} is scheduled {when} in {venue}. Bring your lab record.",
         "{course} Lab Viva"),
        ("[{course}] Makeup quiz",
         "A makeup quiz for students who missed Quiz {n} will be held {when} in {venue}.",
         "{course} Makeup Quiz"),
        ("[{course}] Class test",
         "There will be a short class test {when} in {venue}; it carries 5% weight.",
         "{course} Class Test"),
        # test-only
        ("[{course}] Practical exam",
         "The practical examination for {cname} will take place {when} in {venue}.",
         "{course} Practical Exam"),
        ("[{course}] Minor exam {n}",
         "Minor-{n} examination of {course} is fixed {when}. Report to {venue} 10 minutes early.",
         "{course} Minor Exam {n}"),
    ],
    "class": [
        ("[{course}] Compensatory Class",
         "We will have a compensatory lecture for {course} {when} in {venue}.",
         "{course} Compensatory Lecture"),
        ("[{course}] Extra tutorial",
         "An extra tutorial for {cname} will be held {when} in {venue}.",
         "{course} Extra Tutorial"),
        ("[{course}] Class rescheduled",
         "The {course} lecture has been rescheduled and will now take place {when} in {venue}.",
         "{course} Rescheduled Lecture"),
        ("[{course}] Lab session",
         "The next {cname} lab session will be conducted {when} in {venue}.",
         "{course} Lab Session"),
        ("[{course}] Revision lecture",
         "I will take a revision lecture before the exam {when} in {venue}.",
         "{course} Revision Lecture"),
        ("[{course}] Venue change",
         "The {course} lecture {when} will be held in {venue} instead of the usual room.",
         "{course} Lecture"),
        # test-only
        ("[{course}] Guest lecture in class slot",
         "Our regular {course} slot {when} will be a guest lecture in {venue}; attendance is compulsory.",
         "{course} Guest Lecture"),
        ("[{course}] Extra lab slot",
         "An additional lab slot for {cname} has been arranged {when} in {venue}.",
         "{course} Extra Lab Slot"),
    ],
    "event": [
        ("Distinguished Lecture Series: {cname}",
         "You are cordially invited to a guest talk on recent advances in {cname} {when} in {venue}.",
         "Distinguished Lecture on {cname}"),
        ("Workshop on {cname}",
         "The department is organising a hands-on workshop on {cname} {when} at {venue}. Register using the form.",
         "Workshop on {cname}"),
        ("Hackathon announcement",
         "The Student Gymkhana's annual hackathon kicks off {when} at {venue}.",
         "Gymkhana Hackathon"),
        ("Pre-placement talk",
         "CDC invites all pre-final year students to a pre-placement talk {when} in {venue}.",
         "Pre-Placement Talk"),
        ("Robotics Club orientation",
         "The Robotics Club orientation for freshers is {when} at {venue}. Snacks will be provided.",
         "Robotics Club Orientation"),
        ("Seminar: {cname}",
         "A departmental seminar on {cname} will be delivered by a visiting faculty member {when} in {venue}.",
         "Departmental Seminar on {cname}"),
        # test-only
        ("Cultural night",
         "Join us for the cultural night {when} at {venue}.",
         "Cultural Night"),
        ("Research colloquium",
         "The PhD research colloquium on {cname} is scheduled {when} in {venue}. All are welcome.",
         "Research Colloquium on {cname}"),
    ],
    "meeting": [
        ("[{course}] TA Office Hours",
         "The TAs will hold office hours {when} at {venue}. Bring your assignment queries.",
         "{course} TA Office Hours"),
        ("[{course}] Project review meeting",
         "All project groups must meet me {when} in {venue} to present their progress.",
         "{course} Project Review Meeting"),
        ("Class representatives meeting",
         "A meeting of all class representatives with the academic office is called {when} in {venue}.",
         "Class Representatives Meeting"),
        ("[{course}] Doubt clearing session",
         "There will be a doubt clearing session for {course} {when} in {venue}.",
         "{course} Doubt Clearing Session"),
        ("Mentor-mentee meeting",
         "Your faculty mentor meeting is scheduled {when} in {venue}.",
         "Faculty Mentor Meeting"),
        ("[{course}] Marks discussion",
         "Students who want to discuss their mid-sem marks can meet me {when} in {venue}.",
         "{course} Marks Discussion"),
        # test-only
        ("Hostel committee meeting",
         "The hostel committee general body meeting will be held {when} at {venue}.",
         "Hostel Committee Meeting"),
        ("[{course}] TA meeting",
         "All TAs of {course} please assemble {when} in {venue} for evaluation planning.",
         "{course} TA Meeting"),
    ],
}

# Hinglish (L5): (sentence, title, category). {hdate} {htime} {venue} {course}.
N_TRAIN_HINGLISH = 3
HINGLISH_TEMPLATES = [
    ("{hdate} {htime} baje {venue} mein {course} ka quiz hai, sab log time pe aa jana.", "{course} Quiz", "exam"),
    ("{hdate} {htime} baje {venue} mein {course} ki extra class hogi, attendance compulsory hai.", "{course} Extra Class", "class"),
    ("{hdate} {htime} baje {venue} mein project groups ki meeting rakhi hai, please late mat hona.", "{course} Project Meeting", "meeting"),
    # test-only
    ("{hdate} {htime} baje {venue} mein ek guest talk hai {course} pe, zaroor aana.", "{course} Guest Talk", "event"),
]

# No-event templates: (subject, body sentence). {past} is a date before the email.
N_TRAIN_NO_EVENT = 12
NO_EVENT_TEMPLATES = [
    ("[{course}] Marks published for Quiz {n}",
     "The marks for Quiz {n}, held on {past}, are now available on the portal. Report discrepancies to the TAs."),
    ("Library book return advisory",
     "All students are advised to check their library account for overdue books. Late fines may be levied."),
    ("Campus WiFi maintenance completed",
     "The maintenance of the core network switches in Academic Block 2 has been completed. All services are restored."),
    ("Mess menu feedback",
     "The Mess Committee invites suggestions on the proposed menu revision. The form link is on the hostel portal."),
    ("[{course}] Reading material uploaded",
     "Supplementary reading material on dynamic programming has been uploaded to the course drive."),
    ("Thank you for attending",
     "Thank you all for attending the workshop on {past}. The slides are now on the department website."),
    ("Lost and found",
     "A blue water bottle was found in {venue}. The owner can collect it from the security office."),
    ("Hackathon results",
     "Results of the hackathon held on {past} have been announced. Congratulations to all winners!"),
    ("[{course}] Lecture slides uploaded",
     "Slides for the last two lectures of {cname} are now on Google Classroom."),
    ("Scholarship disbursed",
     "The merit scholarship for this semester has been credited to the registered bank accounts."),
    ("Keep the campus clean",
     "Students are requested not to leave food packets in the lecture halls. Let us keep our campus clean."),
    ("Newsletter released",
     "The latest issue of the institute newsletter is out. Read about the research highlights of the month."),
    # test-only
    ("[{course}] Attendance list uploaded",
     "The attendance record of {course} till {past} has been uploaded. Please verify your entries."),
    ("Congratulations to the cricket team",
     "Our cricket team won the inter-IIT friendly match played on {past}. Well done, team!"),
    ("Holiday list uploaded",
     "The list of holidays for the next semester has been uploaded on the academic calendar page."),
    ("[{course}] Feedback form",
     "Please fill in the mid-semester course feedback form for {cname}. Your responses are anonymous."),
]

# Plausible start times (and optional durations in hours) per category.
START_TIMES = {
    "deadline": ["23:59", "23:59", "23:59", "17:00", "12:00", "18:00", None],
    "exam": ["09:00", "10:00", "14:00", "14:30", "15:00", "11:00"],
    "class": ["08:00", "09:00", "11:00", "14:00", "16:00", "17:30"],
    "event": ["11:00", "15:00", "16:00", "17:30", "18:00", "19:00", None],
    "meeting": ["10:00", "16:00", "17:00", "18:30", "12:00", None],
}
DURATION_HOURS = {"exam": [1, 2, 3], "class": [1], "event": [1, 2], "meeting": [1]}


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _ordinal(day: int) -> str:
    suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix}"


def format_long_date(d: date, with_weekday: bool = False) -> str:
    s = f"{d.day} {MONTH_NAMES[d.month]} {d.year}"
    return f"{DAY_NAMES[d.weekday()]}, {s}" if with_weekday else s


def _format_varied_date(d: date, rng: random.Random) -> str:
    return rng.choice([
        f"{SHORT_MONTHS[d.month]} {_ordinal(d.day)}",
        f"{d.day:02d}/{d.month:02d}/{d.year}",
        f"{d.year}-{d.month:02d}-{d.day:02d}",
        f"{d.day} {SHORT_MONTHS[d.month]}",
        f"{_ordinal(d.day)} {MONTH_NAMES[d.month]}",
        f"{MONTH_NAMES[d.month]} {d.day}",
        f"{SHORT_MONTHS[d.month]} {d.day}, {d.year}",
    ])


def _to_12h(hhmm: str, compact: bool) -> str:
    h, m = map(int, hhmm.split(":"))
    meridiem = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    if compact:
        return f"{h12} {meridiem.lower()}" if m == 0 else f"{h12}:{m:02d} {meridiem.lower()}"
    return f"{h12}:{m:02d} {meridiem}"


def _add_hours(hhmm: str, hours: int) -> str:
    h, m = map(int, hhmm.split(":"))
    return f"{min(h + hours, 23):02d}:{m:02d}"


def render_time(start: Optional[str], end: Optional[str], level: int, rng: random.Random) -> str:
    """Render a start (and optional end) time as natural text. Empty string if no time."""
    if start is None:
        return ""
    varied = level >= 2
    if start == "23:59":
        if not varied:
            return "by 11:59 PM"
        return rng.choice(["by midnight", "before 23:59", "by 11:59 pm", "by EOD", "by end of day", "by 23:59 hrs"])
    if start == "12:00" and varied and end is None and rng.random() < 0.5:
        return rng.choice(["at noon", "by noon"])

    def fmt(t: str) -> str:
        if not varied:
            return _to_12h(t, compact=False)
        return rng.choice([_to_12h(t, compact=True), t, f"{t} hrs", _to_12h(t, compact=False)])

    if end is not None:
        if varied and rng.random() < 0.4:
            return f"{start}-{end}"
        return f"from {fmt(start)} to {fmt(end)}"
    return f"at {fmt(start)}"


def resolve_relative_date(base: date, expr: str) -> date:
    """Resolve a relative date expression using the convention stated in the system prompt.

    today = base; tomorrow = +1; day after tomorrow = +2;
    "this X" = the coming X within the next 1-7 days;
    "next X" = 7 days after "this X".
    """
    if expr == "today":
        return base
    if expr == "tomorrow":
        return base + timedelta(days=1)
    if expr == "day after tomorrow":
        return base + timedelta(days=2)
    prefix, day_name = expr.split(" ", 1)
    target = DAY_NAMES.index(day_name.strip().capitalize())
    diff = (target - base.weekday()) % 7 or 7
    if prefix == "this":
        return base + timedelta(days=diff)
    if prefix == "next":
        return base + timedelta(days=diff + 7)
    raise ValueError(f"Unknown relative date expression: {expr!r}")


def _pick_date(received: date, level: int, rng: random.Random) -> tuple[date, str]:
    """Pick an event date and the phrase used for it in the email (with preposition)."""
    if level == 3:
        expr = rng.choice(["today", "tomorrow", "tomorrow", "day after tomorrow", "this", "this", "next", "next"])
        if expr in ("this", "next"):
            offset = rng.randint(3, 6) if expr == "this" else rng.randint(8, 13)
            expr = f"{expr} {DAY_NAMES[(received.weekday() + offset) % 7]}"
        return resolve_relative_date(received, expr), expr
    d = received + timedelta(days=rng.randint(1, 20))
    if level == 1:
        return d, f"on {format_long_date(d, with_weekday=rng.random() < 0.3)}"
    return d, f"on {_format_varied_date(d, rng)}"


def _join_when(date_phrase: str, time_phrase: str, rng: random.Random) -> str:
    if not time_phrase:
        return date_phrase
    if rng.random() < 0.5:
        return f"{date_phrase} {time_phrase}"
    return f"{time_phrase} {date_phrase}"


class _Ctx:
    """Per-split sampling pools (template indices, courses, venues)."""

    def __init__(self, split: str):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")
        self.test = split == "test"
        train_courses = COURSES[:-N_TEST_ONLY_COURSES]
        train_venues = VENUES[:-N_TEST_ONLY_VENUES]
        self.courses = COURSES if self.test else train_courses
        self.venues = VENUES if self.test else train_venues

    def template_indices(self, n_total: int, n_train: int) -> list[int]:
        return list(range(n_train, n_total)) if self.test else list(range(n_train))


def _make_event(
    rng: random.Random, ctx: _Ctx, received: date, level: int, category: Optional[str] = None
) -> tuple[str, str, dict[str, Any], str]:
    """Build one event: returns (subject, sentence, gold_event, template_id)."""
    category = category or rng.choice(list(EVENT_TEMPLATES))
    templates = EVENT_TEMPLATES[category]
    idx = rng.choice(ctx.template_indices(len(templates), N_TRAIN_TEMPLATES))
    subject_t, sentence_t, title_t = templates[idx]

    course, cname = rng.choice(ctx.courses)
    venue = rng.choice(ctx.venues)
    n = rng.randint(1, 5)

    event_date, date_phrase = _pick_date(received, level, rng)
    start = rng.choice(START_TIMES[category])
    end = None
    if start is not None and category in DURATION_HOURS and rng.random() < 0.4:
        end = _add_hours(start, rng.choice(DURATION_HOURS[category]))
    when = _join_when(date_phrase, render_time(start, end, level, rng), rng)

    slots = {"course": course, "cname": cname, "venue": venue, "n": n, "when": when}
    has_venue = "{venue}" in sentence_t
    gold = {
        "title": title_t.format(**slots),
        "type": category,
        "date": event_date.isoformat(),
        "start_time": start,
        "end_time": end,
        "location": venue if has_venue else None,
    }
    return subject_t.format(**slots), sentence_t.format(**slots), gold, f"{category}_{idx}"


def _wrap_body(rng: random.Random, received: date, sender_name: str, core: list[str]) -> str:
    lines = list(core)
    for _ in range(rng.choice([0, 0, 1, 2])):
        lines.insert(rng.randint(0, len(lines)), rng.choice(NOISE_LINES).format(received=format_long_date(received)))
    return "\n\n".join([rng.choice(GREETINGS), *lines, f"{rng.choice(SIGNOFFS)}\n{sender_name}"])


def _random_received(rng: random.Random) -> date:
    return date(2026, 7, 20) + timedelta(days=rng.randint(0, 130))


# ---------------------------------------------------------------------------
# Email generators per level
# ---------------------------------------------------------------------------

def generate_no_event_email(rng: random.Random, ctx: _Ctx, email_id: str, level: int) -> dict[str, Any]:
    received = _random_received(rng)
    sender_name, sender_email = rng.choice(SENDERS)
    course, cname = rng.choice(ctx.courses)
    idx = rng.choice(ctx.template_indices(len(NO_EVENT_TEMPLATES), N_TRAIN_NO_EVENT))
    subject_t, sentence_t = NO_EVENT_TEMPLATES[idx]
    past = received - timedelta(days=rng.randint(1, 10))
    past_str = format_long_date(past) if level == 1 else _format_varied_date(past, rng)
    slots = {"course": course, "cname": cname, "n": rng.randint(1, 5),
             "venue": rng.choice(ctx.venues), "past": past_str}
    return {
        "id": email_id,
        "received_date": received.isoformat(),
        "sender": sender_email,
        "subject": subject_t.format(**slots),
        "body": _wrap_body(rng, received, sender_name, [sentence_t.format(**slots)]),
        "gold": [],
        "level": level,
        "template_ids": [f"none_{idx}"],
    }


def generate_single_event_email(rng: random.Random, ctx: _Ctx, email_id: str, level: int) -> dict[str, Any]:
    received = _random_received(rng)
    sender_name, sender_email = rng.choice(SENDERS)
    subject, sentence, gold, tid = _make_event(rng, ctx, received, level)
    return {
        "id": email_id,
        "received_date": received.isoformat(),
        "sender": sender_email,
        "subject": subject,
        "body": _wrap_body(rng, received, sender_name, [sentence]),
        "gold": [gold],
        "level": level,
        "template_ids": [tid],
    }


def _with_course_code(sentence: str, gold: dict[str, Any]) -> str:
    """In a digest there is no per-event subject line, so make sure the course code is visible."""
    code = gold["title"].split()[0]
    if re.fullmatch(r"[A-Z]{3}\d{4}", code) and code not in sentence:
        return f"{code}: {sentence}"
    return sentence


def generate_distractor_email(rng: random.Random, ctx: _Ctx, email_id: str) -> dict[str, Any]:
    """L4: postponement, quoted reply chain, or a multi-event digest."""
    received = _random_received(rng)
    sender_name, sender_email = rng.choice(SENDERS)
    scenario = rng.choice(["postponed", "reply_chain", "digest", "digest"])

    if scenario == "digest":
        k = rng.randint(2, 4)
        events = [_make_event(rng, ctx, received, rng.choice([1, 2])) for _ in range(k)]
        # Keep one event per (title, date) so the gold set is unambiguous
        seen, uniq = set(), []
        for e in events:
            key = (e[2]["title"], e[2]["date"])
            if key not in seen:
                seen.add(key)
                uniq.append(e)
        core = ["Please note the following upcoming items:\n" + "\n".join(f"- {_with_course_code(e[1], e[2])}" for e in uniq)]
        return {
            "id": email_id,
            "received_date": received.isoformat(),
            "sender": sender_email,
            "subject": rng.choice(["Important dates this fortnight", "Upcoming schedule", "Reminder: upcoming academic activities"]),
            "body": _wrap_body(rng, received, sender_name, core),
            "gold": [e[2] for e in uniq],
            "level": 4,
            "template_ids": [e[3] for e in uniq],
        }

    category = rng.choice(["deadline", "exam", "class", "meeting"])
    subject, sentence, gold, tid = _make_event(rng, ctx, received, rng.choice([1, 2]), category)
    old = received + timedelta(days=rng.randint(1, 4))
    if old.isoformat() == gold["date"]:
        old += timedelta(days=1)
    old_str = format_long_date(old)

    if scenario == "postponed":
        core = [
            f"Please note: the {gold['title']} originally planned on {old_str} has been postponed.",
            f"Updated schedule: {sentence}",
        ]
        subject = f"Postponed: {subject}"
    else:
        quoted_day = received - timedelta(days=rng.randint(1, 5))
        core = [
            f"Correction to my earlier mail. {sentence}",
            f"On {format_long_date(quoted_day)}, {sender_name} wrote:\n> The {gold['title']} will be on {old_str}.",
        ]
        subject = f"Re: {subject}"
    return {
        "id": email_id,
        "received_date": received.isoformat(),
        "sender": sender_email,
        "subject": subject,
        "body": _wrap_body(rng, received, sender_name, core),
        "gold": [gold],
        "level": 4,
        "template_ids": [tid],
    }


def _hinglish_event(rng: random.Random, ctx: _Ctx, received: date) -> tuple[str, dict[str, Any], str]:
    idx = rng.choice(ctx.template_indices(len(HINGLISH_TEMPLATES), N_TRAIN_HINGLISH))
    sentence_t, title_t, category = HINGLISH_TEMPLATES[idx]
    course, _ = rng.choice(ctx.courses)
    venue = rng.choice(ctx.venues)

    choice = rng.choice(["kal", "parso", "abs"])
    if choice == "kal":
        d, hdate = received + timedelta(days=1), "Kal"
    elif choice == "parso":
        d, hdate = received + timedelta(days=2), "Parso"
    else:
        d = received + timedelta(days=rng.randint(3, 15))
        hdate = f"{d.day} {SHORT_MONTHS[d.month]} ko"

    hour, part = rng.choice([(9, "subah"), (10, "subah"), (11, "subah"), (14, "dopahar"), (15, "dopahar"),
                             (17, "shaam"), (18, "shaam")])
    htime = f"{part} {hour % 12 or 12}"
    slots = {"hdate": hdate, "htime": htime, "venue": venue, "course": course}
    gold = {
        "title": title_t.format(**slots),
        "type": category,
        "date": d.isoformat(),
        "start_time": f"{hour:02d}:00",
        "end_time": None,
        "location": venue,
    }
    return sentence_t.format(**slots), gold, f"hinglish_{idx}"


def generate_noisy_email(rng: random.Random, ctx: _Ctx, email_id: str) -> dict[str, Any]:
    """L5: forwarded thread with headers, noise, and sometimes Hinglish."""
    received = _random_received(rng)
    sender_name, sender_email = rng.choice(SENDERS)

    if rng.random() < 0.4:
        sentence, gold, tid = _hinglish_event(rng, ctx, received)
        subject = f"Fwd: {gold['title']}"
    else:
        subject, sentence, gold, tid = _make_event(rng, ctx, received, rng.choice([2, 3]))
        subject = f"Fwd: {subject}"

    fwd_sent = received - timedelta(days=rng.randint(0, 1))
    header = (
        "---------- Forwarded message ---------\n"
        f"From: TA Head <ta.head@iitj.ac.in>\n"
        f"Date: {DAY_NAMES[fwd_sent.weekday()][:3]}, {fwd_sent.day} {SHORT_MONTHS[fwd_sent.month]} "
        f"{fwd_sent.year} at {rng.choice(['9:15 AM', '11:02 PM', '6:40 PM'])}\n"
        f"Subject: {subject[5:]}\n"
        f"To: {sender_name} <{sender_email}>"
    )
    noise = rng.choice([
        "Sir, students requested this change because of the placement drive.",
        "PFA the updated plan. Pls circulate.",
        "Forwarding as discussed in the faculty meeting.",
    ])
    body = (
        f"{rng.choice(['FYI.', 'Forwarding for your information.', 'Pls see below.'])}\n\n"
        f"{header}\n\n{noise}\n\n{sentence}\n\n{rng.choice(SIGNOFFS)}\nTA Head"
    )
    return {
        "id": email_id,
        "received_date": received.isoformat(),
        "sender": sender_email,
        "subject": subject,
        "body": body,
        "gold": [gold],
        "level": 5,
        "template_ids": [tid],
    }


def generate_sample(rng: random.Random, ctx: _Ctx, email_id: str, level: int) -> dict[str, Any]:
    """Generate a single sample at the given curriculum level."""
    if rng.random() < 0.15:
        return generate_no_event_email(rng, ctx, email_id, level)
    if level in (1, 2, 3):
        return generate_single_event_email(rng, ctx, email_id, level)
    if level == 4:
        return generate_distractor_email(rng, ctx, email_id)
    return generate_noisy_email(rng, ctx, email_id)


def generate_dataset(
    num_samples: int,
    seed: int = 42,
    levels: Optional[list[int]] = None,
    split: str = "train",
) -> list[dict[str, Any]]:
    """Generate a reproducible synthetic dataset with balanced levels.

    split="test" draws only from the held-out templates (plus test-only courses/venues).
    """
    rng = random.Random(f"{split}-{seed}")
    ctx = _Ctx(split)
    levels = levels or [1, 2, 3, 4, 5]
    return [
        generate_sample(rng, ctx, f"synth_{split}_{seed}_{i:05d}", levels[i % len(levels)])
        for i in range(num_samples)
    ]


def save_jsonl(records: list[dict[str, Any]], filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic email datasets for InboxRL.")
    parser.add_argument("--out-dir", type=str, default="data/synthetic")
    parser.add_argument("--train", type=int, default=2000)
    parser.add_argument("--val", type=int, default=200)
    parser.add_argument("--test", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    splits = {
        "train": generate_dataset(args.train, seed=args.seed, split="train"),
        "val": generate_dataset(args.val, seed=args.seed, split="val"),
        "test": generate_dataset(args.test, seed=args.seed, split="test"),
    }
    train_bodies = {r["body"] for r in splits["train"]}
    leaked = sum(r["body"] in train_bodies for r in splits["test"])
    if leaked:
        raise RuntimeError(f"{leaked} test emails are identical to train emails")

    for name, records in splits.items():
        save_jsonl(records, out_dir / f"{name}.jsonl")
        print(f"  {name:5s}: {len(records)} records -> {out_dir / f'{name}.jsonl'}")


if __name__ == "__main__":
    main()
