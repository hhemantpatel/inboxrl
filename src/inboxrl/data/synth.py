"""Synthetic institute email generator with curriculum difficulty levels."""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

COURSES = [
    ("CSL3090", "Artificial Intelligence"),
    ("CSL3010", "Operating Systems"),
    ("CSL3020", "Computer Networks"),
    ("CSL7020", "Deep Reinforcement Learning"),
    ("EEL2010", "Signals and Systems"),
    ("MEL2010", "Fluid Mechanics"),
    ("CYL1010", "Engineering Chemistry"),
    ("MAL2010", "Linear Algebra and Optimization"),
]

INSTRUCTORS = [
    ("Prof. Anand Sharma", "anand.sharma@iitj.ac.in"),
    ("Prof. Ritu Verma", "ritu.verma@iitj.ac.in"),
    ("Dr. Vikram Singh", "vikram.singh@iitj.ac.in"),
    ("Dr. Neha Gupta", "neha.gupta@iitj.ac.in"),
    ("Academic Office", "acad-office@iitj.ac.in"),
    ("CDC Office", "cdc@iitj.ac.in"),
    ("Student Gymkhana", "gymkhana@iitj.ac.in"),
]

VENUES = [
    "LHC-101",
    "LHC-102",
    "LHC-201",
    "Seminar Hall 1",
    "Dept Conference Room",
    "Lab 214",
    "Online (Google Meet)",
    "Auditorium",
]

MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

SHORT_MONTHS = [
    "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _random_date(rng: random.Random, start_year: int = 2026) -> date:
    """Generate a random received date during the semester."""
    start = date(start_year, 8, 1)
    days_range = 100
    return start + timedelta(days=rng.randint(0, days_range))


def _format_date_l1(d: date) -> str:
    """Explicit standard date string."""
    return f"{d.day} {MONTH_NAMES[d.month]} {d.year}"


def _format_date_l2(d: date, rng: random.Random) -> str:
    """Varied date formats."""
    fmt_choice = rng.randint(1, 4)
    if fmt_choice == 1:
        suffix = "th" if 11 <= d.day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(d.day % 10, "th")
        return f"{SHORT_MONTHS[d.month]} {d.day}{suffix}"
    elif fmt_choice == 2:
        return f"{d.day:02d}/{d.month:02d}/{d.year}"
    elif fmt_choice == 3:
        return f"{d.year}-{d.month:02d}-{d.day:02d}"
    else:
        return f"{d.day} {SHORT_MONTHS[d.month]}"


def _resolve_relative_date(base: date, expr: str) -> date:
    """Resolve relative date expression relative to base date."""
    base_weekday = base.weekday()  # 0 is Monday, 6 is Sunday
    if expr == "tomorrow":
        return base + timedelta(days=1)
    elif expr == "day after tomorrow":
        return base + timedelta(days=2)
    elif expr.startswith("this "):
        target_day_name = expr.replace("this ", "").strip().capitalize()
        target_weekday = DAY_NAMES.index(target_day_name)
        diff = (target_weekday - base_weekday) % 7
        diff = 7 if diff == 0 else diff
        return base + timedelta(days=diff)
    elif expr.startswith("next "):
        target_day_name = expr.replace("next ", "").strip().capitalize()
        target_weekday = DAY_NAMES.index(target_day_name)
        diff = (target_weekday - base_weekday) % 7
        return base + timedelta(days=diff + 7)
    return base + timedelta(days=3)


def generate_no_event_email(rng: random.Random, email_id: str, level: int) -> dict[str, Any]:
    """Generate an email that contains informational content but no scheduled event."""
    received = _random_date(rng)
    course_code, course_name = rng.choice(COURSES)
    sender_name, sender_email = rng.choice(INSTRUCTORS)

    templates = [
        (
            f"[{course_code}] Marks Published for Assignment 1",
            f"Dear Students,\n\nThe evaluation for Assignment 1 has been completed. The marks and detailed feedback have been uploaded to the portal.\n\nPlease check your scores and report any discrepancies to the TAs during normal lab hours.\n\nBest regards,\n{sender_name}",
        ),
        (
            "Notice: Library Book Return Advisory",
            "Dear Institute Community,\n\nAll students are advised to check their library account for any overdue books. Late fines may be levied if overdue items are not returned promptly.\n\nLibrarian,\nCentral Library",
        ),
        (
            f"[{course_code}] Reference Reading Material Uploaded",
            f"Dear All,\n\nI have uploaded supplementary reading material on dynamic programming and policy iteration to the LMS drive.\n\nStudents are encouraged to review these lecture notes ahead of upcoming sessions.\n\nRegards,\n{sender_name}",
        ),
        (
            "Campus WiFi Maintenance Completed",
            "Dear Residents,\n\nThe scheduled maintenance on the core campus network switches in Academic Block 2 has been completed successfully. All services have been restored.\n\nComputer Center Team",
        ),
        (
            "Notice: Mess Menu Revision Feedback",
            "Dear Hostellers,\n\nThe Mess Committee invites suggestions regarding the proposed menu revisions for the upcoming month. Google form link is shared on the hostel portal.\n\nMess Committee Secretary",
        ),
    ]

    subject, body = rng.choice(templates)
    return {
        "id": email_id,
        "received_date": received.isoformat(),
        "sender": sender_email,
        "subject": subject,
        "body": body,
        "gold": [],
        "level": level,
    }


def generate_single_event_email(
    rng: random.Random, email_id: str, level: int
) -> dict[str, Any]:
    """Generate an email with exactly one event, tuned to difficulty level."""
    received = _random_date(rng)
    course_code, course_name = rng.choice(COURSES)
    sender_name, sender_email = rng.choice(INSTRUCTORS)
    venue = rng.choice(VENUES)

    # Pick category
    cat = rng.choice(["deadline", "exam", "class", "event", "meeting"])

    offset_days = rng.randint(1, 14)
    target_date = received + timedelta(days=offset_days)

    # Determine time
    time_choice = rng.choice(["23:59", "17:00", "10:00", "14:30", "18:00", None])
    
    # Text representation of date & time according to level
    if level == 1:
        date_str = _format_date_l1(target_date)
        time_str = f"at {time_choice}" if time_choice else "by end of day"
        if time_choice == "23:59":
            time_str = "at 11:59 PM"
    elif level == 2:
        date_str = _format_date_l2(target_date, rng)
        if time_choice == "23:59":
            time_str = rng.choice(["by midnight", "before 23:59", "by 11:59 pm", "EOD"])
        elif time_choice == "17:00":
            time_str = rng.choice(["by 5 pm", "at 17:00", "by 5:00 PM"])
        elif time_choice is None:
            time_str = ""
        else:
            time_str = f"at {time_choice}"
    elif level == 3:
        # Relative date
        if offset_days == 1:
            rel_phrase = "tomorrow"
            target_date = received + timedelta(days=1)
        elif offset_days == 2:
            rel_phrase = "day after tomorrow"
            target_date = received + timedelta(days=2)
        else:
            day_name = DAY_NAMES[(received.weekday() + offset_days) % 7]
            prefix = "this" if offset_days <= 6 else "next"
            rel_phrase = f"{prefix} {day_name}"
            target_date = _resolve_relative_date(received, rel_phrase)
        date_str = rel_phrase
        time_str = f"at {time_choice}" if time_choice else ""
    else:
        # L4 or L5 single event
        date_str = _format_date_l1(target_date)
        time_str = f"at {time_choice}" if time_choice else ""

    # Construct content based on category
    if cat == "deadline":
        title = f"{course_code} Assignment Submission"
        subject = f"[{course_code}] Assignment Submission Deadline"
        body = (
            f"Dear Students,\n\nPlease remember that the submission for Assignment is due on "
            f"{date_str} {time_str}. Kindly upload your code and report to the portal on time.\n\n"
            f"Late submissions will carry a penalty.\n\nRegards,\n{sender_name}"
        )
        location = None
    elif cat == "exam":
        title = f"{course_code} Mid-Semester Exam"
        subject = f"[{course_code}] Mid-Semester Examination Schedule"
        body = (
            f"Dear Class,\n\nThe mid-semester examination for {course_name} ({course_code}) is scheduled "
            f"on {date_str} {time_str}. Venue: {venue}.\n\nPlease be seated 10 minutes prior.\n\n"
            f"Best wishes,\n{sender_name}"
        )
        location = venue
    elif cat == "class":
        title = f"{course_code} Compensatory Lecture"
        subject = f"[{course_code}] Compensatory Class Announcement"
        body = (
            f"Dear All,\n\nWe will have an extra compensatory lecture for {course_code} on "
            f"{date_str} {time_str} at {venue}.\n\nAttendance is mandatory.\n\nRegards,\n{sender_name}"
        )
        location = venue
    elif cat == "meeting":
        title = f"{course_code} TA Office Hours"
        subject = f"[{course_code}] Doubt Clearing Session / TA Office Hours"
        body = (
            f"Hello students,\n\nThe TAs will hold an office hours doubt clearing session on "
            f"{date_str} {time_str} at {venue}. Please bring your assignment queries.\n\nBest,\nTeaching Assistants"
        )
        location = venue
    else:  # event
        title = f"Distinguished Lecture on {course_name}"
        subject = f"Distinguished Lecture Series: {course_name}"
        body = (
            f"Dear Institute Members,\n\nYou are cordially invited to a guest talk on modern advances in {course_name}. "
            f"The seminar will take place on {date_str} {time_str} in {venue}.\n\nAll are welcome.\n\nAcademic Office"
        )
        location = venue

    gold_event = {
        "title": title,
        "type": cat,
        "date": target_date.isoformat(),
        "start_time": time_choice,
        "end_time": None,
        "location": location,
    }

    return {
        "id": email_id,
        "received_date": received.isoformat(),
        "sender": sender_email,
        "subject": subject,
        "body": body,
        "gold": [gold_event],
        "level": level,
    }


def generate_complex_email(
    rng: random.Random, email_id: str, level: int
) -> dict[str, Any]:
    """Generate complex emails with distractors, multiple events, or noisy headers."""
    received = _random_date(rng)
    course_code, course_name = rng.choice(COURSES)
    sender_name, sender_email = rng.choice(INSTRUCTORS)
    venue = rng.choice(VENUES)

    if level == 4:
        # Multi-event or Distractor
        scenario = rng.choice(["extension", "two_events"])
        if scenario == "extension":
            old_date = received + timedelta(days=2)
            new_date = received + timedelta(days=7)
            title = f"{course_code} Project Submission (Extended)"
            subject = f"[{course_code}] Deadline Extension: Course Project"
            body = (
                f"Dear Students,\n\nBased on several student requests, the deadline for the course project, "
                f"which was originally set for {_format_date_l1(old_date)} at 23:59, has now been extended to "
                f"{_format_date_l1(new_date)} at 23:59. No further extensions will be granted.\n\n"
                f"Please ensure timely submission on the portal.\n\nRegards,\n{sender_name}"
            )
            gold = [
                {
                    "title": title,
                    "type": "deadline",
                    "date": new_date.isoformat(),
                    "start_time": "23:59",
                    "end_time": None,
                    "location": None,
                }
            ]
        else:
            # Two events
            date1 = received + timedelta(days=4)
            date2 = received + timedelta(days=8)
            subject = f"[{course_code}] Exam Schedule and Review Session"
            body = (
                f"Dear All,\n\nPlease find the schedule for {course_code} upcoming activities:\n"
                f"1. Doubt clearing review session: {_format_date_l1(date1)} at 16:00 in {venue}.\n"
                f"2. Quiz 2 Examination: {_format_date_l1(date2)} at 10:00 in {rng.choice(VENUES)}.\n\n"
                f"Attendance in the quiz is mandatory.\n\nBest regards,\n{sender_name}"
            )
            gold = [
                {
                    "title": f"{course_code} Doubt Clearing Review Session",
                    "type": "meeting",
                    "date": date1.isoformat(),
                    "start_time": "16:00",
                    "end_time": None,
                    "location": venue,
                },
                {
                    "title": f"{course_code} Quiz 2 Examination",
                    "type": "exam",
                    "date": date2.isoformat(),
                    "start_time": "10:00",
                    "end_time": None,
                    "location": venue,
                },
            ]
    else:
        # Level 5: Forwarded thread, noisy headers, subtle date references
        target_date = received + timedelta(days=5)
        venue5 = rng.choice(VENUES)
        subject = f"Fwd: Urgent: [{course_code}] Lab Viva Rescheduling"
        body = (
            f"---------- Forwarded message ---------\n"
            f"From: TA Head <ta.head@iitj.ac.in>\n"
            f"Date: {_format_date_l1(received)} at 09:15 AM\n"
            f"Subject: [{course_code}] Lab Viva Rescheduling\n"
            f"To: {sender_name} <{sender_email}>\n\n"
            f"Respected Sir,\n"
            f"Due to the upcoming placement drive, students requested shifting the lab evaluation.\n\n"
            f"----------\n"
            f"Dear Students,\n\nForwarding the TA's note. As approved, the {course_code} Lab Viva will be held on "
            f"{_format_date_l1(target_date)} at 14:00 at {venue5}. Kindly bring your printed lab records.\n\n"
            f"Best regards,\n{sender_name}"
        )
        gold = [
            {
                "title": f"{course_code} Lab Viva",
                "type": "exam",
                "date": target_date.isoformat(),
                "start_time": "14:00",
                "end_time": None,
                "location": venue5,
            }
        ]

    return {
        "id": email_id,
        "received_date": received.isoformat(),
        "sender": sender_email,
        "subject": subject,
        "body": body,
        "gold": gold,
        "level": level,
    }


def generate_sample(rng: random.Random, email_id: str, level: int) -> dict[str, Any]:
    """Generate a single sample based on curriculum level."""
    # 15% probability of a no-event email across all levels
    if rng.random() < 0.15:
        return generate_no_event_email(rng, email_id, level)

    if level in (1, 2, 3):
        return generate_single_event_email(rng, email_id, level)
    else:
        return generate_complex_email(rng, email_id, level)


def generate_dataset(
    num_samples: int,
    seed: int = 42,
    levels: Optional[list[int]] = None,
) -> list[dict[str, Any]]:
    """Generate a reproducible synthetic dataset with balanced levels."""
    rng = random.Random(seed)
    if levels is None:
        levels = [1, 2, 3, 4, 5]

    dataset = []
    for i in range(num_samples):
        level = levels[i % len(levels)]
        sample_id = f"synth_{seed}_{i:05d}"
        sample = generate_sample(rng, sample_id, level)
        dataset.append(sample)

    return dataset


def save_jsonl(records: list[dict[str, Any]], filepath: Path) -> None:
    """Save records to a JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic email datasets for InboxRL.")
    parser.add_argument("--out-dir", type=str, default="data/synthetic")
    parser.add_argument("--train", type=int, default=2000, help="Number of train samples")
    parser.add_argument("--val", type=int, default=200, help="Number of val samples")
    parser.add_argument("--test", type=int, default=300, help="Number of test samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    print(f"Generating synthetic datasets in {out_dir}...")

    train_data = generate_dataset(args.train, seed=args.seed)
    val_data = generate_dataset(args.val, seed=args.seed + 1)
    test_data = generate_dataset(args.test, seed=args.seed + 2)

    save_jsonl(train_data, out_dir / "train.jsonl")
    save_jsonl(val_data, out_dir / "val.jsonl")
    save_jsonl(test_data, out_dir / "test.jsonl")

    print(f"Successfully generated:")
    print(f"  - Train: {len(train_data)} records -> {out_dir / 'train.jsonl'}")
    print(f"  - Val:   {len(val_data)} records -> {out_dir / 'val.jsonl'}")
    print(f"  - Test:  {len(test_data)} records -> {out_dir / 'test.jsonl'}")


if __name__ == "__main__":
    main()
