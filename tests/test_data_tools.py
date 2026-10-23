"""Tests for mbox ingestion, anonymisation and the .ics export."""

import mailbox
from email.message import EmailMessage

from icalendar import Calendar

from inboxrl.data.anonymize import anonymize_record, anonymize_text
from inboxrl.data.ingest_mbox import parse_mbox_file, received_date_ist
from inboxrl.demo.ics import events_to_ics
from inboxrl.schema import Event


def test_anonymize_masks_personal_data_but_keeps_offices():
    text = ("Contact Hemant Patel (B23CH1018, b23ch1018@iitj.ac.in, +91 98765 43210, hemant@gmail.com) "
            "or acad-office@iitj.ac.in. Quiz in LHC-101 for CSL3090.")
    out = anonymize_text(text, names=["Hemant Patel"])
    for leaked in ("Hemant", "B23CH1018", "b23ch1018", "98765", "gmail"):
        assert leaked not in out
    assert "acad-office@iitj.ac.in" in out
    assert "LHC-101" in out and "CSL3090" in out


def test_anonymize_sender_drops_display_name():
    rec = anonymize_record({"sender": "Some Student <b23ee1088@iitj.ac.in>", "subject": "", "body": ""})
    assert rec["sender"] == "[PERSON_EMAIL]"


def test_received_date_converted_to_ist():
    # 20:00 UTC on 4 Nov is already 5 Nov in India
    assert received_date_ist("Wed, 4 Nov 2026 20:00:00 +0000") == "2026-11-05"
    assert received_date_ist("not a date") is None


def test_parse_mbox(tmp_path):
    path = tmp_path / "inbox.mbox"
    box = mailbox.mbox(str(path))
    for frm, subj in [("Prof X <prof.x@iitj.ac.in>", "Quiz 2"), ("spam@shop.com", "Sale")]:
        msg = EmailMessage()
        msg["From"], msg["Subject"], msg["Date"] = frm, subj, "Mon, 2 Nov 2026 10:00:00 +0530"
        msg.set_content("Quiz 2 is on 5 Nov at 10 am in LHC-101.")
        box.add(msg)
    box.flush()
    records = parse_mbox_file(path)
    assert len(records) == 1
    assert records[0]["received_date"] == "2026-11-02"
    assert records[0]["sender"] == "[PERSON_EMAIL]"


def test_ics_uses_ist_and_stable_uids():
    ev = Event(title="CSL3090 Quiz 2", type="exam", date="2026-11-05", start_time="10:00", location="LHC-101")
    cal = Calendar.from_ical(events_to_ics([ev, ev]))
    vevents = cal.walk("VEVENT")
    assert vevents[0]["UID"] == vevents[1]["UID"]
    start = vevents[0].decoded("DTSTART")
    assert str(start.tzinfo) == "Asia/Kolkata" and start.hour == 10
    assert cal.walk("VTIMEZONE"), "a VTIMEZONE block should be embedded"
