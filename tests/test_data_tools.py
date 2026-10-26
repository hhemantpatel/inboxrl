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


# ---------------------------------------------------------------- v2 prompt / no-event oversampling ----

def test_v2_prompt_is_selectable_and_v1_unchanged():
    from inboxrl.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_V2, build_chat_messages

    v1 = build_chat_messages("2026-10-01", "a@iitj.ac.in", "s", "b")
    v2 = build_chat_messages("2026-10-01", "a@iitj.ac.in", "s", "b", version=2)
    assert v1[0]["content"] == SYSTEM_PROMPT
    assert v2[0]["content"] == SYSTEM_PROMPT_V2 and "[]" in SYSTEM_PROMPT_V2


def test_no_event_fraction_oversamples_and_default_is_unchanged():
    import random as _random

    from inboxrl.train_grpo import build_train_rows

    rows = [{"gold": [], "level": 1 + i % 2} for i in range(20)] + \
           [{"gold": [{"title": "t", "type": "exam", "date": "2026-10-12"}], "level": 1 + i % 2} for i in range(180)]
    base_cfg = {"use_curriculum": True, "curriculum_stage1_steps": 10, "curriculum_stage1_levels": [1, 2]}
    plain = build_train_rows(rows, [], base_cfg, 4, 50, seed=0)
    boosted = build_train_rows(rows, [], {**base_cfg, "no_event_fraction": 0.3}, 4, 50, seed=0)
    frac = lambda rs: sum(not r["gold"] for r in rs) / len(rs)
    assert len(plain) == len(boosted) == 200
    assert frac(plain) < 0.2 and 0.27 <= frac(boosted) <= 0.33


def test_adapter_settings_prefers_inboxrl_json(tmp_path):
    from inboxrl.evaluate import adapter_settings

    run = tmp_path / "run"
    (run / "adapter").mkdir(parents=True)
    assert adapter_settings(str(run / "adapter"))["prompt_version"] == 1  # old adapters: v1 defaults
    (run / "config.yaml").write_text("prompt_version: 2\nmax_prompt_tokens: 768\nmax_completion_length: 320\n")
    assert adapter_settings(str(run / "adapter")) == {"prompt_version": 2, "max_prompt_tokens": 768, "max_new_tokens": 320}
    (run / "adapter" / "inboxrl.json").write_text('{"prompt_version": 1, "max_prompt_tokens": 640}')
    assert adapter_settings(str(run / "adapter"))["prompt_version"] == 1
