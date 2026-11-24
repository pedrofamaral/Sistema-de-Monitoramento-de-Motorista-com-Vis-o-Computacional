import csv

import pytest

from src.decision.engine import DriverState
from src.decision.events import EventTracker
from src.interface.event_log import EventLogger

N, D, P, F = DriverState.NORMAL, DriverState.DROWSY, DriverState.PHONE, DriverState.FATIGUE


def test_event_has_start_end_and_duration():
    tracker = EventTracker()
    assert tracker.update([N], 0.0) == ([], [])
    started, _ = tracker.update([D], 1.0, details={D: "Olhos fechados há 2.0 s"})
    assert [e.state for e in started] == [D] and started[0].detail == "Olhos fechados há 2.0 s"
    assert tracker.update([D], 2.0) == ([], [])  # continua: não repete o início
    _, ended = tracker.update([N], 3.5)
    assert ended[0].start == 1.0 and ended[0].end == 3.5 and ended[0].duration == pytest.approx(2.5)


def test_simultaneous_events_are_independent():
    tracker = EventTracker()
    tracker.update([D], 0.0)
    started, ended = tracker.update([D, P], 1.0)
    assert [e.state for e in started] == [P] and ended == []
    started, ended = tracker.update([P], 2.0)
    assert started == [] and [e.state for e in ended] == [D]


def test_peak_is_worst_value():
    tracker = EventTracker()
    for t, ear, conf in [(0, 0.12, 0.6), (1, 0.08, 0.9), (2, None, 0.7), (3, 0.10, None)]:
        tracker.update([D, P], t, values={D: ear, P: conf})
    drowsy, phone = sorted(tracker.close(4.0), key=lambda e: e.state.value == "CELULAR DETECTADO")
    assert drowsy.peak == pytest.approx(0.08)  # EAR: o menor
    assert phone.peak == pytest.approx(0.9)    # confiança: a maior


def test_close_ends_open_events():
    tracker = EventTracker()
    tracker.update([F], 10.0)
    ended = tracker.close(25.0)
    assert ended[0].state == F and ended[0].duration == pytest.approx(15.0)
    assert tracker.close(30.0) == []


def test_logger_writes_header_once_and_one_row_per_event(tmp_path):
    path = tmp_path / "logs" / "events.csv"
    tracker = EventTracker()
    tracker.update([P], 1.0, values={P: 0.8}, details={P: "Celular com confiança 0.80"})
    _, ended = tracker.update([N], 2.5)

    for run in range(2):  # duas execuções acrescentam ao mesmo arquivo
        logger = EventLogger(path, source="video.mp4")
        logger.write(ended[0])
        logger.close()

    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["evento"] == "CELULAR DETECTADO"
    assert rows[0]["inicio_s"] == "1.00" and rows[0]["duracao_s"] == "1.50"
    assert rows[0]["valor"] == "0.800" and rows[0]["fonte"] == "video.mp4"
