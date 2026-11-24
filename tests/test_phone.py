from src.decision.engine import DriverState, decide
from src.detectors.phone import PhoneFilter, PhoneMonitor
from src.detectors.phone_yolo import Detection
from src.detectors.temporal import PersistenceFilter

FRAME = (720, 1280, 3)
UPRIGHT_PHONE = Detection(box=(600, 300, 680, 460), confidence=0.8)  # 80x160, altura/largura 2,0


def test_filter_keeps_upright_phone():
    kept, rejected = PhoneFilter().split([UPRIGHT_PHONE], FRAME)
    assert kept == [UPRIGHT_PHONE] and rejected == []


def test_filter_rejects_square_small_and_outside_roi():
    square = Detection(box=(100, 100, 200, 200), confidence=0.9)  # altura/largura 1,0
    small = Detection(box=(100, 100, 120, 140), confidence=0.9)   # 20 px < 6% de 720
    phone_filter = PhoneFilter(roi=(0.0, 0.0, 0.4, 1.0))          # só os 40% da esquerda
    kept, rejected = phone_filter.split([square, small, UPRIGHT_PHONE], FRAME)
    assert kept == [] and len(rejected) == 3


def test_aspect_filter_can_be_disabled():
    square = Detection(box=(100, 100, 200, 200), confidence=0.9)
    kept, _ = PhoneFilter(excluded_aspect=None).split([square], FRAME)
    assert kept == [square]


def test_monitor_confirms_phone_after_persistence():
    monitor = PhoneMonitor(PhoneFilter(), PersistenceFilter(min_seconds=1.0, gap_tolerance=0.5))
    statuses = [monitor.update([UPRIGHT_PHONE], FRAME, i / 30) for i in range(40)]
    assert not statuses[0].in_use and statuses[0].confidence == 0.8
    assert statuses[-1].in_use


def test_decide_allows_simultaneous_states():
    assert decide(drowsy=False, phone_in_use=False, fatigued=False) == [DriverState.NORMAL]
    assert decide(drowsy=True, phone_in_use=False, fatigued=False) == [DriverState.DROWSY]
    assert decide(drowsy=True, phone_in_use=True, fatigued=True) == [
        DriverState.DROWSY, DriverState.PHONE, DriverState.FATIGUE]
