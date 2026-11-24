from src.detectors.temporal import PersistenceFilter

FPS = 30


def feed(filt, signal, start=0.0):
    return [filt.update(active, start + i / FPS) for i, active in enumerate(signal)]


def test_confirms_only_after_min_duration():
    results = feed(PersistenceFilter(min_seconds=1.0), [True] * 45)
    assert not any(results[:30])  # até 0,97 s
    assert all(results[30:])      # a partir de 1,0 s


def test_short_gap_is_tolerated():
    filt = PersistenceFilter(min_seconds=1.0, gap_tolerance=0.5)
    signal = [True] * 15 + [False] * 9 + [True] * 15  # falha de 0,3 s no meio
    assert feed(filt, signal)[-1]


def test_long_gap_resets():
    filt = PersistenceFilter(min_seconds=1.0, gap_tolerance=0.5)
    signal = [True] * 25 + [False] * 30 + [True] * 10  # falha de 1 s
    assert not feed(filt, signal)[-1]
    assert filt.duration < 0.5


def test_without_tolerance_any_gap_resets():
    filt = PersistenceFilter(min_seconds=1.0)
    feed(filt, [True] * 25 + [False])
    assert filt.duration == 0.0
