from src.utils import distance


def test_distance_zero():
    assert distance((0, 0), (0, 0)) == 0.0


def test_distance_positive():
    assert round(distance((0, 0), (3, 4)), 2) == 5.00