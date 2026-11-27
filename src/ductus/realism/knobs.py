"""Switches for realism experiments. The blind-test loop (scripts/blind_iter.py) sets these to
score a candidate change; a change that wins becomes the default value here."""
K: dict = {}


def get(name: str, default):
    return K.get(name, default)
