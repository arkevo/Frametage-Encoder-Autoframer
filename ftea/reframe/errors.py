"""The auto-reframe stage's domain exception."""


class ReframeError(Exception):
    """Raised when a reframe stage cannot complete.

    Every reframe module raises this rather than a bare RuntimeError, so
    ftea/reframe/main.py can distinguish an expected domain failure
    (reported cleanly, exit 1) from an unexpected crash (also exit 1, but
    with the traceback preserved on stderr for diagnosis).
    """
