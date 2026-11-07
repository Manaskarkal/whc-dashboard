"""exceptions.py - the project's own error types.

Callers can catch the specific one they care about (e.g. DuplicateError),
or catch WhcError to handle any expected problem in one place.
"""


class WhcError(Exception):
    """Base class for every expected error in this project."""


class ValidationError(WhcError):
    """The data given is not acceptable."""


class DuplicateError(WhcError):
    """The record already exists."""


class NotFoundError(WhcError):
    """The record does not exist."""