"""ftea: reframe landscape video to vertical and encode it as adaptive HLS.

Deliberately free of imports. The reframe stage's interpreter imports
this package straight from the checkout without installing it, and the
encoder's interpreter must never load the reframe stage's heavy
dependencies.
"""

__version__ = "0.1.0"
