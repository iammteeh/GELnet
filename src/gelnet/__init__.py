"""Python implementation of the GELnet algorithm.

This package provides a NumPy based translation of the Fortran
routines that power the *GLassoElnetFast* R package.  The functions
implemented here are a direct port of the routines required for the
block coordinate descent solver described in
"Graphical Elastic Net" (https://arxiv.org/pdf/2101.02148).
"""

from .algorithm import (
    GelnetLoopResult,
    GlassoElnetResult,
    connect,
    gelnet_loop1,
    glasso_elnet_fast,
)

from .solver import (
    CrossvalidationResult,
    DpgelnetResult,
    crossvalidation,
    dpgelnet,
    gelnet,
    rope,
    target,
)

__all__ = [
    "GelnetLoopResult",
    "GlassoElnetResult",
    "DpgelnetResult",
    "CrossvalidationResult",
    "gelnet_loop1",
    "glasso_elnet_fast",
    "dpgelnet",
    "gelnet",
    "connect",
    "rope",
    "target",
    "crossvalidation",
]