import os
import sys
import numpy as np

import graph_tool.all as gt
from graph_tool.inference.reconstruction import DynamicsBlockStateBase

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)
import libgene_expr_ext

class GeneExprBlockState(DynamicsBlockStateBase):
    r"""Network reconstruction block state for continuous-time Gene Regulatory
    Network (GRN) gene expression dynamics modeled by Euler-Maruyama SDE:
        dx_i(t)/dt = -\gamma_i x_i(t) + V_i \sigma(\sum_j W_{i,j} x_j(t)) + \sqrt{2 D_i} \xi_i(t)
    with Gaussian Onsager-Machlup likelihood (model.md & inference_article__method_breakdown.md).
    """
    def __init__(self, s, g=None, self_loops=True, dt=1.0, **kwargs):
        DynamicsBlockStateBase.__init__(
            self, s, g=g, self_loops=self_loops,
            **dict(dict(directed=True, dt=dt), **kwargs)
        )
        self.theta_first = False

    def _make_state(self):
        return libgene_expr_ext.make_gene_expr_state(
            self._state, self.ot, self.os, self.oactive, self.params
        )
