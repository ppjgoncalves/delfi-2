import numpy as np

from delfi.generator.BaseGenerator import BaseGenerator


class Default(BaseGenerator):
    @copy_ancestor_docstring
    def _feedback_proposed_param(self, param):
        # See BaseGenerator for docstring
        return 'accept'

    @copy_ancestor_docstring
    def _feedback_forward_model(self, data):
        # See BaseGenerator for docstring
        return 'accept'

    @copy_ancestor_docstring
    def _feedback_summary_stats(self, sum_stats):
        # See BaseGenerator for docstring
        return 'accept'
