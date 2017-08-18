import numpy as np

from delfi.generator.BaseGenerator import GeneratorBase


class Default(GeneratorBase):
    @copy_ancestor_docstring
    def _feedback_proposed_param(self, param):
        # See BaseGenerator.py for docstring
        return 'accept'

    @copy_ancestor_docstring
    def _feedback_forward_model(self, data):
        # See BaseGenerator.py for docstring
        return 'accept'

    @copy_ancestor_docstring
    def _feedback_summary_stats(self, sum_stats):
        # See BaseGenerator.py for docstring
        return 'accept'
