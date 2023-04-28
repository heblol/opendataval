import unittest
import warnings

import numpy as np
import torch

from opendataval.dataloader import DataFetcher, Register, mix_labels
from opendataval.experiment import ExperimentMediator, discover_corrupted_sample
from opendataval.model import Model
from opendataval.presets import dummy_evaluators
from opendataval.util import set_random_state


class DummyModel(Model):
    def fit(self, *args, **kwargs):
        return self

    def predict(self, x_train):
        return torch.rand((len(x_train), 1))


Register("test_dataset").from_numpy(np.array([[1, 2], [3, 4], [5, 6]]), 1)


class TestDataEvaluatorDryRun(unittest.TestCase):
    """Quick dry run to ensure all data evaluators are working as intended."""

    def test_dry_run(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # temporarily ignores warnings
            random_state = set_random_state(10)
            fetcher = (
                DataFetcher("iris", random_state=random_state)
                .split_dataset_by_count(3, 2, 2)
                .noisify(mix_labels, noise_rate=0.2)
            )

            # Checks that all evaluators in `dummy_evaluators` can have at least
            # a dry run with low data. Basically a sanity check.
            exper_med = ExperimentMediator(
                fetcher=fetcher,
                pred_model=DummyModel(),
                metric_name="accuracy",
            ).compute_data_values(data_evaluators=dummy_evaluators)

            exper_med.evaluate(discover_corrupted_sample)
