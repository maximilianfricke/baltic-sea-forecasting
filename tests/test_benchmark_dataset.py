"""Tests for benchmark-only dataset lengthening."""

# First-party
from neural_lam.weather_dataset import BenchmarkRepeatDataset


class _TinyDataset:
    def __init__(self):
        self.epochs = []

    def __len__(self):
        return 3

    def __getitem__(self, index):
        return index

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


def test_benchmark_repeat_dataset_repeats_samples_without_changing_them():
    dataset = _TinyDataset()
    repeated = BenchmarkRepeatDataset(dataset, length=8)

    assert len(repeated) == 8
    assert [repeated[index] for index in range(8)] == [
        0,
        1,
        2,
        0,
        1,
        2,
        0,
        1,
    ]

    repeated.set_epoch(4)
    assert dataset.epochs == [4]
