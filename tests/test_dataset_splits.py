"""A downloaded dataset keeps the splits its file names carry.

Handed a flat list of parquet files, HuggingFace concatenates them into a single
`train` split. For mnist that is 70,000 rows with the 10,000 test rows mixed
into the 60,000 to train on -- a mistake that shows up as an accuracy figure
that looks good rather than as an error.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corerun.datasets import _by_split


def test_the_huggingface_naming_convention_is_understood():
    grouped = _by_split([
        "/d/mnist-train-00000-of-00001.parquet",
        "/d/mnist-test-00000-of-00001.parquet",
    ])
    assert set(grouped) == {"train", "test"}
    assert grouped["train"] == ["/d/mnist-train-00000-of-00001.parquet"]
    assert grouped["test"] == ["/d/mnist-test-00000-of-00001.parquet"]


def test_shards_of_one_split_stay_together():
    grouped = _by_split([
        "/d/c4-train-00000-of-00002.parquet",
        "/d/c4-train-00001-of-00002.parquet",
        "/d/c4-validation-00000-of-00001.parquet",
    ])
    assert len(grouped["train"]) == 2
    assert len(grouped["validation"]) == 1


def test_a_file_naming_no_split_makes_the_whole_grouping_untrusted():
    """Guessing for some files and not others is the worst of both."""
    files = [
        "/d/mnist-train-00000-of-00001.parquet",
        "/d/extra.parquet",
    ]
    assert _by_split(files) == files


def test_a_dataset_with_no_splits_is_left_alone():
    files = ["/d/data.parquet", "/d/more.parquet"]
    assert _by_split(files) == files
