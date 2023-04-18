"""Data sets registered with :py:class:`~dataoob.dataloader.register.Register`.

Data sets
=========
.. autosummary::
    :toctree: generated/

    datasets
    imagesets
    nlpsets

Catalog of registered data sets that can be used with
:py:class:`~dataoob.dataloader.fetcher.DataFetcher`. Pass in the ``str`` name
registering the data set to load the data set as needed.

NOTE :py:mod:`~dataoob.dataloader.datasets.imagesets` and
:py:class:`~dataoob.dataloader.datasets.nlpsets` have external dependencies,
run `make install-extra`.
"""
from dataoob.dataloader.datasets import datasets

try:
    from dataoob.dataloader.datasets import imagesets, nlpsets
except ImportError as e:
    print(
        f"Failed to import nlpsets or imagesets, likely optional dependency not found."
        f"Error message is as follows: {e}"
    )
