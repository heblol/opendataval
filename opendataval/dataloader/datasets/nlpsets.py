"""NLP data sets.

Uses HuggingFace
`transformers <https://huggingface.co/docs/transformers/index>`_. as dependency.
"""
from multiprocessing import pool
import os
from pathlib import Path
from typing import Callable
import tqdm
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from opendataval.dataloader.register import Register, cache
from opendataval.dataloader.util import ListDataset

MAX_DATASET_SIZE = 2000000
"""Data Valuation algorithms can take a long time for large data sets, thus cap size."""


def BertEmbeddings(func: Callable[[str, bool], tuple[ListDataset, np.ndarray]]):
    """Convert text data into pooled embeddings with DistilBERT model.

    Given a data set with a list of string, such as NLP data set function (see below),
    converts the sentences into strings. It is the equivalent of training a downstream
    task with bert but all the BERT layers are frozen. It is advised to just
    train with the raw strings with a BERT model located in models/bert.py or defining
    your own model. DistilBERT is just a faster version of BERT

    References
    ----------
    .. [1] J. Devlin, M.W. Chang, K. Lee, and K. Toutanova,
        BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
        arXiv.org, 2018. Available: https://arxiv.org/abs/1810.04805.
    .. [2] V. Sanh, L. Debut, J. Chaumond, and T. Wolf,
        DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter
        arXiv.org, 2019. Available: https://arxiv.org/abs/1910.01108.
    """

    def wrapper(
        cache_dir: str, force_download: bool, *args, **kwargs
    ) -> tuple[torch.Tensor, np.ndarray]:
        from transformers import DistilBertModel, BertTokenizer, BertForSequenceClassification

        print("-" * 30)
        print(
            "Calling BertEmbeddings wrapper. This can take a while, especially with large datasets."
        )
        print("-" * 30)

        BERT_PRETRAINED_NAME = "bert-base-uncased"  # TODO update this

        cache_dir = Path(cache_dir)

        dataset, labels = func(cache_dir, force_download, *args, **kwargs)

        print("-" * 30)
        print("imported raw dataset")

        subset = np.random.RandomState(10).permutation(len(dataset))

        print("-" * 30)
        print("creating subsets")

        dataset_size = min(len(labels), MAX_DATASET_SIZE)

        print("Dataset size: ", dataset_size)
        if len(labels) > MAX_DATASET_SIZE:
            warnings.warn(
                f"""Dataset size is larger than {
                          MAX_DATASET_SIZE}, capping at MAX_DATASET_SIZE"""
            )

        print("-" * 30)
        print("checking max size: ok")

        embed_file_name = f"{func.__name__}_{dataset_size}_embed.pt"
        embed_path = f"{cache_dir}/{embed_file_name}"

        if os.path.exists(embed_path):
            print(
                "Embedding path DOES exist. Loading dataset from cache! Do not reload everything."
            )
            nlp_embeddings = torch.load(embed_path)
            return nlp_embeddings, labels[subset[: len(nlp_embeddings)]]

        print("Embedding path does NOT exist.", embed_path)
        labels = labels[subset[:dataset_size]]
        entries = [entry for entry in dataset[subset[:dataset_size]]]

        # Slow down on gpu vs cpu is quite substantial, uses gpu accel if available
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        print("Just before creating tokenizer ")
        tokenizer = BertTokenizer.from_pretrained(BERT_PRETRAINED_NAME)

        print("Creating bert from_pretrained")
        bert_model = BertForSequenceClassification.from_pretrained(BERT_PRETRAINED_NAME).to(device)

        print("-" * 10)
        print("Calling tokenizer")
        print("-" * 10)
        res = tokenizer.__call__(
            entries, max_length=200, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        print("-" * 10)
        print("Called tokenizer, and got a result")
        print("-" * 10)

        ########################################
        # This was the original code
        ########################################

        # with torch.no_grad():
        #     pooled_embeddings = (
        #         (bert_model(res.input_ids, res.attention_mask)
        #          [0]).detach().cpu()[:, 0]
        #     )

        ########################################
        # This was the original code
        ########################################

        ######################################
        # Trying to batch data
        ######################################

        # Dont batch, that shuffles the data

        print("-" * 10)
        print("Calling tokenizer")
        tokenized_data = tokenizer(
            entries, max_length=200, padding=True, truncation=True, return_tensors="pt"
        )

        dataset = torch.utils.data.TensorDataset(
            tokenized_data.input_ids, tokenized_data.attention_mask
        )

        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

        # Printing the first 20 elements of the dataset tensor
        print(dataset[:20])

        # Initialize an empty list to store pooled embeddings
        pooled_embeddings_list = []

        # Process data in batches
        with torch.no_grad():
            print("-" * 10)
            print("With large datasets, this can take a while. ")
            print("-" * 10)
            for batch in tqdm.tqdm(dataloader):
                batch_input_ids, batch_attention_mask = batch
                batch_input_ids, batch_attention_mask = batch_input_ids.to(
                    device
                ), batch_attention_mask.to(device)

                # Get pooled embeddings for the batch
                batch_pooled_embeddings = (
                    bert_model(batch_input_ids, batch_attention_mask)[0]
                    .detach()
                    .cpu()[:, 0]
                )

                # Append batch_pooled_embeddings to the list
                pooled_embeddings_list.append(batch_pooled_embeddings)

        # Concatenate the pooled embeddings from all batches
        pooled_embeddings = torch.cat(pooled_embeddings_list, dim=0)

        #######################################
        # End of trying to batch data
        #######################################

        print("-" * 30)
        print("Finished BertEmbeddings")
        print("-" * 30)

        if not os.path.exists(cache_dir):
            print(
                f"""cache dir does not exist, creating it cache_dir={
                  cache_dir} """
            )
            os.mkdir(cache_dir)

        torch.save(pooled_embeddings.detach(), embed_path)
        return pooled_embeddings, np.array(labels)

    return wrapper


# def pooled_embeddings_batched(
#     bert_model: DistilBertModel, input_ids, attention_mask, tokenizer
# ):
#     batch_size = 8
#     # Tokenize and create DataLoader
#     tokenized_data = tokenizer(
#         entries, max_length=200, padding=True, truncation=True, return_tensors="pt"
#     )
#     dataset = torch.utils.data.TensorDataset(
#         tokenized_data.input_ids, tokenized_data.attention_mask
#     )
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

#     # Initialize an empty list to store pooled embeddings
#     pooled_embeddings_list = []

#     with torch.no_grad():
#         pooled_embeddings = (
#             (bert_model(input_ids, attention_mask)[0]).detach().cpu()[:, 0]
#         )

#     return pooled_embeddings


@Register("bbc", cacheable=True, one_hot=True)
def download_bbc(cache_dir: str, force_download: bool):
    """Classification data set registered as ``"bbc"``.

    Predicts type of article from the article. Used in NLP data valuation tasks.

    References
    ----------
    .. [1] D. Greene and P. Cunningham,
        Practical Solutions to the Problem of Diagonal Dominance in
        Kernel Document Clustering, Proc. ICML 2006.
    """
    github_url = (
        "https://raw.githubusercontent.com/"
        "mdsohaib/BBC-News-Classification/master/bbc-text.csv"
    )
    filepath = cache(github_url, cache_dir, "bbc-text.csv", force_download)
    df = pd.read_csv(filepath)

    label_dict = {
        "business": 0,
        "entertainment": 1,
        "sport": 2,
        "tech": 3,
        "politics": 4,
    }
    labels = np.fromiter((label_dict[label] for label in df["category"]), dtype=int)

    return ListDataset(df["text"].values), labels


@Register("imdb", cacheable=True, one_hot=True)
def download_imdb(cache_dir: str, force_download: bool):
    """Binary category sentiment analysis data set registered as ``"imdb"``.

    Predicts sentiment analysis of the review as either positive (1) or negative (0).
    Used in NLP data valuation tasks.

    References
    ----------
    .. [1] A. Maas, R. Daly, P. Pham, D. Huang, A. Ng, and C. Potts.
        Learning Word Vectors for Sentiment Analysis.
        The 49th Annual Meeting of the Association for Computational Linguistics (2011).
    """
    github_url = (
        "https://raw.githubusercontent.com/"
        "Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"
    )
    filepath = cache(github_url, cache_dir, "imdb.csv", force_download)
    df = pd.read_csv(filepath)

    label_dict = {"negative": 0, "positive": 1}
    labels = np.fromiter((label_dict[label] for label in df["sentiment"]), dtype=int)

    return ListDataset(df["review"].values), labels


@Register("illuminating", cacheable=True, one_hot=True)
def download_imdb_illuminating(cache_dir: str, force_download: bool):
    """

    Paper of Philip: Illuminating blindspots

    """

    filepath = "opendataval/data_files/illuminating_blindspots/hypo-llm-imdb-bert-original-dwb-gpt-generation-response-filtered-gpt-generation.csv"

    df = pd.read_csv(filepath)

    print("This is the illuminating d", df)

    label_dict = {0: 0, 1: 1}
    labels = np.fromiter((label_dict[label] for label in df["label"]), dtype=int)

    return ListDataset(df["text"].values), labels


@Register("illuminating-original-synthetic-combined", cacheable=True, one_hot=True)
def download_imdb_illuminating_original_synthetic_combined(
    cache_dir: str, force_download: bool
):
    """

    Paper of Philip: Illuminating blindspots

    """

    filepath = (
        "opendataval/data_files/illuminating_blindspots/original_synthetic_combined.csv"
    )

    df = pd.read_csv(filepath)

    print("dataset head: ", df.head(), df.shape)

    label_dict = {0: 0, 1: 1}
    labels = np.fromiter((label_dict[label] for label in df["label"]), dtype=int)

    return ListDataset(df["text"].values), labels


@Register("illuminating-original-synthetic-combined_2000", cacheable=True, one_hot=True)
def download_imdb_illuminating_original_synthetic_combined_2000(
    cache_dir: str, force_download: bool
):
    """

    Paper of Philip: Illuminating blind spots

    """

    filepath = "opendataval/data_files/illuminating_blindspots/original_synthetic_combined_2000.csv"

    df = pd.read_csv(filepath)

    print("dataset head: ", df.head())

    label_dict = {0: 0, 1: 1}
    labels = np.fromiter((label_dict[label] for label in df["label"]), dtype=int)

    return ListDataset(df["text"].values), labels


@Register(
    "download_imdb_illuminating_original_synthetic_combined_2000_v2",
    cacheable=False,
    one_hot=True,
)
def download_imdb_illuminating_original_synthetic_combined_2000_v2(
    cache_dir: str = None, force_download: bool = True
):
    """

    Paper of Philip: Illuminating blind spots

    """

    filepath = "opendataval/data_files/illuminating_blindspots/original_synthetic_combined_2000.csv"

    df = pd.read_csv(filepath)

    print("dataset head: ", df.head())

    label_dict = {0: 0, 1: 1}
    labels = np.fromiter((label_dict[label] for label in df["label"]), dtype=int)

    return ListDataset(df["text"].values), labels


illuminating_original_synthetic_combined_embeddings_2000_v2_bert = Register(
    "illuminating_original_synthetic_combined_embeddings_2000_v2_bert", True, True
)(BertEmbeddings(download_imdb_illuminating_original_synthetic_combined_2000_v2))
"""Classification data set registered as ``"illuminating_original_synthetic_combined_embeddings_2000_v2"``, BERT text embeddings."""


illuminating_original_synthetic_combined_embedding = Register(
    "illuminating_original_synthetic_combined_embeddings", True, True
)(BertEmbeddings(download_imdb_illuminating_original_synthetic_combined))
"""Classification data set registered as ``"illuminating_original_synthetic_combined_embeddings"``, BERT text embeddings."""

illuminating_original_synthetic_combined_embeddings_2000 = Register(
    "illuminating_original_synthetic_combined_embeddings_2000", True, True
)(BertEmbeddings(download_imdb_illuminating_original_synthetic_combined_2000))
"""Classification data set registered as ``"illuminating_original_synthetic_combined_embeddings_2000"``, BERT text embeddings."""

illuminating_embedding = Register("illuminating-embeddings", True, True)(
    BertEmbeddings(download_imdb_illuminating)
)
"""Classification data set registered as ``"illuminating-embeddings"``, BERT text embeddings."""

bbc_embedding = Register("bbc-embeddings", True, True)(BertEmbeddings(download_bbc))
"""Classification data set registered as ``"bbc-embeddings"``, BERT text embeddings."""

imdb_embedding = Register("imdb-embeddings", True, True)(BertEmbeddings(download_imdb))
"""Classification data set registered as ``"imdb-embeddings"``, BERT text embeddings."""
