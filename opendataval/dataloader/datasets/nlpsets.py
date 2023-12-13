"""NLP data sets.

Uses HuggingFace
`transformers <https://huggingface.co/docs/transformers/index>`_. as dependency.
"""
from multiprocessing import pool
import os
from pathlib import Path
from typing import Callable, Sequence
from tqdm import tqdm
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from opendataval.dataloader.register import Register, cache
from opendataval.dataloader.util import FolderDataset, ListDataset
from opendataval.util import batched


def BertEmbeddings(
    func: Callable[[str, bool], tuple[Sequence[str], np.ndarray]], batch_size: int = 128
):
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
        from transformers import DistilBertModel, DistilBertTokenizerFast

        BERT_PRETRAINED_NAME = "distilbert-base-uncased"  # TODO update this

        force_download = True

        print("-" * 40)
        print("# Running BertEmbeddings")
        print("-" * 40)

        cache_dir = Path(cache_dir)
        # embed_path = cache_dir / f"{func.__name__}_embed"

        dataset, labels = func(cache_dir, force_download, *args, **kwargs)

        embed_file_name = f"{func.__name__}_{len(labels)}_embed.pt"
        embed_path = f"{cache_dir}/{func.__name__}_embed"

        print("This is the cache_dir", cache_dir)

        if os.path.exists(Path(f"{embed_path}/{embed_file_name}")):
            print(f"# Found Cached dataset!", embed_path)
            nlp_embeddings = torch.load(f"{embed_path}/{embed_file_name}")
            return nlp_embeddings, labels
            # return FolderDataset.load(embed_path), labels

        # Slow down on gpu vs cpu is quite substantial, uses gpu accel if available
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_PRETRAINED_NAME)
        bert_model = DistilBertModel.from_pretrained(BERT_PRETRAINED_NAME).to(device)
        folder_dataset = FolderDataset(embed_path)

        # Initialize an empty list to store pooled embeddings
        pooled_embeddings_list = []

        print("-" * 40)
        print(
            "# Need to tokenize and embedd all datapoints. This can take a while with large datasets."
        )
        print("-" * 40)
        for batch_num, batch in enumerate(tqdm(batched(dataset, n=batch_size))):
            bert_inputs = tokenizer.__call__(
                batch,
                max_length=200,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            bert_inputs = {inp: bert_inputs[inp] for inp in tokenizer.model_input_names}

            with torch.no_grad():
                pool_embed = bert_model(**bert_inputs)[0]
                word_embeddings = pool_embed.detach().cpu()[:, 0]

                # I added this line of code:
                pooled_embeddings_list.append(word_embeddings)

            # folder_dataset.write(batch_num, word_embeddings)

        # Concatenate the pooled embeddings from all batches
        pooled_embeddings = torch.cat(pooled_embeddings_list, dim=0)

        ### -- I created this, saving the model -- ###
        if not os.path.exists(cache_dir):
            print(
                f"""cache dir does not exist, creating it cache_dir={
                  cache_dir} """
            )
            os.mkdir(cache_dir)

        # Concatenate the pooled embeddings from all batches
        torch.save(pooled_embeddings.detach(), f"{embed_path}/{embed_file_name}")
        ### -- /END I created this, saving the model -- ###

        folder_dataset.save()
        print("-" * 40)
        print("# Finished BertEmbeddings wrapper.")
        print("-" * 40)

        return pooled_embeddings, np.array(labels)

    return wrapper


def BertEmbeddingsForSentenceTuple(
    func: Callable[[str, bool], tuple[Sequence[str], np.ndarray]],
    batch_size: int = 128,
):
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
        from transformers import DistilBertModel, DistilBertTokenizerFast

        BERT_PRETRAINED_NAME = "distilbert-base-uncased"  # TODO update this

        force_download = True

        print("-" * 40)
        print("# Running BertEmbeddings For Sentence Tuples")
        print("-" * 40)

        cache_dir = Path(cache_dir)
        # embed_path = cache_dir / f"{func.__name__}_embed"

        dataset, labels = func(cache_dir, force_download, *args, **kwargs)

        print("These are text1, text2", dataset)

        embed_file_name = f"{func.__name__}_{len(labels)}_embed.pt"
        embed_path = f"{cache_dir}/{func.__name__}_embed"

        print("This is the cache_dir", cache_dir)

        # if os.path.exists(Path(f"{embed_path}/{embed_file_name}")):
        #     print(f"# Found Cached dataset!", embed_path)
        #     nlp_embeddings = torch.load(f"{embed_path}/{embed_file_name}")
        #     return nlp_embeddings, labels
        #     # return FolderDataset.load(embed_path), labels

        # Slow down on gpu vs cpu is quite substantial, uses gpu accel if available
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_PRETRAINED_NAME)
        bert_model = DistilBertModel.from_pretrained(BERT_PRETRAINED_NAME).to(device)
        folder_dataset = FolderDataset(embed_path)

        # Initialize an empty list to store pooled embeddings
        pooled_embeddings_list = []

        print("SIZE DATASET", len(dataset))

        print("-" * 40)
        print(
            "# Need to tokenize and embedd all datapoints. This can take a while with large datasets."
        )
        print("-" * 40)
        for batch_num, batch in enumerate(tqdm(batched(dataset, n=batch_size))):
            t1, t2 = batch[0], batch[1]

            bert_inputs = tokenizer.__call__(
                t1,
                t2,
                max_length=200,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            bert_inputs = {inp: bert_inputs[inp] for inp in tokenizer.model_input_names}

            with torch.no_grad():
                pool_embed = bert_model(**bert_inputs)[0]
                word_embeddings = pool_embed.detach().cpu()[:, 0]

                # I added this line of code:
                pooled_embeddings_list.append(word_embeddings)

            # folder_dataset.write(batch_num, word_embeddings)

        # Concatenate the pooled embeddings from all batches
        pooled_embeddings = torch.cat(pooled_embeddings_list, dim=0)

        ### -- I created this, saving the model -- ###
        if not os.path.exists(cache_dir):
            print(
                f"""cache dir does not exist, creating it cache_dir={
                  cache_dir} """
            )
            os.mkdir(cache_dir)

        # Concatenate the pooled embeddings from all batches
        torch.save(pooled_embeddings.detach(), f"{embed_path}/{embed_file_name}")
        ### -- /END I created this, saving the model -- ###

        folder_dataset.save()
        print("-" * 40)
        print("# Finished BertEmbeddings wrapper.")
        print("-" * 40)

        return pooled_embeddings, np.array(labels)

    return wrapper


# def BertEmbeddings(func: Callable[[str, bool], tuple[ListDataset, np.ndarray]]):
#     """Convert text data into pooled embeddings with DistilBERT model.

#     Given a data set with a list of string, such as NLP data set function (see below),
#     converts the sentences into strings. It is the equivalent of training a downstream
#     task with bert but all the BERT layers are frozen. It is advised to just
#     train with the raw strings with a BERT model located in models/bert.py or defining
#     your own model. DistilBERT is just a faster version of BERT

#     References
#     ----------
#     .. [1] J. Devlin, M.W. Chang, K. Lee, and K. Toutanova,
#         BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
#         arXiv.org, 2018. Available: https://arxiv.org/abs/1810.04805.
#     .. [2] V. Sanh, L. Debut, J. Chaumond, and T. Wolf,
#         DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter
#         arXiv.org, 2019. Available: https://arxiv.org/abs/1910.01108.
#     """

#     def wrapper(
#         cache_dir: str, force_download: bool, *args, **kwargs
#     ) -> tuple[torch.Tensor, np.ndarray]:
#         from transformers import (
#             DistilBertModel,00
#             BertTokenizer,
#             BertForSequenceClassification,
#         )

#         print("-" * 30)
#         print(
#             "Calling BertEmbeddings wrapper. This can take a while, especially with large datasets."
#         )
#         print("-" * 30)

#         BERT_PRETRAINED_NAME = "bert-base-uncased"  # TODO update this

#         cache_dir = Path(cache_dir)

#         dataset, labels = func(cache_dir, force_download, *args, **kwargs)

#         print("-" * 30)
#         print("imported raw dataset")

#         subset = np.random.RandomState(10).permutation(len(dataset))

#         print("-" * 30)
#         print("creating subsets")

#         dataset_size = min(len(labels), MAX_DATASET_SIZE)

#         print("Dataset size: ", dataset_size)
#         if len(labels) > MAX_DATASET_SIZE:
#             warnings.warn(
#                 f"""Dataset size is larger than {
#                           MAX_DATASET_SIZE}, capping at MAX_DATASET_SIZE"""
#             )

#         print("-" * 30)
#         print("checking max size: ok")

#         embed_file_name = f"{func.__name__}_{dataset_size}_embed.pt"
#         embed_path = f"{cache_dir}/{embed_file_name}"

#         if os.path.exists(embed_path) and not force_download:
#             print(
#                 "Embedding path DOES exist. Loading dataset from cache! Do not reload everything."
#             )
#             nlp_embeddings = torch.load(embed_path)

#             print("These are the nlp_embeddings?", nlp_embeddings.shape)
#             print("EMBEDDING", nlp_embeddings)
#             return nlp_embeddings, labels[subset[: len(nlp_embeddings)]]

#         print("Embedding path does NOT exist.", embed_path)
#         labels = labels[subset[:dataset_size]]
#         entries = [entry for entry in dataset[subset[:dataset_size]]]

#         # Slow down on gpu vs cpu is quite substantial, uses gpu accel if available
#         device = torch.device(
#             "cuda"
#             if torch.cuda.is_available()
#             else "mps"
#             if torch.backends.mps.is_available()
#             else "cpu"
#         )

#         print("Just before creating tokenizer ")
#         tokenizer = BertTokenizer.from_pretrained(BERT_PRETRAINED_NAME)

#         print("Creating bert from_pretrained")
#         bert_model = BertForSequenceClassification.from_pretrained(
#             BERT_PRETRAINED_NAME
#         ).to(device)

#         print("-" * 10)
#         print("Calling tokenizer using", device)
#         print("-" * 10)

#         if device == "cpu":
#             raise Exception("Dont finetune Bert using CPU")
#         # res = tokenizer.__call__(
#         #     entries, max_length=200, padding=True, truncation=True, return_tensors="pt"
#         # ).to(device)

#         ########################################
#         # This was the original code
#         ########################################

#         # with torch.no_grad():
#         #     pooled_embeddings = (
#         #         (bert_model(res.input_ids, res.attention_mask)
#         #          [0]).detach().cpu()[:, 0]
#         #     )

#         ########################################
#         # This was the original code
#         ########################################

#         ######################################
#         # Trying to batch data
#         ######################################

#         # Dont batch, that shuffles the data

#         tokenized_data = tokenizer(
#             entries, max_length=200, padding=True, truncation=True, return_tensors="pt"
#         )
#         print("-" * 10)
#         print("Called tokenizer, and got a result")
#         print("-" * 10)

#         dataset = torch.utils.data.TensorDataset(
#             tokenized_data.input_ids, tokenized_data.attention_mask
#         )

#         dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

#         # Printing the first 20 elements of the dataset tensor
#         print(dataset[:20])

#         # Initialize an empty list to store pooled embeddings
#         pooled_embeddings_list = []

#         # Process data in batches
#         with torch.no_grad():
#             print("-" * 10)
#             print("With large datasets, this can take a while. ")
#             print("-" * 10)
#             for batch in tqdm.tqdm(dataloader):
#                 batch_input_ids, batch_attention_mask = batch
#                 batch_input_ids, batch_attention_mask = batch_input_ids.to(
#                     device
#                 ), batch_attention_mask.to(device)

#                 # Get pooled embeddings for the batch
#                 batch_pooled_embeddings = (
#                     bert_model(batch_input_ids, batch_attention_mask)[0]
#                     .detach()
#                     .cpu()[:, 0]
#                 )

#                 print("This is the batch")

#                 # Append batch_pooled_embeddings to the list
#                 pooled_embeddings_list.append(batch_pooled_embeddings)

#         # Concatenate the pooled embeddings from all batches
#         pooled_embeddings = torch.cat(pooled_embeddings_list, dim=0)

#         #######################################
#         # End of trying to batch data
#         #######################################

#         print("-" * 30)
#         print("Finished BertEmbeddings")
#         print("-" * 30)

#         if not os.path.exists(cache_dir):
#             print(
#                 f"""cache dir does not exist, creating it cache_dir={
#                   cache_dir} """
#             )
#             os.mkdir(cache_dir)

#         torch.save(pooled_embeddings.detach(), embed_path)
#         return pooled_embeddings, np.array(labels)

#     return wrapper


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
def download_bbc(cache_dir: Path, force_download: bool):
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
def download_imdb(cache_dir: Path, force_download: bool):
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
def download_imdb_illuminating(cache_dir: Path, force_download: bool):
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
    cache_dir: Path, force_download: bool
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
