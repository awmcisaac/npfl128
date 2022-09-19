#!/usr/bin/env python3
# Andrew McIsaac

import argparse
import pandas as pd
import numpy as np
import fasttext

from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import f1_score
from scipy.sparse import coo_matrix, hstack

from pathlib import Path
from typing import Tuple, Dict, Union
from numpy.typing import ArrayLike
from pandas._typing import FilePath

parser = argparse.ArgumentParser()

parser.add_argument("train", type=Path, help="path to training data")
parser.add_argument("test", type=Path, help="path to test data")

parser.add_argument(
    "--classifier",
    default="svm",
    choices=["svm", "ft"],
    help="choose classifier to use, either Linear Kernel SVM or FastText")

parser.add_argument(
    "--target",
    default=False,
    action="store_true",
    help="include target of interest presence/absence as a feature. \
          Default is False")
parser.add_argument(
    "--we",
    default=False,
    action="store_true",
    help="include FastText word embeddings as additional SVM features. \
          Default is False")
parser.add_argument(
    "--epochs", 
    default=25, 
    type=int, 
    help="number of epochs for FastText training")
parser.add_argument(
    "--wordNgrams", 
    nargs=2,
    metavar=('minn', 'maxn'),
    default=[1, 3], 
    type=int, 
    help="word n-gram range. Default is [1,3]")
parser.add_argument(
    "--charNgrams", 
    nargs=2,
    metavar=('minn', 'maxn'),
    default=[2, 5], 
    type=int, 
    help="character n-gram range. Default is [2,5]")


def preprocess(in_file: FilePath) -> pd.DataFrame:
    """
    Pre-process input files into appropriate DataFrame
    """
    df = pd.read_csv(in_file, sep="\t")
    df = df.drop(['ID', 'Sentiment'], axis=1)
    df["Tweet"] = df["Tweet"].str.lower()
    # map labels to ints, suitable for sklearn
    # target here refers to whether target is mentioned in tweet
    target_map = {"OTHER": 0, "NO ONE": 0, "TARGET": 1}
    stance_map = {"AGAINST": 0, "FAVOR": 1, "NONE": 2}
    df["Opinion towards"] = df["Opinion towards"].map(target_map)
    df["Stance"] = df["Stance"].map(stance_map)
    return df


def prepare_fasttext(train_df: pd.DataFrame,
                     test_df: pd.DataFrame) -> None:
    """
    FastText requires labels to be annotated with __label__ prefix
    and data must be read from a file, so save in appropriate format
    """
    train_df["Stance"] = "__label__" + train_df["Stance"].astype(str)
    train_df.to_csv("ft_train.csv", sep="\t", header=False, index=False)
    test_df["Stance"] = "__label__" + test_df["Stance"].astype(str)
    test_df.to_csv("ft_test.csv", sep="\t", header=False, index=False)


def fasttext_classifier(train: pd.DataFrame, 
                        test: pd.DataFrame,
                        args: argparse.Namespace,
                        predict: bool = False) -> \
                                Union[Dict, fasttext.FastText._FastText]:
    """
    Train FastText supervised model
    """
    # saves train and test data as csv files in appropriate format
    prepare_fasttext(train, test)

    # train w2v model for `args.epochs` epochs with
    # 1 to `args.wordNgrams[1]` word ngrams and
    # `args.charNgrams[0]` to `args.charNgrams[1]` char ngrams
    model = fasttext.train_supervised(input="ft_train.csv",
                                      epoch=args.epochs,
                                      wordNgrams=args.wordNgrams[1],
                                      minn=args.charNgrams[0],
                                      maxn=args.charNgrams[1])
    if predict:
        return model.test_label("ft_test.csv")

    return model


def svm_classifier(train: pd.DataFrame, 
                   test: pd.DataFrame,
                   args: argparse.Namespace) -> ArrayLike:
    # use both word and char ngrams as features
    analyzers = [
        ("word", TfidfVectorizer(
            ngram_range=args.wordNgrams, analyzer="word")), 
        ("char_wb", TfidfVectorizer(
            ngram_range=args.charNgrams, analyzer="char_wb"))]
    vectorizer = FeatureUnion(analyzers)

    x_train = vectorizer.fit_transform(train["Tweet"])
    x_test = vectorizer.transform(test["Tweet"])

    if args.target:
        # include whether opinion towards mentions the target as feature
        x_train = hstack([x_train, coo_matrix(train["Opinion towards"]).T])
        x_test = hstack([x_test, coo_matrix(test["Opinion towards"]).T])

    # include FastText sentence embeddings as additional features 
    if args.we:
        model = fasttext_classifier(train, test, args)
        ft_train_sents = np.array(
            [model.get_sentence_vector(t) for t in train["Tweet"]])
        ft_test_sents = np.array(
            [model.get_sentence_vector(t) for t in test["Tweet"]])

        x_train = hstack([x_train, coo_matrix(ft_train_sents)])
        x_test = hstack([x_test, coo_matrix(ft_test_sents)])

    y_train = train["Stance"]

    clf = svm.SVC(kernel="linear")
    clf.fit(x_train, y_train)
    
    y_preds = clf.predict(x_test)

    return y_preds


def main(args: argparse.Namespace) -> None:
    # prepare dataframes
    train_df = preprocess(Path(args.train)) 
    test_df = preprocess(Path(args.test))

    if args.classifier == "svm":
        y_preds = svm_classifier(train_df, test_df, args)
        y_true = test_df["Stance"]
        f_against, f_favor, _ = f1_score(y_true, y_preds, average=None)
        # average F1 score as defined in Mohammad et al. (2017)
        avg_f1 = (f_against + f_favor) / 2
        print("Average F1:", avg_f1)

    elif args.classifier == "ft":
        metrics = fasttext_classifier(train_df, test_df, args, predict=True)
        # test_label gives F1 per label, so average of 1 and 0 (favor
        # and against) can be calculated
        avg_f1 = (metrics['__label__0']['f1score'] + 
                metrics['__label__1']['f1score']) / 2
        print("Average F1:", avg_f1)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
