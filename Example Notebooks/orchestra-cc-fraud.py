#!/usr/bin/env python3

import os
import logging
import pandas as pd
from datetime import datetime
from collections import deque

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sentence_transformers import SentenceTransformer, models
from torch import nn
from google.cloud import bigquery

from orchestralib import OrchestraClient

orchestra = OrchestraClient(
    organization="my organization",
    environment="development",
)

user_to_last5_ue = {}


def _get_unexpectedness_score(user_id, business_address, payment_method):
    """
    Score describing how unexpected the transaction is for the given user.
    """
    if payment_method == "online":
        return 0.0

    # Given that the data is fake, there is no real business_address or user_address.
    # Simply return a random unexpectedness_score.
    import random

    return random.uniform(0.0, 150.0)


def _avg_last_5_unexpectedness(user_id, unexpectedness_score):
    if user_id not in user_to_last5_ue:
        user_to_last5_ue[user_id] = deque(maxlen=5)  # Doubly ended queue
        user_to_last5_ue[user_id].append(unexpectedness_score)
    unexpectedness_score_list = list(user_to_last5_ue[user_id])
    return sum(unexpectedness_score_list) / len(unexpectedness_score_list)


def _get_business_embeddings(business_name, address, sentence_transformer_model):
    return list(sentence_transformer_model.encode([business_name + " " + address])[0])


def write_to_bigquery(df, table_name, code_block_name):
    orchestra.log_code_block(code_block_name).start(input_features=[df])
    client = bigquery.Client(project="novuslabs-ml")
    full_table = "cc_fraud." + table_name
    job = client.load_table_from_dataframe(df, full_table)

    orchestra.log_features(
        df, "bigquery://novuslabs-ml.cc_fraud." + table_name, output_for=code_block_name
    )
    # The output_features are already logged into BQ. No need to specify in-memory
    # output_features in the end() API.
    orchestra.log_code_block(code_block_name).end()
    logging.info("Bigquery job for storing data: %s", job)


def s3_to_bigquery():
    orchestra.log_code_block("read-from-s3").start()
    s3_url = "s3://orchestra-ml-prototype-prod/feature-productionization/cc-fraud.csv"
    df = pd.read_csv(s3_url)
    orchestra.log_features(df, s3_url, input_for="read-from-s3")
    orchestra.log_code_block("read-from-s3").end(output_features=[df])

    orchestra.log_code_block("remove-na").start(input_features=[df])
    df.dropna(inplace=True)
    orchestra.log_code_block("remove-na").end(output_features=[df])

    write_to_bigquery(df, "clean_data", "write-cleandata-to-bigquery")


def _transform_timestamp(df):
    """
    Process ts into hour, minute, day of week, etc.
    """
    orchestra.log_code_block("transform-timestamp").start(input_features=[df])
    df["hour"] = df.apply(
        lambda row: datetime.fromtimestamp(row["timestamp"]).hour, axis=1
    )
    df["minute"] = df.apply(
        lambda row: datetime.fromtimestamp(row["timestamp"]).minute, axis=1
    )
    df["month"] = df.apply(
        lambda row: datetime.fromtimestamp(row["timestamp"]).month, axis=1
    )
    df["day"] = df.apply(
        lambda row: datetime.fromtimestamp(row["timestamp"]).day, axis=1
    )
    df["dayoftheweek"] = df.apply(
        lambda row: datetime.fromtimestamp(row["timestamp"]).isoweekday(), axis=1
    )
    orchestra.log_code_block("transform-timestamp").end(
        output_features=[df],
        created_using=orchestra.created_using(
            df, ["timestamp"], df, ["hour", "minute", "month", "day", "dayoftheweek"]
        ),
    )


def _transform_businessdata(df):
    orchestra.log_code_block("one-hot-encoding-txn-type").start(input_features=[df])
    tempdf = pd.DataFrame()
    le_txn = LabelEncoder()
    tempdf["txn_type"] = le_txn.fit_transform(df.payment_method)
    txn_ohe = OneHotEncoder()
    X = txn_ohe.fit_transform(tempdf.txn_type.values.reshape(-1, 1)).toarray()
    tempdf = pd.DataFrame(X, columns=["txn_online", "txn_atm", "txn_pos"])
    df = df.join(tempdf)
    orchestra.log_code_block("one-hot-encoding-txn-type").end(
        output_features=[df],
        created_using=orchestra.created_using(
            df, ["txn_type"], df, ["txn_online", "txn_atm", "txn_pos"]
        ),
    )

    orchestra.log_code_block("business-embedding-computation").start(
        input_features=[df]
    )
    # Generate embeddings based on business name and business address
    word_embedding_model = models.Transformer("bert-base-uncased", max_seq_length=256)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    dense_model = models.Dense(
        in_features=pooling_model.get_sentence_embedding_dimension(),
        out_features=5,
        activation_function=nn.Tanh(),
    )
    model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model, dense_model]
    )
    model.save("/tmp/cc-fraud-embedding")
    x = df.apply(
        lambda row: _get_business_embeddings(
            row["business_name"], row["business_address"], model
        ),
        axis=1,
    )
    i = 0
    for index, values in x.iteritems():
        df.loc[
            i,
            [
                "business_embedding_1",
                "business_embedding_2",
                "business_embedding_3",
                "business_embedding_4",
                "business_embedding_5",
            ],
        ] = values
        i = i + 1
    orchestra.log_code_block("business-embedding-computation").end(
        output_features=[df],
        created_using=orchestra.created_using(
            df,
            ["business_name", "business_address"],
            df,
            [
                "business_embedding_1",
                "business_embedding_2",
                "business_embedding_3",
                "business_embedding_4",
                "business_embedding_5",
            ],
        ),
    )

    return df


def transform_data():
    client = bigquery.Client()

    df = pd.DataFrame()

    orchestra.log_code_block("read-cleandata-from-bigquery").start()
    query_job = None

    # Creating embeddings for large number of text strings takes a long time.
    # To keep this demo easily executable within a short period of time
    # and only using local CPUs, the following code simply reads a sample
    # of input_features from the clean_data table.
    for payment_type in ["online", "atm", "pos"]:
        QUERY = "SELECT * fROM novuslabs-ml.cc_fraud.clean_data WHERE payment_method = '{ptype}' LIMIT 3".format(
            ptype=payment_type
        )
        query_job = client.query(QUERY)
        df = pd.concat([df, query_job.result().to_dataframe()], ignore_index=True)
    orchestra.log_features(
        df,
        "bigquery://novuslabs-ml.cc_fraud.clean_data",
        input_for="read-cleandata-from-bigquery",
    )
    orchestra.log_code_block("read-cleandata-from-bigquery").end(output_features=[df])

    _transform_timestamp(df)

    orchestra.log_code_block("uescore-calculation").start(input_features=[df])
    df["uescore"] = df.apply(
        lambda row: _get_unexpectedness_score(
            row["user_id"], row["business_address"], row["payment_method"]
        ),
        axis=1,
    )
    orchestra.log_code_block("uescore-calculation").end(
        output_features=[df],
        created_using=orchestra.created_using(
            df, ["user_id", "business_address", "payment_method"], df, ["uescore"]
        ),
    )

    df = _transform_businessdata(df)

    orchestra.log_code_block("avg_5_ue").start(input_features=[df])
    df["avg_5_ue"] = df.apply(
        lambda row: _avg_last_5_unexpectedness(row["user_id"], row["uescore"]), axis=1
    )
    orchestra.log_code_block("avg_5_ue").end(
        output_features=[df],
        created_using=orchestra.created_using(
            df, ["user_id", "uescore"], df, ["avg_5_ue"]
        ),
    )

    write_to_bigquery(df, "cc_fraud_model_ready", "write-model-ready-bq")


def train_model():
    orchestra.log_code_block("read-model-ready-bq").start()

    client = bigquery.Client()
    QUERY = "SELECT * fROM novuslabs-ml.cc_fraud.cc_fraud_model_ready"
    query_job = client.query(QUERY)
    df = query_job.result().to_dataframe()

    orchestra.log_features(
        df,
        "bigquery://novuslabs-ml.cc_fraud.cc_fraud_model_ready",
        input_for="read-model-ready-bq",
    )
    orchestra.log_code_block("read-model-ready-bq").end(output_features=[df])

    # orchestra.log_modeltraining_code('cc-fraud-model-training-code').start(input_features=[df])
    orchestra.log_code_block("cc-fraud-model-trainer").start(input_features=[df])
    data = df[
        [
            "hour",
            "minute",
            "month",
            "day",
            "dayoftheweek",
            "purchase_amount",
            "txn_online",
            "txn_atm",
            "txn_pos",
            "business_embedding_1",
            "business_embedding_2",
            "business_embedding_3",
            "business_embedding_4",
            "business_embedding_5",
            "uescore",
            "avg_5_ue",
            "class",
        ]
    ]
    X = data.drop("class", axis=1)
    y = data["class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.25, random_state=1
    )

    xgb = XGBClassifier(max_depth=4)
    xgb.fit(X_train, y_train)
    orchestra.log_code_block("cc-fraud-model-trainer").end(
        output_models=["cc-fraud-model"]
    )

    # MLFlow integration
    orchestra.init_mlflow(os.environ["MLFLOW_ENDPOINT"])
    orchestra.log_model(
        "cc-fraud-model",
        xgb,
        X,
        y,
        ["float64"],
        None,
        output_for="cc-fraud-model-trainer",
    )


s3_to_bigquery()
transform_data()
train_model()

# print(orchestra.get_yaml())
mldepyaml = orchestra.get_yaml()
with open("/tmp/orchestra.yaml", "w") as f:
    f.write(mldepyaml)

print("The orchestra yaml has been written to /tmp/orchestra.yaml")

