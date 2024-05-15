from matplotlib import pyplot as plt
from transformers import AutoModel
import torch
from tqdm import tqdm

from utils.custom_dataset import CustomDataset

import pandas as pd
from torch.utils.data import DataLoader

from streamlit import cache_data, cache_resource

from dateutil import parser
import isodate

import streamlit as st

from nltk.corpus import stopwords
from wordcloud import WordCloud

import numpy as np


def is_channel_id_provided(channel_ids: list[str]) -> bool:
    for id in channel_ids:
        if id:
            return True

    return False

@cache_resource
def get_model(token: str = "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"):
    return AutoModel.from_pretrained("DeepPavlov/rubert-base-cased", token=token)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output["last_hidden_state"]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def make_embeddings(model, eval_dataloader) -> torch.Tensor:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    embeddings = torch.Tensor().to(device)

    with torch.no_grad():
        for n_batch, batch in enumerate(tqdm(eval_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            embeddings = torch.cat(
                [embeddings, mean_pooling(outputs, batch["attention_mask"])]
            )
        embeddings = embeddings.cpu().numpy()
        return embeddings

def get_evals(df):
    comments = []

    for index, row in df.iterrows():
        items = row["comments"]
        comments.extend([(item["comment"], item["author"]) for item in items])

    corpus = np.array([item[0] for item in comments])  # Extract only comments for processing
    authors = np.array([item[1] for item in comments])  # Maintain a separate list of authors

    eval_ds = CustomDataset(corpus)
    eval_dataloader = DataLoader(eval_ds, batch_size=10)

    return eval_ds, eval_dataloader, corpus, authors

@cache_data
def make_subscribers_df(channel_data) -> pd.DataFrame:
    subscribers_df = pd.DataFrame()

    subscribers_df["subscribers"] = []
    subscribers_df["channelName"] = []

    for index, row in channel_data.iterrows():
        channel_name = row["channelName"]
        subscribers = row["subscribers"]

        subscribers_df.loc[len(subscribers_df)] = {
            "channelName": channel_name,
            "subscribers": subscribers,
        }
    
    old_subscribers_df = subscribers_df.copy()

    subscribers_df.set_index("channelName", inplace=True)
    return old_subscribers_df, subscribers_df

@cache_data
def make_views_df(channel_data) -> pd.DataFrame:
    views_df = pd.DataFrame()

    views_df["views"] = []
    views_df["channelName"] = []

    for index, row in channel_data.iterrows():
        channel_name = row["channelName"]
        views = row["views"]

        views_df.loc[len(views_df)] = {"channelName": channel_name, "views": views}
    
    old_view_df = views_df.copy()

    views_df.set_index("channelName", inplace=True)

    return old_view_df, views_df

@cache_data
def transfrom_video(video_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["viewCount", "likeCount", "favouriteCount", "commentCount"]
    video_df[cols] = video_df[cols].apply(pd.to_numeric, errors="coerce", axis=1)

    # Create publish day (in the week) column
    video_df["publishedAt"] = video_df["publishedAt"].apply(lambda x: parser.parse(x))
    video_df["pushblishDayName"] = video_df["publishedAt"].apply(
        lambda x: x.strftime("%A")
    )

    # convert duration to seconds
    video_df["durationSecs"] = video_df["duration"].apply(
        lambda x: isodate.parse_duration(x)
    )
    video_df["durationSecs"] = video_df["durationSecs"].astype("timedelta64[s]")

    # Add number of tags
    video_df["tagsCount"] = video_df["tags"].apply(lambda x: 0 if x is None else len(x))

    # Comments and likes per 1000 view ratio
    video_df["likeRatio"] = video_df["likeCount"] / video_df["viewCount"] * 1000
    video_df["commentRatio"] = video_df["commentCount"] / video_df["viewCount"] * 1000

    # Title character length
    video_df["titleLength"] = video_df["title"].apply(lambda x: len(x))

    return video_df

def plot_cloud(wordcloud):
    fig = plt.figure(figsize=(30, 20))
    # fig.images(wordcloud)
    plt.imshow(wordcloud)
    # fig.axes("off")
    plt.axis("off")
    st.pyplot(fig)

    return fig

@cache_data
def make_wordcloud(video_df: pd.DataFrame):
    stop_words = set(stopwords.words("english", "russian"))
    video_df["title_no_stopwords"] = video_df["title"].apply(
        lambda x: [item for item in str(x).split() if item not in stop_words]
    )

    all_words = list([a for b in video_df["title_no_stopwords"].tolist() for a in b])
    all_words_str = " ".join(all_words)

    return WordCloud(
        width=2000,
        height=1000,
        random_state=1,
        background_color="black",
        colormap="viridis",
        collocations=False,
    ).generate(all_words_str)

def make_day(video_df: pd.DataFrame) -> pd.DataFrame:
    weekdays = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    # Assume the correct column name is 'pushblishDayName'
    day_df = video_df["pushblishDayName"].value_counts().reindex(weekdays, fill_value=0)

    # Create a DataFrame for plotting
    day_df = pd.DataFrame({"Day": day_df.index, "Count": day_df.values})

    old_day_df = day_df.copy()

    day_df.set_index("Day", inplace=True)

    return old_day_df, day_df

def show_examples(corpus, emb_2d, cluster, n):
    for i in range(min(n, len(corpus[emb_2d['label'] == cluster]))):
        st.write(i, corpus[emb_2d['label'] == cluster][i].split('.')[0])