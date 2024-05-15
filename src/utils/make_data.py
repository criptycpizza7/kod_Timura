from utils.get_stats import (
    get_channel_stats,
    get_comments_in_videos,
    get_video_details,
    get_video_ids,
)
from utils.youtube import make_youtube
import pandas as pd

from streamlit import cache_data
import streamlit as st

from googleapiclient.errors import HttpError

from neo.database import open_driver, close_driver, run_query


def prepare_channel_data(channel_ids: list[str], api_key) -> pd.DataFrame | str:
    try:
        youtube = make_youtube(api_key)
        channel_data = get_channel_stats(youtube, channel_ids)
        if isinstance(channel_data, str):
            if channel_data == "error":
                return channel_data
        numeric_cols = ["subscribers", "views", "totalVideos"]
        channel_data[numeric_cols] = channel_data[numeric_cols].apply(
            pd.to_numeric, errors="coerce"
        )
    except HttpError:
        st.write("Not valid api_key")
        return "error"

    return channel_data

def get_video_statistics(channel_data: pd.DataFrame, api_key):
    # Create a dataframe with video statistics and comments from all channels

    youtube = make_youtube(api_key)

    video_df = pd.DataFrame()
    comments_df = pd.DataFrame()

    for c in channel_data["channelName"].unique():
        print("Getting video information from channel: " + c)
        playlist_id = channel_data.loc[
            channel_data["channelName"] == c, "playlistId"
        ].iloc[0]
        video_ids = get_video_ids(youtube, playlist_id)

        query = "CREATE (c:Channel {name: '" + c + "'})"

        print(query)
        
        driver = open_driver()
        with driver.session() as session:
            session.run(query)

        # get video data
        video_data = get_video_details(youtube, video_ids)
        # get comment data
        comments_data = get_comments_in_videos(youtube, video_ids)

        # append video data together and comment data together
        video_df = pd.concat([video_df, pd.DataFrame(video_data)], ignore_index=True)
        comments_df = pd.concat(
            [comments_df, pd.DataFrame(comments_data)], ignore_index=True
        )

    return video_df, comments_df
