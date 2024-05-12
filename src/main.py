import isodate
from matplotlib import pyplot as plt
import pandas as pd
import streamlit as st
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

from utils.make_data import get_video_statistics, prepare_channel_data
from utils.misc import is_channel_id_provided, get_model

from dateutil import parser

from transformers import BertModel, BertTokenizer


if 'submit' not in st.session_state:
    st.session_state.submit = False

if not st.session_state.submit:
    if 'n_rows' not in st.session_state:
        st.session_state.n_rows = 1

    st.session_state.api_key = st.text_input(label=f"Api Key", placeholder="Api key", label_visibility="hidden")
    st.session_state.hf_api_key = st.text_input(label=f"hf Api Key", placeholder="hf Api key", label_visibility="hidden")

    cols = st.columns([1, 1, 8])

    add = cols[0].button(label="Add")
    delete = cols[1].button(label="Del")


    if add:
        st.session_state.n_rows += 1
        st.rerun()

    if delete:
        st.session_state.n_rows -= 1
        if st.session_state.n_rows < 1:
            st.session_state.n_rows = 1
        st.rerun()

    channel_ids = []

    for i in range(st.session_state.n_rows):
        channel_ids.append(st.text_input(label=f"Channel id {i + 1}",
                                    placeholder=f"Channel id {i + 1}",
                                    label_visibility="hidden",
                                    key=i)) #Pass index as key

    cols2 = st.columns([4, 2, 4])
    st.session_state.submit = cols2[1].button(label="Lets gooooo!!!", use_container_width=True)

    st.session_state.channel_ids = channel_ids

    if st.session_state.submit and is_channel_id_provided(channel_ids):
        st.rerun()

if st.session_state.submit and is_channel_id_provided(st.session_state.channel_ids):
    st.subheader("Graphs")

    if not st.session_state.submit:
        st.rerun()    

    channel_ids = st.session_state.channel_ids

    channel_data = prepare_channel_data(channel_ids, st.session_state.api_key)

    subscribers_df = pd.DataFrame()

    subscribers_df["subscribers"] = []
    subscribers_df["channelName"] = []

    for index, row in channel_data.iterrows():
        channel_name = row["channelName"]
        subscribers = row["subscribers"]

        subscribers_df.loc[len(subscribers_df)] = {"channelName": channel_name, "subscribers": subscribers}

    subscribers_df.set_index("channelName", inplace=True)

    st.subheader("Subscribers")
    st.bar_chart(subscribers_df)

    views_df = pd.DataFrame()

    views_df["views"] = []
    views_df["channelName"] = []

    for index, row in channel_data.iterrows():
        channel_name = row["channelName"]
        views = row["views"]

        views_df.loc[len(views_df)] = {"channelName": channel_name, "views": views}
    
    views_df.set_index("channelName", inplace=True)
    
    st.subheader("Views")
    st.bar_chart(views_df)

    video_df, comments_df = get_video_statistics(channel_data, st.session_state.api_key)

    cols = ['viewCount', 'likeCount', 'favouriteCount', 'commentCount']
    video_df[cols] = video_df[cols].apply(pd.to_numeric, errors='coerce', axis=1)

    # Create publish day (in the week) column
    video_df['publishedAt'] =  video_df['publishedAt'].apply(lambda x: parser.parse(x))
    video_df['pushblishDayName'] = video_df['publishedAt'].apply(lambda x: x.strftime("%A"))

    # convert duration to seconds
    video_df['durationSecs'] = video_df['duration'].apply(lambda x: isodate.parse_duration(x))
    video_df['durationSecs'] = video_df['durationSecs'].astype('timedelta64[s]')

    # Add number of tags
    video_df['tagsCount'] = video_df['tags'].apply(lambda x: 0 if x is None else len(x))

    # Comments and likes per 1000 view ratio
    video_df['likeRatio'] = video_df['likeCount']/ video_df['viewCount'] * 1000
    video_df['commentRatio'] = video_df['commentCount']/ video_df['viewCount'] * 1000

    # Title character length
    video_df['titleLength'] = video_df['title'].apply(lambda x: len(x))

    fig = plt.figure(figsize = (18, 6))
    sns.violinplot(x='channelTitle', y='viewCount', data=video_df, palette='pastel')
    fig.suptitle('Views per channel', fontsize=14)
    st.pyplot(fig)

    stop_words = set(stopwords.words('english'))
    video_df['title_no_stopwords'] = video_df['title'].apply(lambda x: [item for item in str(x).split() if item not in stop_words])

    all_words = list([a for b in video_df['title_no_stopwords'].tolist() for a in b])
    all_words_str = ' '.join(all_words)

    def plot_cloud(wordcloud):
        fig = plt.figure(figsize=(30, 20))
        # fig.images(wordcloud)
        plt.imshow(wordcloud)
        # fig.axes("off")
        plt.axis('off')
        st.pyplot(fig)

    wordcloud = WordCloud(width = 2000, height = 1000, random_state=1, background_color='black',
                        colormap='viridis', collocations=False).generate(all_words_str)
    plot_cloud(wordcloud)

    weekdays = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    # Assume the correct column name is 'pushblishDayName'
    day_df = video_df['pushblishDayName'].value_counts().reindex(weekdays, fill_value=0)

    # Create a DataFrame for plotting
    day_df = pd.DataFrame({'Day': day_df.index, 'Count': day_df.values})

    print(day_df)

    # Plotting the bar chart

    day_df.set_index("Day", inplace=True)
    st.bar_chart(day_df)

    cols2 = st.columns([4, 2, 4])
    st.session_state.submit = not cols2[1].button(label="Fuck, go back", use_container_width=True)

    # if "hf_api_key" in st.session_state:
    #     model = get_model(st.session_state.hf_api_key)

    #     tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
    #     model = BertModel.from_pretrained('DeepPavlov/rubert-base-cased', output_hidden_states = True)
    # else:
    #     st.write("hf api key was not provided")

if st.session_state.submit and not is_channel_id_provided(channel_ids):
    st.write("No channel idx")