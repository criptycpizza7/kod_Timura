import streamlit as st
css = """
                <style>
                div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                #MainMenu {
                visibility: hidden;
                height: 0%;
                }
                header {
                visibility: hidden;
                height: 0%;
                }
                footer {
                visibility: hidden;
                height: 0%;
                }
                </style>
                """
st.markdown(css, unsafe_allow_html=True)

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from utils.make_data import get_video_statistics, prepare_channel_data
from utils.misc import get_evals, is_channel_id_provided, get_model, make_day, make_embeddings, make_subscribers_df, make_views_df, make_wordcloud, plot_cloud, show_examples, transfrom_video

from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

from fpdf import FPDF

import shutil

from neo.database import open_driver

from py2neo import Graph
import networkx as nx


driver = open_driver()

if "submit" not in st.session_state:
    st.session_state.submit = False

if "start" not in st.session_state:
    st.session_state.start = False

if "analysis" not in st.session_state:
    st.session_state.analysis = False

if st.session_state.start:
    st.session_state.submit = False
    st.session_state.start = False


if not st.session_state.submit:

    if "n_rows" not in st.session_state:
        st.session_state.n_rows = 1

    st.session_state.api_key = st.text_input(
        label="Api Key", placeholder="Api Ключ", label_visibility="hidden"
    )
    st.session_state.hf_api_key = st.text_input(
        label="hf Api Key", placeholder="hf Api Ключ", label_visibility="hidden"
    )

    cols = st.columns([3, 3, 13])

    add = cols[0].button(label="Добавить")
    delete = cols[1].button(label="Удалить")

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
        channel_ids.append(
            st.text_input(
                label=f"Channel id {i + 1}",
                placeholder=f"id Канала {i + 1}",
                label_visibility="hidden",
                key=i,
            )
        )

    cols2 = st.columns([4, 2, 4])
    st.session_state.submit = cols2[1].button(
        label="Принять", use_container_width=True
    )

    st.session_state.channel_ids = channel_ids

    if st.session_state.submit and is_channel_id_provided(channel_ids):
        st.rerun()
    
    st.session_state.analysis = st.checkbox("Продвинутая аналитика")

if st.session_state.submit:


    st.subheader("Графики")

    pdf = FPDF()

    if not st.session_state.submit:
        st.rerun()

    channel_ids = st.session_state.channel_ids

    if not is_channel_id_provided(channel_ids):
        st.session_state.submit = False
        st.rerun()

    channel_data = prepare_channel_data(channel_ids, st.session_state.api_key)

    if isinstance(channel_data, str):
        st.session_state.submit = False
        st.rerun()

    old_subscribers_df, subscribers_df = make_subscribers_df(channel_data)

    fig = plt.bar(x=old_subscribers_df["channelName"], height=old_subscribers_df["subscribers"])

    plt.title("Подписчки")

    img1 = "bar_1.png"
    plt.savefig(img1)
    pdf.add_page()
    pdf.image(img1, x=0, y=0, h=pdf.h, w=pdf.w)
    plt.clf()

    st.subheader("Подписчики")
    st.bar_chart(subscribers_df)

    old_views_df, views_df = make_views_df(channel_data)

    fig = plt.bar(x=old_views_df["channelName"], height=old_views_df["views"])

    plt.title("Просмотры")

    img2 = "bar_2.png"
    plt.savefig(img2)
    pdf.add_page()
    pdf.image(img2, x=0, y=0, h=pdf.h, w=pdf.w)
    plt.clf()

    st.subheader("Просмотры")
    st.bar_chart(views_df)

    video_df, comments_df = get_video_statistics(channel_data, st.session_state.api_key)

    video_df.to_csv('./src/contents/video_data_top10_channels.csv')
    comments_df.to_csv('./src/contents/comments_data_top10_channels.csv')

    video_df = transfrom_video(video_df)

    # users
    for index, row in comments_df.iterrows():
        comments = row["comments"]

        users = [{"name": item["author"][1:]} for item in comments]
        query = """UNWIND $users AS user CREATE (u:User {name: user.name})
                """
        with driver.session() as session:
            session.run(query, users=users)
    
    # relation
    for index, row in comments_df.iterrows():
        comments = row["comments"]

        for item in comments:
            author = item["author"]
            video_id = row["video_id"]  

            for index2, row_vid in video_df.iterrows():
                if row_vid["video_id"] == video_id:
                    query = "MATCH (u:User {name: '" + author[1:] + "'}), (c:Channel {name: '" + row_vid["channelTitle"] + "'}) MERGE (u)-[:SUBSCRIBED_TO]->(c);"
                    with driver.session() as session:
                        session.run(query)
                    break

    fig = plt.figure(figsize=(18, 6))
    sns.violinplot(x="channelTitle", y="viewCount", data=video_df)
    fig.suptitle("Просмотры по каналу", fontsize=14)

    plt.title("Расределение промотров по каналам")
    st.subheader("Расределение промотров по каналам")
    st.pyplot(fig)

    img3 = "violin.png"
    plt.savefig(img3)
    pdf.add_page()
    pdf.image(img3, x=0, y=0, h=pdf.h, w=pdf.w)
    plt.clf()

    st.subheader("Облако слов")
    wordcloud = make_wordcloud(video_df)

    fig = plot_cloud(wordcloud)

    img4 = "cloud.png"
    plt.title("Облако слов")
    plt.savefig(img4)
    pdf.add_page()
    pdf.image(img4, x=0, y=0, h=pdf.h, w=pdf.w)
    plt.clf()

    old_day_df, day_df = make_day(video_df)
    st.subheader("Выпуск видео по дням недели")
    st.bar_chart(day_df)


    fig = plt.bar(x=old_day_df["Day"], height=old_day_df["Count"])

    plt.title("Выпуск видео по дням недели")
    img5 = "day.png"
    plt.savefig(img5)
    pdf.add_page()
    pdf.image(img5, x=0, y=0, h=pdf.h, w=pdf.w)
    plt.clf()
    if st.session_state.analysis:
        if st.session_state.hf_api_key:
            # pass
            model = get_model(st.session_state.hf_api_key)
            eval_ds, eval_dataloader, corpus, authors = get_evals(comments_df)

            embeddings = make_embeddings(model, eval_dataloader)

            pca = PCA(n_components=15, random_state=42)
            emb_15d = pca.fit_transform(embeddings)

            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.6,
                metric="cosine",
                linkage="average",
            ).fit(emb_15d)


            pca = PCA(n_components=2, random_state=42)
            emb_2d = pd.DataFrame(pca.fit_transform(embeddings), columns=["x1", "x2"])
            emb_2d["label"] = clustering.labels_
            emb_2d["label"].nunique()

            for index, item in enumerate(emb_2d["label"]):
                query = "CREATE (cl:Cluster {name: '" + str(index + 1) + "'})"
                with driver.session() as session:
                    session.run(query)
            
            for index in range(len(clustering.labels_)):
                items = authors[emb_2d['label'] == index]
                for author in items:
                    query = "MATCH (u:User {name: '" + author[1:] +"'}), (cl:Cluster {name: '" + str(index) + "'}) MERGE (u)-[:CONNECTED_TO]->(cl);"
                    with driver.session() as session:
                        session.run(query)

            fig = plt.scatter(x=emb_2d["x1"], y=emb_2d["x2"], c=emb_2d["label"])

            plt.title("Точечный график")
            img6 = "scatter.png"
            plt.savefig(img6)
            pdf.add_page()
            pdf.image(img6, x=0, y=0, h=pdf.h, w=pdf.w)
            plt.clf()
            st.subheader("Точечный график")
            st.scatter_chart(emb_2d, x="x1", y="x2", color="label", width=800, height=600)

            query = """
            MATCH (c:Channel)
            WITH c ORDER BY rand() LIMIT 1

            // Match 20 users subscribed to the selected channel
            MATCH (u:User)-[:SUBSCRIBED_TO]->(c)
            WITH c, u LIMIT 20

            OPTIONAL MATCH (u)-[:CONNECTED_TO]->(cl:Cluster)
            RETURN c, u, collect(distinct cl) AS clusters;
            """

            graph = Graph("bolt://94.228.122.139:7687", auth=("neo4j", "12345678"))
            results = graph.run(query).data()

            G = nx.DiGraph()
            for result in results:
                c = result['c']
                u = result['u']
                cl = result['clusters']
                G.add_node(c['name'])
                G.add_node(u['name'])
                G.add_edge(c['name'], u['name'], label="subscribed_to")
                for cluster in cl:
                    G.add_edge(cluster, u["name"], label="connected_to")

            pos = nx.spring_layout(G)
            labels = nx.get_edge_attributes(G, 'label')
            nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

            graph_img = "graph.png"
            plt.title("Граф базы данных")
            plt.savefig(graph_img)
            pdf.add_page()
            pdf.image(graph_img, x=0, y=0, h=pdf.h, w=pdf.w)
            st.subheader("Граф базы данных")
            st.image(graph_img)
            plt.clf()

            channel_names = video_df["channelTitle"].unique()

            for name in channel_names:
                st.write(name)
                query = "MATCH (c:Channel {name: '" + name + "'})<-[:SUBSCRIBED_TO]-(u:User)-[:CONNECTED_TO]->(cl:Cluster)"
                query = query + "WITH c, cl, COUNT(u) AS userCount ORDER BY userCount DESC LIMIT 3 RETURN c, cl, userCount;"

                results = graph.run(query).data()

                for index, result in enumerate(results):
                    st.write(f"Кластер {index + 1}")
                    show_examples(corpus, emb_2d, int(result["cl"]["name"]), 10)
            
            
            query = """
            MATCH (n)
            DETACH DELETE n;
            """
            with driver.session() as session:
                session.run(query)

        else:
            st.write("hf апи ключ не предоставлен")
    
    cols2 = st.columns([4, 2, 4])

    st.session_state.start = cols2[1].button(label="Назад", use_container_width=True)

    pdf_name = "src/contents/plots.pdf"

    shutil.make_archive("Files", 'zip', "src/contents")

    pdf.output(pdf_name)
    with open("Files.zip", 'rb') as f:
        cols2[0].download_button(
            label="Скачать",
            data=f,
            file_name="Files.zip",
        )

if st.session_state.submit and not is_channel_id_provided(channel_ids):
    st.write("Не предоставлены id каналов")
