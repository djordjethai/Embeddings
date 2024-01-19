from os import environ
import pinecone
from umap import UMAP
import streamlit as st
import matplotlib.pyplot as plt     # 2D
import plotly.graph_objs as go      # 3D
from openai import OpenAI
from mpl_toolkits.mplot3d import Axes3D     # mora, koristi se implicitno
from myfunc.mojafunkcija import st_style

client = OpenAI()

import warnings
warnings.filterwarnings("ignore")       # globalno

st.set_page_config(page_title="UMAP", page_icon="🗺️", layout="wide")
st_style()
st.header("UMAP testiranje")

upit = st.text_input("Unesite upit: ")
threeD = True

_ = environ["OPENAI_API_KEY"]

col1, col2, col3, _, col5 = st.columns(5)

namespace = col1.radio("Namespace", ("zapisnici", "pravnik"), index=0)
dimens = col2.radio("Dimensionality", ("2D", "3D"), index=1)
top_k = col3.slider("top_k", 1, 20, 10, 1)
metric = col5.radio("Distance metric", ("cosine", "euclidean"), index=0)

def embed_text_with_openai(text):
    return client.embeddings.create(
        input=[text], 
        model="text-embedding-ada-002"
    ).data[0].embedding

def query_pinecone(vector):
    return index.query(vector, top_k=top_k, namespace=namespace, include_values=True)


if upit not in ["", " "] and st.button("Vizualizuj"):
    pinecone_api_key = environ["PINECONE_API_KEY_POS"]
    pinecone.init(api_key=pinecone_api_key, environment="us-west1-gcp-free")
    index = pinecone.Index("positive")

    def visualize_results(results, query_vector):
        if not results["matches"]:
            st.write("Nema podataka za vizualizaciju.")
            return

        data = [res["values"] for res in results["matches"]]
        ids = [res["id"][:5] for res in results["matches"]]
        if len(data) == 0 or len(data[0]) == 0:
            st.write("Podaci nisu u pravilnom formatu za UMAP.")
            return

        if dimens=="3D":
            umap_model = UMAP(n_components=3, metric=metric)
            transformed_data = umap_model.fit_transform(data)

            if transformed_data.shape[1] != 3:
                st.write("Error: Transformed data does not have 3 dimensions")
                return

            query_transformed = umap_model.transform([query_vector])
            if query_transformed.shape[1] != 3:
                st.write("Error: Query vector transformation does not have 3 dimensions")
                return

            avg_vector = transformed_data.mean(axis=0)

            trace_data = go.Scatter3d(
                x=transformed_data[:, 0],
                y=transformed_data[:, 1],
                z=transformed_data[:, 2],
                mode='markers+text',
                text=ids,
                marker=dict(size=5, opacity=0.8)
            )

            trace_avg = go.Scatter3d(
                x=[avg_vector[0]],
                y=[avg_vector[1]],
                z=[avg_vector[2]],
                mode='markers+text',
                text=["avg"],
                marker=dict(color='green', size=10)
            )

            trace_query = go.Scatter3d(
                x=[query_transformed[0, 0]],
                y=[query_transformed[0, 1]],
                z=[query_transformed[0, 2]],
                mode='markers+text',
                text=["query"],
                marker=dict(color='purple', size=10)
            )

            layout = go.Layout(
                margin=dict(l=0, r=0, b=0, t=0),
                scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
            )

            fig = go.Figure(data=[trace_data, trace_query, trace_avg], layout=layout)
            st.plotly_chart(fig, use_container_width=True)

        else:
            umap_model = UMAP(n_components=2, n_neighbors=11, metric="cosine", random_state=42)
            transformed_data = umap_model.fit_transform(data)
            avg_vector = transformed_data.mean(axis=0)

            fig, ax = plt.subplots()
            
            for i, txt in enumerate(ids):
                ax.annotate(txt, (transformed_data[i, 0], transformed_data[i, 1]))
            ax.scatter(transformed_data[:, 0], transformed_data[:, 1])

            query_transformed = umap_model.transform([query_vector])
            ax.scatter(query_transformed[:, 0], query_transformed[:, 1], color="purple")
            ax.text(query_transformed[:, 0], query_transformed[:, 1], "query", fontsize=10, ha="right", color="purple")

            ax.scatter(avg_vector[0], avg_vector[1], color="green")
            ax.text(avg_vector[0], avg_vector[1], "avg", fontsize=10, ha="right", color="green")

            st.pyplot(fig)



    vector = embed_text_with_openai(upit)
    results = query_pinecone(vector)
    visualize_results(results, vector)

