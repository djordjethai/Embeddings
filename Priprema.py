import streamlit as st

st.set_page_config(page_title="Embeddings", page_icon="📔", layout="wide")
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader, UnstructuredPDFLoader
import os

from myfunc.mojafunkcija import st_style, pinecone_stats, positive_login, show_logo
from time import sleep
from tqdm.auto import tqdm
from uuid import uuid4
import openai
import json
import numpy as np
import Scrapper

# this code is used to split the document into smaller parts and create OpenAI embeddings


st_style()


def def_chunk():
    with st.sidebar:
        chunk_size = st.slider(
            "Set chunk size in characters (200 - 8000)",
            200,
            8000,
            1500,
            step=100,
            help="Velicina chunka odredjuje velicinu indeksiranog dokumenta. Veci chunk obezbedjuje bolji kontekst, dok manji chunk omogucava precizniji odgovor.",
        )
        chunk_overlap = st.slider(
            "Set overlap size in characters (0 - 1000), must be less than the chunk size",
            0,
            1000,
            0,
            step=10,
            help="Velicina overlapa odredjuje velicinu preklapanja sardzaja dokumenta. Veci overlap obezbedjuje bolji prenos konteksta.",
        )
        return chunk_size, chunk_overlap


def main():
    show_logo()

    if "pod" not in st.session_state:
        st.session_state["pod"] = False

    if "kre" not in st.session_state:
        st.session_state["kre"] = False

    if "man" not in st.session_state:
        st.session_state["man"] = False

    if "prip" not in st.session_state:
        st.session_state["prip"] = False

    if "sta" not in st.session_state:
        st.session_state["sta"] = False

    if "scr" not in st.session_state:
        st.session_state["scr"] = False

    st.subheader("Izaberite operaciju za Embeddings")
    with st.expander("Procitajte uputstvo:"):
        st.caption(
            "Prethodni korak bio je kreiranje pitanja. To smo radili pomocu besplatnog CHATGPT modela. Iz svake oblasti (ili iz dokumenta) zamolimo CHATGPT da kreira relevantna pitanja. Na pitanja mozemo da odgovorimo sami ili se odgovori mogu izvuci iz dokumenta."
        )
        st.caption(
            "Ukoliko zelite da vam model kreira odgovore, odaberite ulazni fajl sa pitanjma iz prethodnog koraka. Opciono, ako je za odgovore potreban izvor, odaberite i fajl sa izvorom. Unesite sistemsku poruku (opis ponasanja modela) i naziv FT modela. Kliknite na Submit i sacekajte da se obrada zavrsi. Fajl sa odgovorima cete kasnije korisiti za kreiranje FT modela."
        )
        st.caption(
            "Pre prelaska na sledecu fazu OBAVEZNO pregledajte izlazni dokument sa odgovorima i korigujte ga po potrebi. "
        )

    import streamlit as st


import Pinecone_Utility
from mojafunkcija import pinecone_stats
import pinecone

if "podeli_button" not in st.session_state:
    st.session_state["podeli_button"] = False
if "manage_button" not in st.session_state:
    st.session_state["manage_button"] = False
if "kreiraj_button" not in st.session_state:
    st.session_state["kreiraj_button"] = False
if "stats_button" not in st.session_state:
    st.session_state["stats_button"] = False
if "screp_button" not in st.session_state:
    st.session_state["screp_button"] = False

if "dokum" not in st.session_state:
    st.session_state["dokum"] = None


def main():
    col1, col2 = st.columns(2)
    chunk_size, chunk_overlap = def_chunk()
    with col1:
        with st.form(key="podeli", clear_on_submit=False):
            st.session_state.podeli_button = st.form_submit_button(
                label="Podeli dokument u delove za indeksiranje",
                use_container_width=True,
                help="Podela dokumenta na delove za indeksiranje",
                on_click=prepare_embeddings(chunk_size, chunk_overlap),
            )
            if st.session_state.podeli_button:
                st.session_state.pod = True
                st.session_state.kre = False
                st.session_state.man = False
                st.session_state.sta = False
                st.session_state.scr = False

    with col2:
        with st.form(key="kreiraj", clear_on_submit=False):
            st.session_state.kreiraj_button = st.form_submit_button(
                label="Kreiraj Pinecone Index",
                use_container_width=True,
                help="Kreiranje Pinecone Indexa",
                on_click=do_embeddings(),
            )
            if st.session_state.kreiraj_button:
                st.session_state.pod = False
                st.session_state.kre = True
                st.session_state.man = False
                st.session_state.sta = False
                st.session_state.scr = False
    with col1:
        with st.form(key="manage", clear_on_submit=False):
            st.session_state.manage_button = st.form_submit_button(
                label="Manage Pinecone Index",
                use_container_width=True,
                help="Manage Pinecone Indexa",
                on_click=Pinecone_Utility.main(),
            )
            if st.session_state.manage_button:
                st.session_state.pod = False
                st.session_state.kre = False
                st.session_state.man = True
                st.session_state.sta = False
                st.session_state.scr = False
    with col2:
        with st.form(key="stats", clear_on_submit=False):
            index = pinecone.Index("embedings1")
            st.session_state.stats_button = st.form_submit_button(
                label="Pokazi stats za Pinecone Index",
                use_container_width=True,
                help="Stats za Pinecone Indexa",
                on_click=pinecone_stats(index),
            )
            if st.session_state.stats_button:
                st.session_state.pod = False
                st.session_state.kre = False
                st.session_state.man = False
                st.session_state.sta = True
                st.session_state.scr = False
    with col2:
        with st.form(key="screp", clear_on_submit=False):
            st.session_state.screp_button = st.form_submit_button(
                label="Scrape URL",
                use_container_width=True,
                help="Scrape URL",
                on_click=Scrapper.main(),
            )
            if st.session_state.screp_button:
                st.session_state.pod = False
                st.session_state.kre = False
                st.session_state.man = False
                st.session_state.sta = False
                st.session_state.scr = True
    st.divider()


def prepare_embeddings(chunk_size, chunk_overlap):
    # with st.form(key="my_form_prepare", clear_on_submit=False):
    st.subheader("Upload documents and metadata for Pinecone Index")

    dokum = st.file_uploader(
        "Izaberite dokument/e", key="upload_file", type=["txt", "pdf", "docx"]
    )
    # prefix moze da se definise i dinamicki
    text_prefix = st.text_input(
        "Unesi prefix za tekst: ",
        help="Prefix se dodaje na pocetak teksta pre podela na delove za indeksiranje",
    )
    st.session_state.submit_button = st.form_submit_button(
        label="Submit",
        help="Submit dugme pokrece podelu dokumenta na delove za indeksiranje",
    )
    st.write(f"Chunk size: {chunk_size}, chunk overlap: {chunk_overlap}")
    if len(text_prefix) > 0:
        text_prefix = text_prefix + " "

    if dokum is not None and st.session_state.submit_button == True:
        if ".pdf" in st.session_state.dokum.name:
            loader = UnstructuredPDFLoader(
                st.session_state.dokum.name, encoding="utf-8"
            )
        else:
            loader = UnstructuredFileLoader(
                st.session_state.dokum.name, encoding="utf-8"
            )

        data = loader.load()

        # Split the document into smaller parts
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        texts = text_splitter.split_documents(data)

        #
        ###############
        #
        # dodati mogucnost za prefix i sufix na embeddinge
        #
        ###############
        # # Ask the user if they want to do OpenAI embeddings

        # # Create the OpenAI embeddings
        st.write(f"Ucitano {len(texts)} tekstova")

        # Define a custom method to convert Document to a JSON-serializable format
        output_json_list = []

        # Loop through the Document objects and convert them to JSON
        for document in texts:
            output_dict = {
                "id": str(uuid4()),
                "text": text_prefix + document.page_content,
                "source": document.metadata.get("source", ""),
            }
            output_json_list.append(output_dict)

        # # Specify the file name where you want to save the JSON data
        json_file_path = "chunks.json"

        # # Save 'chunks' to a JSON file
        with open(json_file_path, "w", encoding="utf-8") as json_file:
            json_file.write("[ ")  # Start with an opening bracket

            for index, item in enumerate(output_json_list):
                if index > 0:
                    json_file.write(
                        ",\n"
                    )  # Add a comma and newline for all except the first item
                json.dump(item, json_file, ensure_ascii=False)

            json_file.write(" ]")  # End with a closing bracket

        st.success(f"Texts saved to {json_file_path} and are now ready for Embeddings")


#
def do_embeddings():
    err_log = ""
    # Read the texts from the .txt file
    chunks = []
    dokum = st.file_uploader(
        "Izaberite dokument/e",
        key="upload_json_file",
        type=[".json"],
        help="Izaberite dokument koji ste podelili na delove za indeksiranje",
    )

    # Now, you can use stored_texts as your texts
    # with st.form(key="my_form2", clear_on_submit=False):
    namespace = st.text_input(
        "Unesi naziv namespace-a: ",
        help="Naziv namespace-a je obavezan za kreiranje Pinecone Indexa",
    )
    submit_button = st.form_submit_button(
        label="Submit", help="Submit dugme pokrece kreiranje Pinecone Indexa"
    )
    if submit_button and dokum and namespace:
        file = open(dokum.name, "r", encoding="utf-8")
        for line in file:
            # Remove leading/trailing whitespace and add to the list
            chunks.append(line.strip())
        # Initialize OpenAI and Pinecone API key
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
        PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")

        # initializing openai and pinecone
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
        index_name = "embedings1"

        # # embedding start !!!

        # Set the embedding model name
        embed_model = "text-embedding-ada-002"

        # Set the index name and namespace
        index_name = "embedings1"
        # Initialize the Pinecone index
        index = pinecone.Index(index_name)
        batch_size = 100  # how many embeddings we create and insert at once
        progress_text2 = "Upserting to Pinecone in progress. Please wait."
        progress_bar2 = st.progress(0.0, text=progress_text2)
        with open(dokum.name, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)

        # Now, 'data' contains the contents of the JSON file as a Python data structure (usually a dictionary or a list, depending on the JSON structure)
        # You can access the data and work with it as needed

        # For example, if 'data' is a list of dictionaries, you can iterate through it like this:

        ph2 = st.empty()
        for i in tqdm(range(0, len(data), batch_size)):
            # find end of batch
            i_end = min(len(chunks), i + batch_size)
            meta_batch = data[i:i_end]

            # get texts to encode
            ids_batch = [x["id"] for x in meta_batch]
            texts = [x["text"] for x in meta_batch]

            # create embeddings (try-except added to avoid RateLimitError)
            try:
                res = openai.Embedding.create(input=texts, engine=embed_model)

            except:
                done = False
                while not done:
                    sleep(5)
                    try:
                        res = openai.Embedding.create(input=texts, engine=embed_model)
                        done = True

                    except:
                        pass

            # cleanup metadata

            cleaned_meta_batch = []  # To store records without [nan] embeddings
            embeds = [record["embedding"] for record in res["data"]]

            # Check for [nan] embeddings

            if embeds:
                to_upsert = list(zip(ids_batch, embeds, meta_batch))
            else:
                err_log += f"Greska: {meta_batch}\n"
            # upsert to Pinecone
            err_log += f"Upserting {len(to_upsert)} embeddings\n"
            with open("err_log.txt", "w", encoding="utf-8") as file:
                file.write(err_log)
            index.upsert(vectors=to_upsert, namespace=namespace)
            stodva = len(data)
            if i_end > i:
                deo = i_end
            else:
                deo = i
            progress = deo / stodva
            l = int(deo / stodva * 100)

            ph2.text(f"Ucitano je {deo} od {stodva} linkova sto je {l} %")

            progress_bar2.progress(progress, text=progress_text2)

        # gives stats about index
        st.write("Napunjen pinecone")
        index = pinecone.Index(index_name)
        st.write(f"Sacuvano u Pinecone")
        pinecone_stats(index)


# Koristi se samo za deploy na streamlit.io
name, authentication_status, username = positive_login(main, "08.09.23")
