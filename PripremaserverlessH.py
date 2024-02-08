from ssl import ALERT_DESCRIPTION_CERTIFICATE_REVOKED
import streamlit as st

st.set_page_config(page_title="Embeddings", page_icon="📔", layout="wide")
from openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
import os
from myfunc.mojafunkcija import (
    st_style,
    positive_login,
    show_logo,
    def_chunk,
    pinecone_stats
)
import Pinecone_Utility
import ScrapperH
import PyPDF2
import io
import re
from pinecone_text.sparse import BM25Encoder
import datetime
import json
from uuid import uuid4
from io import StringIO
from pinecone import Pinecone

version = "07.02.24. 3072"
st_style()

api_key = os.environ.get("PINECONE_API_KEY")
host = os.environ.get("PINECONE_HOST")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def main():
    show_logo()
    chunk_size, chunk_overlap = def_chunk()
    #chunk_size = 50
    st.markdown(
        f"<p style='font-size: 10px; color: grey;'>{version}</p>",
        unsafe_allow_html=True,
    )
    st.subheader("Izaberite operaciju za Embeding HYBRID Method - Serverless")
    with st.expander("Pročitajte uputstvo:"):
        st.caption(
            """
                   Prethodni korak bio je kreiranje pitanja. To smo radili pomoću besplatnog ChatGPT modela. Iz svake oblasti (ili iz dokumenta)
                   zamolimo ChatGPT da kreira relevantna pitanja. Na pitanja mozemo da odgovorimo sami ili se odgovori mogu izvuci iz dokumenta.\n
                   Ukoliko zelite da vam model kreira odgovore, odaberite ulazni fajl sa pitanjma iz prethodnog koraka.
                   Opciono, ako je za odgovore potreban izvor, odaberite i fajl sa izvorom. Unesite sistemsku poruku (opis ponašanja modela)
                   i naziv FT modela. Kliknite na Submit i sačekajte da se obrada završi.
                   Fajl sa odgovorima ćete kasnije korisiti za kreiranje FT modela.\n
                   Pre prelaska na sledeću fazu OBAVEZNO pregledajte izlazni dokument sa odgovorima i korigujte ga po potrebi.
                   """
        )

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

    if "submit_b" not in st.session_state:
        st.session_state["submit_b"] = False

    if "submit_b2" not in st.session_state:
        st.session_state["submit_b2"] = False

    if "nesto" not in st.session_state:
        st.session_state["nesto"] = 0

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        with st.form(key="podeli", clear_on_submit=False):
            st.session_state.podeli_button = st.form_submit_button(
                label="Pripremi Dokument",
                use_container_width=True,
                help="Podela dokumenta na delove za indeksiranje",
            )
            if st.session_state.podeli_button:
                st.session_state.nesto = 1
    with col3:
        with st.form(key="kreiraj", clear_on_submit=False):
            st.session_state.kreiraj_button = st.form_submit_button(
                label="Kreiraj Embeding",
                use_container_width=True,
                help="Kreiranje Pinecone Indeksa",
            )
            if st.session_state.kreiraj_button:
                st.session_state.nesto = 2
    with col4:
     
        with st.form(key="manage", clear_on_submit=False):
            st.session_state.manage_button = st.form_submit_button(
                label="Upravljaj sa Pinecone",
                use_container_width=True,
                help="Manipulacije sa Pinecone Indeksom",
            )
            if st.session_state.manage_button:
                st.session_state.nesto = 3
    with col5:
        with st.form(key="stats", clear_on_submit=False):
            index_name = st.text_input(
                           "Unesite indeks : ", help="Unesite ime indeksa koji želite da vidite")
            st.session_state.stats_button = st.form_submit_button(
                label="Pokaži Statistiku",
                use_container_width=True,
                help="Statistika Pinecone Indeksa",
            )
            if st.session_state.stats_button:
                st.session_state.nesto = 4
            
            if index_name!="":
                if index_name == "positive":
                        PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY_POS_STARI")
                        pinecone=Pinecone(api_key=PINECONE_API_KEY, host="https://positive-882bcef.svc.us-west1-gcp-free.pinecone.io") #positive (medakovic, free)
                        index = pinecone.Index(host="https://positive-882bcef.svc.us-west1-gcp-free.pinecone.io") #positive
                elif index_name=="embedings1":
                        PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
                        pinecone=Pinecone(api_key=PINECONE_API_KEY, host="https://embedings1-b1b39e1.svc.us-west1-gcp.pinecone.io") #embedings1 (thai, free)
                        index = pinecone.Index(host="https://embedings1-b1b39e1.svc.us-west1-gcp.pinecone.io") #embedings1
                elif index_name=="neo-positive":
                        PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY_S")
                        pinecone=Pinecone(api_key=PINECONE_API_KEY, host="https://neo-positive-a9w1e6k.svc.apw5-4e34-81fa.pinecone.io") #neo-positive (thai, serverless, 3072)
                        index = pinecone.Index(host="https://neo-positive-a9w1e6k.svc.apw5-4e34-81fa.pinecone.io") #neo-positive
                elif index_name=="positive-s":
                        PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY_S")
                        pinecone=Pinecone(api_key=PINECONE_API_KEY, host="https://positive-s-a9w1e6k.svc.apw5-4e34-81fa.pinecone.io") #positive-s (thai, serverless, 1536)
                        index = pinecone.Index(host="https://positive-s-a9w1e6k.svc.apw5-4e34-81fa.pinecone.io") #positive-s  
                # with phmain.container():
                #         pinecone_stats(index, index_name)
            
            
    with col2:
   
        with st.form(key="screp", clear_on_submit=False):
            st.session_state.screp_button = st.form_submit_button(
                label="Pripremi Websajt", use_container_width=True, help="Scrape URL"
            )
            if st.session_state.screp_button:
                st.session_state.nesto = 5
    st.divider()
    phmain = st.empty()

    if st.session_state.nesto == 1:
        with phmain.container():
            prepare_embeddings(chunk_size, chunk_overlap)
    elif st.session_state.nesto == 2:
        with phmain.container():
            do_embeddings()
    elif st.session_state.nesto == 3:
        with phmain.container():
            Pinecone_Utility.main()
    elif st.session_state.nesto == 4:
        with phmain.container():
            # pinecone = Pinecone(api_key=api_key, host=host)
            # index_name = index_name
            # index = pinecone.Index(host=host)
            pinecone_stats(index, index_name)
    elif st.session_state.nesto == 5:
        with phmain.container():
            ScrapperH.main(chunk_size, chunk_overlap)


def prepare_embeddings(chunk_size, chunk_overlap):
    skinuto = False
    napisano = False

    file_name = "chunks.json"
    with st.form(key="my_form_prepare", clear_on_submit=False):
        st.subheader("Učitajte dokumenta i metadata za Pinecone Indeks")

        dokum = st.file_uploader(
            "Izaberite dokument/e", key="upload_file", type=["txt", "pdf", "docx"]
        )
        # define delimiter
        text_delimiter = st.text_input(
            "Unesite delimiter: ",
            help="Delimiter se koristi za podelu dokumenta na delove za indeksiranje. Prazno za paragraf",
        )
        # define prefix
        text_prefix = st.text_input(
            "Unesite prefiks za tekst: ",
            help="Prefiks se dodaje na početak teksta pre podela na delove za indeksiranje",
        )
        add_schema = st.radio(
            "Da li želite da dodate Schema Data (može značajno produžiti vreme potrebno za kreiranje): ",
            ("Da", "Ne"),
            key="add_schema_doc",
            help="Schema Data se dodaje na početak teksta",
        )
        st.session_state.submit_b = st.form_submit_button(
            label="Submit",
            help="Pokreće podelu dokumenta na delove za indeksiranje",
        )
        st.info(f"Chunk veličina: {chunk_size}, chunk preklapanje: {chunk_overlap}")
        if len(text_prefix) > 0:
            text_prefix = text_prefix + " "

        if dokum is not None and st.session_state.submit_b == True:
            with io.open(dokum.name, "wb") as file:
                file.write(dokum.getbuffer())

            if text_delimiter == "":
                text_delimiter = "\n\n"

            if ".pdf" in dokum.name:
                pdf_reader = PyPDF2.PdfReader(dokum)
                num_pages = len(pdf_reader.pages)
                text_content = ""

                for page in range(num_pages):
                    page_obj = pdf_reader.pages[page]
                    text_content += page_obj.extract_text()
                text_content = text_content.replace("•", "")
                text_content = re.sub(r"(?<=\b\w) (?=\w\b)", "", text_content)
                with io.open("temp.txt", "w", encoding="utf-8") as f:
                    f.write(text_content)

                loader = UnstructuredFileLoader("temp.txt", encoding="utf-8")
            else:
                # Creating a file loader object
                loader = UnstructuredFileLoader(dokum.name, encoding="utf-8")

            data = loader.load()

            # Split the document into smaller parts, the separator should be the word "Chapter"
            text_splitter = CharacterTextSplitter(
                separator=text_delimiter,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            texts = text_splitter.split_documents(data)


            # # Create the OpenAI embeddings
            st.write(f"Učitano {len(texts)} tekstova")

            # Define a custom method to convert Document to a JSON-serializable format
            output_json_list = []
            # Loop through the Document objects and convert them to JSON
            i = 0
            for document in texts:
                i += 1
                try:
                    if add_schema == "Da":
                        document.page_content = ScrapperH.add_schema_data(
                            document.page_content
                        )

                        with st.expander(
                            f"Obrađeni tekst: {i} od {len(texts)} ", expanded=False
                        ):
                            st.write(document.page_content)

                except Exception as e:
                    st.error("Schema nije na raspolaganju za ovaj chunk. {e}")

                # # Specify the file name where you want to save the data
                output_dict = {
                    "id": str(uuid4()),
                    "chunk": i,
                    "text": text_prefix + document.page_content,
                    "source": document.metadata.get("source", ""),
                    "date": datetime.datetime.now().strftime("%d.%m.%Y")
                }
                output_json_list.append(output_dict)

            # # Specify the file name where you want to save the JSON data
            json_string = (
                "["
                + ",\n".join(
                    json.dumps(d, ensure_ascii=False) for d in output_json_list
                )
                + "]"
            )

            # Now, json_string contains the JSON data as a string

            napisano = st.info(
                "Tekstovi su sačuvani u JSON obliku, downloadujte ih na svoj računar"
            )

    if napisano:
        file_name = os.path.splitext(dokum.name)[0]
        skinuto = st.download_button(
            "Download JSON",
            data=json_string,
            file_name=f"{file_name}.json",
            mime="application/json",
        )
    if skinuto:
        st.success(f"Tekstovi sačuvani na {file_name} su sada spremni za Embeding")



def do_embeddings():
    with st.form(key="my_form_do", clear_on_submit=False):
        dokum = st.file_uploader("Izaberite dokument/e", key="upload_json_file", type=[".json"], help="Izaberite dokument koji ste podelili na delove za indeksiranje")

        namespace = st.text_input("Unesi naziv namespace-a: ", help="Naziv namespace-a je obavezan za kreiranje Pinecone Indeksa")
        submit_b2 = st.form_submit_button(label="Submit", help="Pokreće kreiranje Pinecone Indeksa")

        if submit_b2 and dokum and namespace:
            stringio = StringIO(dokum.getvalue().decode("utf-8"))
            data = json.load(stringio)
            texts = [item['text'] for item in data]
            my_meta = [{key: value for key, value in item.items() if key != 'text'} for item in data]

            embeddings = []
            for text in texts:
                response = client.embeddings.create(
                    input=text,
                    model="text-embedding-3-large"
                )
                embedding = response.data[0].embedding
                embeddings.append(embedding)


            pinecone=Pinecone(api_key=api_key, host=host)
            index = pinecone.Index(host=host)

            upsert_data = [
                {"id": str(i), "values": embeddings[i], "metadata": my_meta[i]}
                for i in range(len(embeddings))
            ]

            index.upsert(vectors=upsert_data, namespace=namespace)

            st.info("Pinecone index updated successfully.")
            st.success("Data successfully saved in Pinecone.")






# Koristi se samo za deploy na streamlit.io
deployment_environment = os.environ.get("DEPLOYMENT_ENVIRONMENT")

if deployment_environment == "Streamlit":
    name, authentication_status, username = positive_login(main, " ")
else:
    if __name__ == "__main__":
        main()
