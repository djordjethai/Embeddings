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
    pinecone_stats,
)
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder
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
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
            
            
            

st_style()
index_name="neo-positive"
api_key = os.environ.get("PINECONE_API_KEY")
host = os.environ.get("PINECONE_HOST")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
version = "14.02.24"


def add_self_data(line):
    """
    Extracts the person's name and topic from a given line of text using a GPT-4 model.

    This function sends a request to a GPT-4 model with a specific prompt that instructs the model to use JSON format
    for extracting a person's name ('person_name') and a topic from the provided text ('line'). The prompt includes instructions
    to use the Serbian language for extraction. If the model cannot decide on a name, it is instructed to return 'John Doe'.

    Parameters:
    - line (str): A line of text from which the person's name and topic are to be extracted.

    Returns:
    - tuple: A tuple containing the extracted person's name and topic. If the extraction is successful, it returns
      (person_name, topic). If the model cannot decide on a name, it returns ('John Doe', topic).

    Note:
    The function assumes that the response from the GPT-4 model is in a JSON-compatible format and that the keys
    'person_name' and 'topic' are present in the JSON object returned by the model.
    """
    
    system_prompt = "Use JSON format to extract person_name and topic. Extract the pearson name and the topic. If you can not decide on the name, return 'John Doe'. Use the Serbian language "
    
    response = client.chat.completions.create(
                        model="gpt-4-turbo-preview",
                        temperature=0,
                        response_format = { "type": "json_object" },
                        messages=[
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {
                                "role": "user",
                                "content": line
                            }
                            
                        ]
                    )
    json_content = response.choices[0].message.content.strip()
    content_dict = json.loads(json_content)
    person_name = content_dict.get("person_name", "John Doe")  # Fallback to "John Doe" if "person_name" is not found
    topic = content_dict["topic"]
    
    return person_name, topic

def format_output_text(prefix, question, content):
    return prefix + question + content

def get_current_date_formatted():
    return datetime.datetime.now().strftime("%d.%m.%Y")


def add_question(chunk_text):
    """
    Ads a question to a chunk of text that will best match given statement and get the most similar content as the input if asked.
    input chunk_text is a source text used to create question as a string
    output is the question as a string
    """
    result = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "[Use only Serbian language] You are a top interviewer. Create a question that will best match given statement and get the most similar content as the input if asked."
            },
            {
                "role": "user",
                "content": chunk_text
            }
        ]
    )

    return result.choices[0].message.content
    

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
                   Korisničko uputstvo za rad sa aplikacijom za Embeddings za Hybrid Search

**Uvod**
Ovo uputstvo je namenjeno korisnicima koji žele da koriste aplikaciju za kreiranje i upravljanje embeddings-ima koristeći Streamlit interfejs i Pinecone servis. Aplikacija omogućava pripremu dokumenata, kreiranje embeddings-a, upravljanje Pinecone indeksima i prikaz statistike.

***Aplikacija puni index prilagodjen Hybrid Search-u***

**Priprema dokumenata**
1. Odaberite opciju "Pripremi Dokument" u aplikaciji.
2. Učitajte dokument(e) koji želite da obradite. Podržani formati su `.txt`, `.pdf`, `.docx`.
3a. Definišite delimiter za podelu dokumenta na delove, prefiks koji će biti dodat na početak svakog dela, i opcionalno, da li želite dodavanje metapodataka i pitanja u tekst.
***Ako ostavite polje prazno, default delimiter je novi paragraf***
3b. Opciono definišite prefikks koji ce biti dodat na pocetak teksta.
3c. Za Self_query model ce definisati dodatne meta podatke person name i topic.
3d. Mozete naloziti modelu da definise pitanje za svaki tekst. Pitanje se dodaje polse prefiksa a pre teksta.
3e. Mozete zatraziti Semantic chunking dokumenta, podelu i spajanje pasusa prema znacenju. U tom slucaju velicina chunka i overlap nisu u upotrebi

Duzina chunka i overlap se mogu podesavati po potrebi. Generalno, ako vam j edokument vec struktuiran po paragrafima mozete korstiti vema male duzine (50) a podela ce se izvrsiti na prvom paragrafu. 
Default duzina chunka je 1500 karaktera. Testirajte koji vam parametri najvise odgovaraju za odredjeni tip teksta.

4. Kliknite na "Submit" da pokrenete obradu dokumenta. Dokument će biti podeljen na delove za indeksiranje.
Pre prelaska na sledeću fazu OBAVEZNO pregledajte izlazni dokument sa odgovorima i korigujte ga po potrebi.

**Kreiranje Embeddings-a**
1. Nakon pripreme dokumenata, odaberite opciju "Kreiraj Embeding".
2. Učitajte JSON fajl sa delovima teksta pripremljenim u prethodnom koraku.
3. Unesite naziv namespace-a za Pinecone indeks.
4. Kliknite na "Submit" da pokrenete kreiranje embeddings-a. Tekstovi će biti procesirani i sačuvani u Pinecone indeksu.

**Upravljanje sa Pinecone**
Odaberite opciju "Upravljaj sa Pinecone" za manipulaciju sa Pinecone indeksom. Funkcije uključuju brisanje podataka prema nazivu namespace-a I opciono filter a za metadata.

**Prikaz Statistike**
Odaberite opciju "Pokaži Statistiku" za pregled statističkih podataka Pinecone indeksa. Potrebno je uneti naziv indeksa za koji želite da vidite statistiku naziv namespace-ova i broj vektora.

**Zaključak**
Po završetku rada sa aplikacijom, možete preuzeti rezultate obrade u JSON formatu i koristiti ih dalje u svojim projektima. 

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
            "Da li želite da dodate Metadata (Dodaje ime i temu u metadata): ",
            ("Ne", "Da"),
            key="add_schema_doc",
            help="Dodaje u metadata ime i temu",
        )
        add_pitanje = st.radio(
            "Da li želite da dodate pitanje: ",
            ("Ne", "Da"),
            key="add_pitanje_doc",
            help="Dodaje pitanje u text",
        )
        semantic = st.radio(
            "Da li želite semantic chunking: ",
            ("Ne", "Da"),
            key="semantic",
            help="Greg Kamaradt Semantic Chunker",
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
            if semantic == "Da":
                text_splitter = SemanticChunker(OpenAIEmbeddings())
            else:
                text_splitter = CharacterTextSplitter(
                        separator=text_delimiter,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )

            texts = text_splitter.split_documents(data)


            # # Create the OpenAI embeddings
            st.success(f"Učitano {len(texts)} tekstova")

            # Define a custom method to convert Document to a JSON-serializable format
            output_json_list = []
            
            # Loop through the Document objects and convert them to JSON
            i = 0
            for document in texts:
                i += 1
                if add_pitanje=="Da":
                    pitanje = add_question(document.page_content) + " "
                    st.info(f"Dodajem pitanje u tekst {i}")
                else:
                    pitanje = ""
      
                output_dict = {
                    "id": str(uuid4()),
                    "chunk": i,
                    "text": format_output_text(text_prefix, pitanje, document.page_content),
                    "source": document.metadata.get("source", ""),
                    "date": get_current_date_formatted(),
                }

                if add_schema == "Da":
                    try:
                        person_name, topic = add_self_data(document.page_content)
                    except Exception as e:
                        st.write(f"An error occurred: {e}")
                        person_name, topic = "John Doe", "Any"
    
                    output_dict["person_name"] = person_name
                    output_dict["topic"] = topic
                    st.success(f"Processing {i} of {len(texts)}, {person_name}, {topic}")

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
    
        namespace = st.text_input(
            "Unesi naziv namespace-a: ",
            help="Naziv namespace-a je obavezan za kreiranje Pinecone Indeksa",
        )
        submit_b2 = st.form_submit_button(
            label="Submit", help="Pokreće kreiranje Pinecone Indeksa"
        )
        if submit_b2 and dokum and namespace:
            stringio = StringIO(dokum.getvalue().decode("utf-8"))

            # Directly load the JSON data from file content
            data = json.load(stringio)

            # Initialize lists outside the loop
            my_list = []
            my_meta = []

            # Process each JSON object in the data
            for item in data:
                # Append the text to my_list
                my_list.append(item['text'])
    
                # Append other data to my_meta
                meta_data = {key: value for key, value in item.items() if key != 'text'}
                my_meta.append(meta_data)
                
            pinecone=Pinecone(api_key=api_key, host=host)
            index = pinecone.Index(host=host)
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            
            # upsert data
            bm25_encoder = BM25Encoder()
            # fit tf-idf values on your corpus
            bm25_encoder.fit(my_list)

            retriever = PineconeHybridSearchRetriever(
                embeddings=embeddings,
                sparse_encoder=bm25_encoder,
                index=index,
            )

            retriever.add_texts(texts=my_list, metadatas=my_meta, namespace=namespace)
            
            # gives stats about index
            st.info("Napunjen Pinecone")

            st.success(f"Sačuvano u Pinecone-u")
            pinecone_stats(index, index_name)




# Koristi se samo za deploy na streamlit.io
deployment_environment = os.environ.get("DEPLOYMENT_ENVIRONMENT")

if deployment_environment == "Streamlit":
    name, authentication_status, username = positive_login(main, " ")
else:
    if __name__ == "__main__":
        main()