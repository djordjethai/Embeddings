# this code is used to split the document into smaller parts and create OpenAI embeddings

import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader, UnstructuredPDFLoader
import os
import streamlit as st
from myfunc.mojafunkcija import st_style, pinecone_stats, positive_login



st.set_page_config(
    page_title="Embeddings",
    page_icon="👋",
    layout="wide"
)

st_style()


def main():
    with st.sidebar:
        st.image(
                "https://test.georgemposi.com/wp-content/uploads/2023/05/positive-logo-red.jpg", width=150)
    st.subheader('Izaberite operaciju za Embeddings')
    with st.expander("Procitajte uputstvo:"):
        st.caption("Prethodni korak bio je kreiranje pitanja. To smo radili pomocu besplatnog CHATGPT modela. Iz svake oblasti (ili iz dokumenta) zamolimo CHATGPT da kreira relevantna pitanja. Na pitanja mozemo da odgovorimo sami ili se odgovori mogu izvuci iz dokumenta.")
        st.caption("Ukoliko zelite da vam model kreira odgovore, odaberite ulazni fajl sa pitanjma iz prethodnog koraka. Opciono, ako je za odgovore potreban izvor, odaberite i fajl sa izvorom. Unesite sistemsku poruku (opis ponasanja modela) i naziv FT modela. Kliknite na Submit i sacekajte da se obrada zavrsi. Fajl sa odgovorima cete kasnije korisiti za kreiranje FT modela.")
        st.caption(
            "Pre prelaska na sledecu fazu OBAVEZNO pregledajte izlazni dokument sa odgovorima i korigujte ga po potrebi. ")
 
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        with st.form(key='podeli', clear_on_submit=False):
            podeli_button = st.form_submit_button(
                label='Podeli dokument u delove za indeksiranje', use_container_width=True, help = "Podela dokumenta na delove za indeksiranje")

    with col2:
        with st.form(key='kreiraj', clear_on_submit=False):
            kreiraj_button = st.form_submit_button(
                label='Kreiraj Pinecone Index', use_container_width=True, help = "Kreiranje Pinecone Indexa")
    with col3:
        with st.form(key='manage', clear_on_submit=False):
            manage_button = st.form_submit_button(
                label='Manage Pinecone Index', use_container_width=True, help = "Manage Pinecone Indexa")
    with col4:
        with st.form(key='stats', clear_on_submit=False):
            stats_button = st.form_submit_button(
                label='Pokazi stats za Pinecone Index', use_container_width=True, help = "Stats za Pinecone Indexa")
    st.divider()
    if podeli_button:
        prepare_embeddings()
    if kreiraj_button:
        do_embeddings()
    if manage_button:
        import Pinecone_Utility
        Pinecone_Utility.main()
    if stats_button:
        from mojafunkcija import pinecone_stats
        import pinecone
        index = pinecone.Index('embedings1')
        pinecone_stats(index)


def prepare_embeddings():
    st.subheader('Upload documents and metadata for Pinecone Index')
    with st.form(key='my_form', clear_on_submit=False):
        # Streamlit app
        dokum = st.file_uploader(
            "Izaberite dokument/e", key="upload_file", type=['txt', 'pdf', 'docx'])
        chunk_size = st.slider(
            'Set chunk size in characters (200 - 8000)', 200, 8000, 1500, step=100, help="Velicina chunka odredjuje velicinu indeksiranog dokumenta. Veci chunk obezbedjuje bolji kontekst, dok manji chunk omogucava precizniji odgovor.")
        chunk_overlap = st.slider(
            'Set overlap size in characters (0 - 1000), must be less than the chunk size', 0, 1000, 0, step=10, help="Velicina overlapa odredjuje velicinu preklapanja sardzaja dokumenta. Veci overlap obezbedjuje bolji prenos konteksta.")

        submit_button = st.form_submit_button(label='Submit', help="Submit dugme pokrece podelu dokumenta na delove za indeksiranje")

        if chunk_overlap < chunk_size and submit_button and dokum:
            # Load the file
            if ".pdf" in dokum.name:
                loader = UnstructuredPDFLoader(
                    dokum.name, encoding="utf-8")
            else:
                # Creating a file loader object
                loader = UnstructuredFileLoader(
                    dokum.name, encoding="utf-8")

            data = loader.load()  # Loading text from the file

            # Split the document into smaller parts
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            texts = text_splitter.split_documents(data)

            # Ask the user if they want to do OpenAI embeddings

            # Create the OpenAI embeddings
            st.write(f'Ucitano {len(texts)} tekstova')

            with open('output.txt', 'w', encoding='utf-8') as file:
                for text in texts:
                    # Write each text segment to the file on a new line
                    file.write(text + '\n')
            st.write(f'Sacuvano u output.txt')


def do_embeddings():
    # Read the texts from the .txt file
    texts = []
    dokum = st.file_uploader(
        "Izaberite dokument/e", key="upload_file", type=['txt', 'pdf', 'docx'], help="Izaberite dokument koji ste podelili na delove za indeksiranje")


# Now, you can use stored_texts as your texts
    with st.form(key='my_form2', clear_on_submit=False):
        namespace = st.text_input('Unesi naziv namespace-a: ', help="Naziv namespace-a je obavezan za kreiranje Pinecone Indexa")
        submit_button = st.form_submit_button(label='Submit', help="Submit dugme pokrece kreiranje Pinecone Indexa")
        if submit_button and dokum and namespace:
            file = open(dokum.name, 'r', encoding='utf-8')
            for line in file:
                # Remove leading/trailing whitespace and add to the list
                texts.append(line.strip())
            # Initialize OpenAI and Pinecone API key
            OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
            PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
            PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

            # initializing openai and pinecone
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            pinecone.init(
                api_key=PINECONE_API_KEY,
                environment=PINECONE_API_ENV
            )
            index_name = 'embedings1'
            docsearch = Pinecone.from_texts(
                [t.page_content for t in texts], embeddings, index_name=index_name, namespace=namespace)
            st.write('Napunjen pinecone')
            index = pinecone.Index(index_name)
            st.write(f'Sacuvano u Pinecone')
            pinecone_stats(index)


name, authentication_status, username = positive_login(main, "08.09.23")
