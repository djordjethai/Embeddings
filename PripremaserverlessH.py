import streamlit as st
from openai import OpenAI
import os
from myfunc.mojafunkcija import (
    st_style,
    positive_login,
    show_logo,
    def_chunk,
)
from myfunc.retrievers import PineconeUtility
from myfunc.various_tools import do_embeddings, prepare_embeddings, main_scraper
from pinecone import Pinecone

st.set_page_config(page_title="Embeddings", page_icon="📔", layout="wide")

if "init_prompts" not in st.session_state:
    st.session_state.init_prompts = True
    from myfunc.retrievers import PromptDatabase
    with PromptDatabase() as db:
        prompt_map = db.get_prompts_by_names(["system_prompt_add_self", "system_prompt_add_q"],[os.getenv("ADD_SELF_DATA"), os.getenv("MULTI_H_QA_SYSTEM")])
        st.session_state.system_prompt_add_self = prompt_map.get("system_prompt_add_self", "You are helpful assistant that always writes in Sebian.")
        st.session_state.system_prompt_add_q = prompt_map.get("system_prompt_add_q", "You are helpful assistant that always writes in Sebian.")

st_style()
index_name="neo-positive"
#api_key = os.environ.get("PINECONE_API_KEY")
host = os.environ.get("PINECONE_HOST")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
version = "25.03.24"


def main():
    show_logo()
    chunk_size, chunk_overlap = def_chunk()
    #chunk_size = 50
    st.markdown(
        f"<p style='font-size: 10px; color: grey;'>{version}</p>",
        unsafe_allow_html=True,
    )
    st.subheader("Embedings - Hybrid (neo-positive) i Semantic (embedings1)")
    st.caption("Odaberite operaciju")
    with st.expander("Pročitajte uputstvo:"):
        st.caption(
            """
                   Korisničko uputstvo za rad sa aplikacijom za Embeddings

**Uvod**
Ovo uputstvo je namenjeno korisnicima koji žele da koriste aplikaciju za kreiranje i upravljanje embeddings-ima koristeći Streamlit interfejs i Pinecone servis. Aplikacija omogućava pripremu dokumenata, kreiranje embeddings-a, upravljanje Pinecone indeksima i prikaz statistike.

***Aplikacija puni index prilagodjen Hybrid Search-u***

**Priprema dokumenata**
1. Odaberite opciju "Pripremi Dokument" u aplikaciji.
2. Iz side bar-a učitajte dokument(e) koji želite da obradite. Podržani formati su `.txt`, `.pdf`, `.docx`. Definisite duzini chunka i overlap.
3a. Definišite delimiter za podelu dokumenta na delove, prefiks koji će biti dodat na početak svakog dela, i opcionalno, da li želite dodavanje metapodataka i pitanja u tekst.
***Ako ostavite polje prazno, default delimiter je novi paragraf***
3b. Opciono definišite prefikks koji ce biti dodat na pocetak teksta.
3c. Za Self_query model ce definisati dodatne meta podatke person name i topic.
3d. Mozete naloziti modelu da definise pitanje za svaki tekst. Pitanje se dodaje polse prefiksa a pre teksta.
3e. Mozete zatraziti Semantic chunking dokumenta, podelu i spajanje pasusa prema znacenju. U tom slucaju velicina chunka i overlap nisu u upotrebi

Duzina chunka i overlap se mogu podesavati po potrebi. Generalno, ako vam j edokument vec struktuiran po paragrafima mozete korstiti vema male duzine (50) a podela ce se izvrsiti na prvom paragrafu. 
Default duzina chunka je 1500 karaktera. Testirajte koji vam parametri najvise odgovaraju za odredjeni tip teksta.

4. Kliknite na "Submit" da pokrenete obradu dokumenta. Dokument će biti podeljen na delove za indeksiranje.
Pre prelaska na sledeću fazu OBAVEZNO uploadujte i pregledajte izlazni dokument sa odgovorima i korigujte ga po potrebi.

**Priprema websajta**
1. Odaberite opciju "Pripremi Websajt" u aplikaciji.
2. Unestite url.
3. Definisite da li citate body ili body main objekat.

Duzina chunka i overlap se mogu podesavati po potrebi. Generalno, ako vam j edokument vec struktuiran po paragrafima mozete korstiti vema male duzine (50) a podela ce se izvrsiti na prvom paragrafu. 
Default duzina chunka je 1500 karaktera. Testirajte koji vam parametri najvise odgovaraju za odredjeni tip teksta.

4. Kliknite na "Submit" da pokrenete obradu dokumenta. Dokument će biti podeljen na delove za indeksiranje.
Pre prelaska na sledeću fazu OBAVEZNO uploadujte i pregledajte izlazni dokument sa odgovorima i korigujte ga po potrebi.

**Kreiranje Knowledge Graph-a**
1. Iz side bar-a učitajte fajl sa tekstom koji želite da obradite.
2. Graf ime_originalnog_fajla.gml fajl ce biti sacuvan lokalno. Videcete i graficku reprezentaciju Knowledge Graph-a.


**Kreiranje Embeddings-a  - vazi za Hybrid (neo-positive) i Semantic (embedings1) indexe **
1. Nakon pripreme dokumenata, odaberite opciju "Kreiraj Embeding".
2. Iz side bar-a učitajte JSON fajl sa delovima teksta pripremljenim u prethodnom koraku.
3. Odaberite Index i unesite naziv namespace-a (neo_positive=Hybrid, embedings1=Semantic).
4. Kliknite na "Submit" da pokrenete kreiranje embeddings-a. Tekstovi će biti procesirani i sačuvani u odabranom Pinecone indeksu.

**Upravljanje sa Pinecone**

1.Odaberite naziv Indexa
2. Odaberite opciju "Upravljaj sa Pinecone" za manipulaciju sa Pinecone indeksom. Funkcije uključuju brisanje podataka prema nazivu namespace-a I opciono filter a za metadata.
3. **Prikaz Statistike** je pregled statističkih podataka Pinecone indeksa, naziv namespace-ova i broj vektora.
                   """
        )
    st.caption("Odaberite operaciju")
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
    with st.sidebar:
        st.subheader("Učitajte dokument za pripremu Pinecone Indeksa")

        dokum = st.file_uploader(
            "Izaberite dokument/e", key="upload_file", type=["txt", "pdf", "docx", "JSON"]
        )
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
        with st.form(key="graf", clear_on_submit=False):
            st.session_state.podeli_button = st.form_submit_button(
                label="Kreiraj Knowledge Graph",
                use_container_width=True,
                help="Kreiranje Knowledge Graph-a",
            )
            if st.session_state.podeli_button:
                st.session_state.nesto = 6            
    with col4:
        with st.form(key="kreiraj", clear_on_submit=False):
            st.session_state.kreiraj_button = st.form_submit_button(
                label="Kreiraj Embeding",
                use_container_width=True,
                help="Kreiranje Pinecone Indeksa",
            )
            if st.session_state.kreiraj_button:
                st.session_state.nesto = 2
    with col5:
     
        with st.form(key="manage", clear_on_submit=False):
            st.session_state.manage_button = st.form_submit_button(
                label="Upravljaj sa Pinecone",
                use_container_width=True,
                help="Manipulacije sa Pinecone Indeksom",
            )
            if st.session_state.manage_button:
                st.session_state.nesto = 3
            
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
            if dokum is not None: 
                prepare_embeddings(chunk_size, chunk_overlap, dokum)
            else:
                st.error("Uploadujte dokument")
    elif st.session_state.nesto == 2:
        with phmain.container():
            if dokum is not None: 
                index_name = st.selectbox("Odaberite index", ["neo-positive", "embedings1"], help="Unesite ime indeksa", key="opcije"
                )
                if index_name is not None and index_name!=" " and index_name !="" :
                                            
                        if index_name=="embedings1":
        
                            pinecone=Pinecone(api_key=os.environ.get("PINECONE_API_KEY_STARI"), host="https://embedings1-b1b39e1.svc.us-west1-gcp.pinecone.io") #embedings1 (thai, free)
                            index = pinecone.Index(host="https://embedings1-b1b39e1.svc.us-west1-gcp.pinecone.io") #embedings1
                            do_embeddings(dokum, "semantic", os.environ.get("PINECONE_API_KEY_STARI"), host="https://embedings1-b1b39e1.svc.us-west1-gcp.pinecone.io", index_name=index_name, index=index)
                        elif index_name=="neo-positive":
        
                            pinecone=Pinecone(api_key=os.environ.get("PINECONE_API_KEY_S"), host="https://neo-positive-a9w1e6k.svc.apw5-4e34-81fa.pinecone.io") #neo-positive (thai, serverless, 3072)
                            index = pinecone.Index(host="https://neo-positive-a9w1e6k.svc.apw5-4e34-81fa.pinecone.io") #neo-positive
                            do_embeddings(dokum=dokum, tip="hybrid", api_key=os.environ.get("PINECONE_API_KEY_S"), host="https://neo-positive-a9w1e6k.svc.apw5-4e34-81fa.pinecone.io", index_name=index_name, index=index )
                            
                        else:
                            st.error("Index ne postoji")
                            st.stop()
            else:
                st.error("Uploadujte JSON dokument")
    elif st.session_state.nesto == 3:
        with phmain.container():
            PineconeUtility.obrisi_index()
    
    elif st.session_state.nesto == 5:
        with phmain.container():
            main_scraper(chunk_size, chunk_overlap)
    elif st.session_state.nesto == 6:
        with phmain.container():
            if dokum is not None: 
                PineconeUtility.create_graph(dokum)
            else:
                st.error("Uploadujte dokument")


# Koristi se samo za deploy na streamlit.io
deployment_environment = os.environ.get("DEPLOYMENT_ENVIRONMENT")

if deployment_environment == "Streamlit":
    name, authentication_status, username = positive_login(main, " ")
else:
    if __name__ == "__main__":
        main()