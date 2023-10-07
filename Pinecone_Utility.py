import streamlit as st
import pinecone
import sys
import os
import time
from myfunc.mojafunkcija import pinecone_stats, st_style


st_style()


def main():
    with st.form(key="util", clear_on_submit=True):
        st.subheader("Uklanjanje namespace-a iz Pinecone Indeksa")
        col1, col2 = st.columns(2)
        with col1:
            # with st.form(key='my_form', clear_on_submit=True):

            index_name = st.text_input(
                "Unesite indeks : ", help="Unesite ime indeksa koji želite da obrišete"
            )
            namespace = st.text_input(
                "Unesite namespace : ",
                help="Unesite namespace koji želite da obrišete (prazno za sve)",
            )
            moj_filter = st.text_input(
                "Unesite filter za source (prazno za sve) : ",
                help="Unesite filter za source (prazno za sve) : ",
            )
            nastavak = st.radio(
                f"Da li ukloniti namespace {namespace} iz indeksa {index_name}",
                ("Da", "Ne"),
                help="Da li ukloniti namespace iz indeksa?",
            )

            submit_button = st.form_submit_button(
                label="Submit",
                help="Pokreće uklanjanje namespace-a iz indeksa",
            )
            if submit_button:
                if not nastavak == "Da":
                    placeholder = st.empty()
                    with placeholder.container():
                        st.write("Izlazim iz programa")
                        time.sleep(2)
                        placeholder.empty()
                    sys.exit()
                else:
                    with st.spinner("Sačekajte trenutak..."):
                        if index_name == "positive-hybrid":
                            PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY_POS")
                            PINECONE_API_ENV = os.environ.get(
                                "PINECONE_ENVIRONMENT_POS"
                            )
                        else:
                            PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
                            PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")
                        # initialize pinecone
                        pinecone.init(
                            api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV
                        )
                        index = pinecone.Index(index_name)

                        # ukoliko zelimo da izbrisemo samo nekle recorde bazirano na meta data
                        try:
                            if not moj_filter == "":
                                index.delete(
                                    filter={"person_name": {"$in": [moj_filter]}},
                                    namespace=namespace,
                                )
                            elif not namespace == "":
                                index.delete(delete_all=True, namespace=namespace)
                            else:
                                index.delete(delete_all=True)
                        except Exception as e:
                            st.error(f"Proverite ime indeksa koji ste uneli {e}")
                            sys.exit()

                with col2:
                    pinecone_stats(index)
                    st.write("Uspešno obrisano")
