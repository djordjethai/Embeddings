import streamlit as st
import pinecone
import sys
import os
import time
from myfunc.mojafunkcija import pinecone_stats, st_style


st_style()


def main():
    st.subheader('Uklanjanje namespace iz Pinecone indexa')
    col1, col2 = st.columns(2)
    with col1:
        with st.form(key='my_form', clear_on_submit=True):

            index_name = st.text_input("Unesi index : ", help = "Unesi ime indexa koji zelis da obrises")
            namespace = st.text_input("Unesi namespace : ", help = "Unesi namespace koji zelis da obrises")
            moj_filter = st.text_input("Unesi filter za source (prazno za sve) : ", help = "Unesi filter za source (prazno za sve) : ")
            nastavak = st.radio(
                f"Da li da uklonim namespace {namespace} iz indexa {index_name}",  ('Da', 'Ne'), help = "Da li da uklonim namespace iz indexa")

            submit_button = st.form_submit_button(label='Submit', help="Submit dugme pokrece uklanjanje namespace iz indexa")
            if submit_button:
                if not nastavak == "Da":
                    placeholder = st.empty()
                    with placeholder.container():
                        st.write("Izlazim iz programa")
                        time.sleep(2)
                        placeholder.empty()
                    sys.exit()
                else:
                    with st.spinner("Sacekajte trenutak..."):
                        PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
                        PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')
                        # initialize pinecone
                        pinecone.init(
                            api_key=PINECONE_API_KEY,
                            environment=PINECONE_API_ENV
                        )
                        index = pinecone.Index(index_name)

                        # ukoliko zelimo da izbrisemo samo nekle recorde bazirano na meta data
                        try:
                            if not moj_filter == "":
                                index.delete(
                                    filter={"person_name": {"$in": [moj_filter]}}, namespace=namespace)
                            else:
                                index.delete(delete_all=True,
                                             namespace=namespace)
                        except Exception as e:
                            st.error(
                                f"Proverite ime indexa koji ste uneli {e}")
                            sys.exit()

                with col2:
                    pinecone_stats(index)


