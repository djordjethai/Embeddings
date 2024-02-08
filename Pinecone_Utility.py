import streamlit as st
from pinecone import Pinecone
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
                    index_name = st.text_input(
                           "Unesite indeks : ", help="Unesite ime indeksa koji želite da vidite"
                            )
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
                    pinecone_stats(index, index_name)
                    st.write("Uspešno obrisano")
