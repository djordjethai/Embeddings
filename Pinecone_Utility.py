import streamlit as st
from pinecone import Pinecone
import sys
import os
import time
from myfunc.mojafunkcija import pinecone_stats, st_style
from langchain.indexes import GraphIndexCreator
from langchain_openai import ChatOpenAI
import networkx as nx
import matplotlib.pyplot as plt
from langchain.indexes.graph import NetworkxEntityGraph
from langchain.chains import GraphQAChain
import PyPDF2
import io
import re
from langchain_community.document_loaders import UnstructuredFileLoader
import unidecode


st_style()


def obrisi_index():
    
    index_name = st.selectbox("Odaberite index", ["neo-positive", "embedings1"], help="Unesite ime indeksa", key="opcije"
    )
    if index_name is not None and index_name!=" " and index_name !="" :
        col1, col2 = st.columns(2)
        if index_name=="embedings1":
           
            pinecone=Pinecone(api_key=os.environ.get("PINECONE_API_KEY_STARI"), host="https://embedings1-b1b39e1.svc.us-west1-gcp.pinecone.io") #embedings1 (thai, free)
            index = pinecone.Index(host="https://embedings1-b1b39e1.svc.us-west1-gcp.pinecone.io") #embedings1
        elif index_name=="neo-positive":
            
            pinecone=Pinecone(api_key=os.environ.get("PINECONE_API_KEY_S"), host="https://neo-positive-a9w1e6k.svc.apw5-4e34-81fa.pinecone.io") #neo-positive (thai, serverless, 3072)
            index = pinecone.Index(host="https://neo-positive-a9w1e6k.svc.apw5-4e34-81fa.pinecone.io") #neo-positive
        else:
            st.error("Index ne postoji")
            st.stop()
        with col2:
              pinecone_stats(index, index_name)
       
        with col1:
             with st.form(key="util", clear_on_submit=True):
                st.subheader("Uklanjanje namespace-a iz Pinecone Indeksa")
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
                        
                            # ukoliko zelimo da izbrisemo samo nekle recorde bazirano na meta data
                            try:
                                if not moj_filter == "":
                                    index.delete(
                                        filter={"person_name": {"$in": [moj_filter]}},
                                        namespace=namespace,
                                    )
                                #elif not namespace == "":
                                else:
                                    index.delete(delete_all=True, namespace=namespace)
                                # else:
                                #     index.delete(delete_all=True)
                            except Exception as e:
                                match = re.search(r'"message":"(.*?)"', str(e))

                                if match:
                                      # Prints the extracted message
                                    st.error(f"Proverite ime indeksa koji ste uneli: {match.group(1)}")
                                sys.exit()

                
                            st.success("Uspešno obrisano")

def create_node(id, label):
    ascii_label = unidecode.unidecode(label)  # Convert label to ASCII
    return f'  node [\n    id {id}\n    label "{ascii_label}"\n  ]\n'

# Function to create an edge entry
def create_edge(source, target, relation):
    ascii_relation = unidecode.unidecode(relation)  # Convert relation to ASCII
    return f'  edge [\n    source {source}\n    target {target}\n    relation "{ascii_relation}"\n  ]\n'

def create_graph_structure(data):
    graph = "graph [\n  directed 1\n"
    nodes = {}
    edges = []
    graph_structure = "graph [\n  directed 1\n"
    node_id = 0
    for item in data:
    # Check and add the first element of the tuple as a node
        if item[0] not in nodes:
            nodes[item[0]] = node_id
            graph_structure += create_node(node_id, item[0])
            node_id += 1
    
        # Check and add the second element of the tuple as a node
        if item[1] not in nodes:
            nodes[item[1]] = node_id
            graph_structure += create_node(node_id, item[1])
            node_id += 1

        # Create an edge based on the relation (item[2])
        graph_structure += create_edge(nodes[item[0]], nodes[item[1]], item[2])

# Close the graph structure
    graph_structure += "]"
    return graph_structure

# create graph 
def create_graph(dokum):
    skinuto = False
    napisano = False
    slika_grafa = False
    if "skinuto" not in st.session_state:
        st.session_state.skinuto = False
    
    if st.session_state.skinuto == False:
        with st.spinner("Kreiram Graf, molim vas sacekajte..."):
            buffer = io.BytesIO()
            # Write data to the buffer
            buffer.write(dokum.getbuffer())
            # Get the byte data from the buffer
            byte_data = buffer.getvalue()
            all_text = byte_data.decode('utf-8')
                  
            # initialize graph engine
            index_creator = GraphIndexCreator(llm=ChatOpenAI(temperature=0, model="gpt-4-turbo-preview"))
            text = "\n".join(all_text.split("\n\n"))
        
            # create graph
            graph = index_creator.from_text(text)
            prikaz = graph.get_triples()
            with st.expander("Graf:"):
                st.write(prikaz)
            # Don't forget to close the buffer when done
            buffer.close() 
            # save graph, with the same name different extension
            file_name = os.path.splitext(dokum.name)[0]
        
            graph_structure = create_graph_structure(prikaz)
            napisano = st.info(
                    f"Tekst je sačuvan u gml obliku kao {file_name}.gml, downloadujte ga na svoj računar"
            )
        
            skinut = st.download_button(
                "Download GML",
                data=graph_structure,
                file_name=f"{file_name}.gml",
                mime='ascii',
            )
            
            st.session_state.skinuto = True
            st.success(f"Graf je sačuvan kao {file_name}.gml")
                
        # Load the GML file
    
    
    if st.session_state.skinuto:
        st.markdown("**Kada downloadujete graph file, učitajte GRAF ako zelite graficki prikaz**")
        slika_grafa = st.file_uploader(
            "Izaberite GRAF dokument", key="upload_graf", type=["gml"]
        )      
            
        if slika_grafa is not None:
            with io.open(slika_grafa.name, "wb") as file:
                    file.write(slika_grafa.getbuffer())
            
            G = nx.read_gml(slika_grafa.name)
            nx.draw(G, with_labels=True, node_size=200, font_size=5)
            st.pyplot(plt)  # Display the plot in Streamlit
        

def read_uploaded_file(dokum, text_delimiter="space"):
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
    return data