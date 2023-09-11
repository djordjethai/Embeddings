# This code scrapes a website, splits the text into chunks, and embeds them using OpenAI and Pinecone.

from tqdm.auto import tqdm
from uuid import uuid4
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import re
import html
import urllib.parse
from bs4 import BeautifulSoup
import requests
import openai
import pinecone
from bs4 import BeautifulSoup
import sys
import streamlit as st
from mojafunkcja import st_style
from Priprema import def_chunk
import json

st_style()


def scrape(url: str):
    global headers, sajt, err_log, tiktoken_len, vrsta
    # Send a GET request to the URL
    res = requests.get(url, headers=headers)

    # Check the response status code
    if res.status_code != 200:
        # If the status code is not 200 (OK), write the status code and return None
        err_log += f"{res.status_code} for {url}\n"
        return None

    # If the status code is 200, initialize BeautifulSoup with the response text
    soup = BeautifulSoup(res.text, "html.parser")
    # soup = BeautifulSoup(res.text, 'lxml')

    # Find all links to local pages on the website
    local_links = []
    for link in soup.find_all("a", href=True):
        if (
            link["href"].startswith(sajt)
            or link["href"].startswith("/")
            or link["href"].startswith("./")
        ):
            href = link["href"]
            base_url, extension = os.path.splitext(href)
            if not extension and not "mailto" in href and not "tel" in href:
                local_links.append(urllib.parse.urljoin(sajt, href))

                # Find the main content using CSS selectors
                try:
                    # main_content_list = soup.select('body main')
                    main_content_list = soup.select(vrsta)

                    # Check if 'main_content_list' is not empty
                    if main_content_list:
                        main_content = main_content_list[0]

                        # Extract the plaintext of the main content
                        main_content_text = main_content.get_text()

                        # Remove all HTML tags
                        main_content_text = re.sub(r"<[^>]+>", "", main_content_text)

                        # Remove extra white space
                        main_content_text = " ".join(main_content_text.split())

                        # Replace HTML entities with their corresponding characters
                        main_content_text = html.unescape(main_content_text)

                    else:
                        # Handle the case when 'main_content_list' is empty
                        main_content_text = "error"
                        err_log += f"Error in page structure, use body instead\n"
                        st.error(err_log)
                        sys.exit()
                except Exception as e:
                    err_log += f"Error while discivering page content\n"
                    return None

    # return as json
    return {"url": url, "text": main_content_text}, local_links


# Now you can work with the parsed content using Beautiful Soup


def main(chunk_size, chunk_overlap):
    with st.form(key="my_form_scrape", clear_on_submit=False):
        global res, err_log, headers, sajt, source, vrsta
        st.subheader("Pinecone Scraping")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }

        # Set the domain URL

        # with st.form(key="my_form", clear_on_submit=False):
        sajt = st.text_input("Unesi sajt : ")
        # prefix moze da se definise i dinamicki
        text_prefix = st.text_input(
            "Unesi prefix za tekst: ",
            help="Prefix se dodaje na pocetak teksta pre podela na delove za indeksiranje",
        )
        vrsta = st.radio("Unesi vrstu (default je body main): ", ("body main", "body"))
        # chunk_size, chunk_overlap = def_chunk()
        submit_button = st.form_submit_button(label="Submit")
        st.info(f"Chunk size: {chunk_size}, chunk overlap: {chunk_overlap}")
        if len(text_prefix) > 0:
            text_prefix = text_prefix + " "
        if submit_button and not sajt == "":
            res = requests.get(sajt, headers=headers)
            err_log = ""

            # Read OpenAI API key from file
            openai.api_key = os.environ.get("OPENAI_API_KEY")

            # Retrieving API keys from files
            PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

            # Setting the environment for Pinecone API
            PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")

            pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

            # Initialize BeautifulSoup with the response text
            soup = BeautifulSoup(res.text, "html.parser")
            # soup = BeautifulSoup(res.text, 'html5lib')

            # Define a function to scrape a given URL

            links = [sajt]
            scraped = set()
            data = []
            i = 0
            placeholder = st.empty()

            with st.spinner(f"Scraping "):
                # while True:
                while i < 3:
                    i += 1
                    if len(links) == 0:
                        st.success("URL list complete")
                        break
                    url = links[0]

                    # st.write(f'{url}, ">>", {i}')
                    placeholder.text(f"Obradjujem link broj {i}")
                    try:
                        res = scrape(url)
                        err_log += f" OK scraping {url}: {i}\n"
                    except Exception as e:
                        err_log += f"An error occurred while scraping {url}: page can not be scraped.\n"

                    scraped.add(url)

                    if res is not None:
                        page_content, local_links = res
                        data.append(page_content)
                        # add new links to links list
                        links.extend(local_links)
                        # remove duplicates
                        links = list(set(links))
                    # remove links already scraped
                    links = [link for link in links if link not in scraped]

                # Initialize RecursiveCharacterTextSplitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", " ", ""],
                )

            chunks = []
            progress_text = "Embeding creation in progress. Please wait."
            progress_bar = st.progress(0.0, text=progress_text)
            ph = st.empty()
            # Iterate over data records
            for idx, record in enumerate(tqdm(data)):
                # Split the text into chunks using the text splitter
                texts = text_splitter.split_text(record["text"])

                sto = len(data)
                odsto = idx + 1
                procenat = odsto / sto
                progress_bar.progress(procenat, text=progress_text)
                k = int(odsto / sto * 100)
                ph.text(f"Ucitano {odsto} od {sto} linkova sto je {k} % ")
                # Create a list of chunks for each text
                chunks.extend(
                    [
                        {
                            "id": str(uuid4()),
                            "text": text_prefix + texts[i],
                            "source": record["url"],
                        }
                        for i in range(len(texts))
                    ]
                )

            # Assuming 'chunks' is your list of dictionaries

            # Specify the file name where you want to save the JSON data
            json_file_path = "chunks.json"

            # # Save 'chunks' to a JSON file
            with open(json_file_path, "w", encoding="utf-8") as json_file:
                json_file.write("[ ")  # Start with an opening bracket

                for index, item in enumerate(chunks):
                    if index > 0:
                        json_file.write(
                            ",\n"
                        )  # Add a comma and newline for all except the first item
                    json.dump(item, json_file, ensure_ascii=False)

                json_file.write(" ]")  # End with a closing bracket

            st.success(
                f"Texts saved to {json_file_path} and are now ready for Embeddings"
            )


# if __name__ == "__main__":
#     main()
