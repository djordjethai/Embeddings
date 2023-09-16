
# Embeddings and Similarity Search

## Streamlit Utilities and Callbacks - `priprema.py`

This Python script, `priprema.py`, is a collection of Streamlit utilities and callback functions designed to enhance your Streamlit applications. These utilities and callbacks simplify common tasks, such as handling user interactions, displaying information, and integrating with other libraries.

## Overview - `priprema.py`

Streamlit is a powerful tool for creating interactive web applications with minimal effort. However, as your applications grow in complexity, you may need additional functionality and customization. This script serves as a utility library for enhancing your Streamlit projects.

## Key Features - `priprema.py`

Here are some key features and functions provided by this script:

- **Streamlit Style Customization**: Customize the style of your Streamlit app, including hiding the main menu and footer for a cleaner interface.

- **Positive Authentication**: Implement user authentication using the `streamlit-authenticator` library. Control access levels and user privileges based on predefined credentials.

- **Text File Handling**: Read text files from your local file system, which can be useful for loading data or configurations.

- **Streamlit Redirect**: Redirect the standard output of your Python code to Streamlit's interface. This can be handy for displaying real-time updates or logs.

- **Token Length Calculator**: Calculate the token length of a given text using the `tiktoken` library.

- **Pinecone Statistics**: Display statistics and information about a Pinecone index for embedding retrieval.

- **Data Flattening**: Flatten nested dictionaries into a more accessible format.

- **Streamlit Callbacks**: Implement Streamlit callbacks for handling user interactions and providing dynamic updates.

## Getting Started - `priprema.py`

To use this script in your Streamlit application, follow these steps:

1. Clone this repository or download the `priprema.py` file.

2. Import the necessary functions and classes from the script into your Streamlit application.

3. Utilize these utilities and callbacks to enhance your app's functionality.

4. Customize the script as needed to suit your specific requirements.

## Usage Guidelines - `priprema.py`

Refer to the documentation within the script and the comments provided for each function to understand how to use them effectively. You can also modify and extend these utilities to fit your project's needs.

For additional information and updates, please refer to the [Streamlit documentation](https://docs.streamlit.io/) and the relevant libraries used in this script.

## Script Details - `priprema.py`

- **Author**: Positive
- **Date**: 07.09.2023
- **License**: MIT

## Pinecone Namespace Removal - `pinecone_utility.py`

This script is designed to assist with the removal of a specified namespace from a Pinecone index. Pinecone is a real-time vector database that allows efficient storage and retrieval of embeddings, and this script provides a convenient way to manage the contents of a Pinecone index.

In certain scenarios, you may need to clean up your Pinecone index by removing specific namespaces. Namespaces are useful for organizing and isolating data within an index, but there may come a time when you want to remove data associated with a particular namespace. This script streamlines that process.

## Key Features - `pinecone_utility.py`

Here are some key features and functionalities offered by this script:

- **Namespace Removal**: Remove a specified namespace from a Pinecone index, effectively deleting all associated data.
- **Filtering Options**: Optionally, you can apply a filter to select specific records within the namespace to remove, making the removal process more targeted.
- **Streamlit Interface**: The script provides a user-friendly Streamlit interface for easy interaction and execution.

## Getting Started - `pinecone_utility.py`

To use this script effectively, follow these steps:

1. Ensure you have Pinecone credentials set up with the required API key and environment variables.

2. Clone or download the script to your local machine.

3. Customize the script by specifying the target Pinecone index, the namespace to remove, and an optional filter for more precise removal.

4. Run the script using Streamlit to initiate the namespace removal process.

## Usage Guidelines - `pinecone_utility.py`

- **Input Fields**: Fill in the required input fields, including the Pinecone index name, the namespace to remove, and an optional filter based on metadata. You can specify a filter to remove only certain records from the namespace.

- **Confirmation**: The script will prompt you to confirm whether you want to proceed with the namespace removal. Ensure that you want to delete the specified namespace before confirming.

- **Execution**: Upon confirmation, the script will execute the removal process. All data associated with the specified namespace will be deleted from the Pinecone index.

- **Statistics**: After the removal process is complete, the script provides statistics and insights about the Pinecone index to give you an overview of the changes.

## Script Details - `pinecone_utility.py`

- **Author**: Positive
- **Date**: 07.09.2023
- **License**: MIT

Please note that this script is designed for use with Pinecone, and it's essential to have Pinecone credentials and permissions set up correctly.

## Web Scraping and Text Embedding Tool - `scrapper.py`

This code is designed to scrape content from websites, split text into chunks, and embed them using OpenAI and Pinecone. It combines web scraping techniques, text processing, and integration with external APIs to facilitate data extraction and indexing.

## Overview - `scrapper.py`

The primary objective of this code is to extract textual data from web pages, segment it into manageable chunks, and prepare it for embedding into Pinecone, a vector database. It achieves this by parsing web pages, removing HTML tags, and dividing the text into smaller units for efficient indexing.

## Key Features - `scrapper.py`

Here are the key features and functionalities offered by this code:

- **Web Scraping**: Users can specify a website URL to scrape textual content from web pages. The code navigates the website, collects data, and prepares it for further processing.

- **Text Segmentation**: The code splits the extracted text into smaller chunks, making it suitable for embedding. Users can customize the chunk size and overlap to control the segmentation process.

- **Error Handling**: The code includes error handling mechanisms to handle cases where web pages cannot be scraped or if errors occur during the process.

- **Embedding Preparation**: After scraping and segmentation, the code prepares the text data for embedding, including adding a prefix to the text and creating a JSON file with the prepared data.

- **Pinecone Integration**: The code integrates with Pinecone, allowing users to efficiently index and search the prepared text data using Pinecone's capabilities.

## Getting Started - `scrapper.py`

To use this code for web scraping and text embedding, follow these steps:

1. Clone or download the code to your local machine.

2. Ensure you have the required Python libraries installed, including `requests`, `beautifulsoup4`, `tqdm`, `openai`, `pinecone`, `streamlit`, and any other dependencies.

3. Configure your Pinecone API key and environment variables to enable integration with Pinecone.

4. Run the code using Streamlit to initiate the web scraping and embedding process.

5. Enter the website URL you want to scrape, specify text prefix (if needed), and customize the text chunking parameters.

6. Review the progress as the code scrapes and prepares the text data.

7. Once completed, a JSON file containing the prepared data will be generated and ready for embedding into Pinecone.

## Usage Guidelines - `scrapper.py`

- **Website URL**: Enter the URL of the website you want to scrape in the provided input field.

- **Text Prefix**: Optionally, add a prefix to the extracted text to distinguish it during embedding.

- **Chunk Size and Overlap**: Customize the chunk size and overlap to control how the text is segmented.

- **Embedding**: After running the code, the prepared text data can be embedded into Pinecone for efficient indexing and searching.

## Script Details - `scrapper.py`

- **Author**: Positive
- **Date**: 07.09.2023
- **License**: MIT

Please note that this code relies on external APIs, including Pinecone and OpenAI, for specific functionalities. Proper configuration and API key setup are essential for the code to work correctly.
