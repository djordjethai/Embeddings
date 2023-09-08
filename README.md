# Streamlit Utilities and Callbacks - `priprema.py`

This Python script, `priprema.py`, is a collection of Streamlit utilities and callback functions designed to enhance your Streamlit applications. These utilities and callbacks simplify common tasks, such as handling user interactions, displaying information, and integrating with other libraries.

## Overview

Streamlit is a powerful tool for creating interactive web applications with minimal effort. However, as your applications grow in complexity, you may need additional functionality and customization. This script serves as a utility library for enhancing your Streamlit projects.

## Key Features

Here are some key features and functions provided by this script:

- **Streamlit Style Customization**: Customize the style of your Streamlit app, including hiding the main menu and footer for a cleaner interface.

- **Positive Authentication**: Implement user authentication using the `streamlit-authenticator` library. Control access levels and user privileges based on predefined credentials.

- **Text File Handling**: Read text files from your local file system, which can be useful for loading data or configurations.

- **Streamlit Redirect**: Redirect the standard output of your Python code to Streamlit's interface. This can be handy for displaying real-time updates or logs.

- **Token Length Calculator**: Calculate the token length of a given text using the `tiktoken` library.

- **Pinecone Statistics**: Display statistics and information about a Pinecone index for embedding retrieval.

- **Data Flattening**: Flatten nested dictionaries into a more accessible format.

- **Streamlit Callbacks**: Implement Streamlit callbacks for handling user interactions and providing dynamic updates.

## Getting Started

To use this script in your Streamlit application, follow these steps:

1. Clone this repository or download the `priprema.py` file.

2. Import the necessary functions and classes from the script into your Streamlit application.

3. Utilize these utilities and callbacks to enhance your app's functionality.

4. Customize the script as needed to suit your specific requirements.

## Usage Guidelines

Refer to the documentation within the script and the comments provided for each function to understand how to use them effectively. You can also modify and extend these utilities to fit your project's needs.

For additional information and updates, please refer to the [Streamlit documentation](https://docs.streamlit.io/) and the relevant libraries used in this script.

---

## Script Details

- **Author**: Positive
- **Date**: 07.09.2023
- **License**: MIT

---

# Pinecone Namespace Removal - `pinecone_utility.py`

This script is designed to assist with the removal of a specified namespace from a Pinecone index. Pinecone is a real-time vector database that allows efficient storage and retrieval of embeddings, and this script provides a convenient way to manage the contents of a Pinecone index.

## Overview

In certain scenarios, you may need to clean up your Pinecone index by removing specific namespaces. Namespaces are useful for organizing and isolating data within an index, but there may come a time when you want to remove data associated with a particular namespace. This script streamlines that process.

## Key Features

Here are some key features and functionalities offered by this script:

- **Namespace Removal**: Remove a specified namespace from a Pinecone index, effectively deleting all associated data.
- **Filtering Options**: Optionally, you can apply a filter to select specific records within the namespace to remove, making the removal process more targeted.
- **Streamlit Interface**: The script provides a user-friendly Streamlit interface for easy interaction and execution.

## Getting Started

To use this script effectively, follow these steps:

1. Ensure you have Pinecone credentials set up with the required API key and environment variables.

2. Clone or download the script to your local machine.

3. Customize the script by specifying the target Pinecone index, the namespace to remove, and an optional filter for more precise removal.

4. Run the script using Streamlit to initiate the namespace removal process.

## Usage Guidelines

- **Input Fields**: Fill in the required input fields, including the Pinecone index name, the namespace to remove, and an optional filter based on metadata. You can specify a filter to remove only certain records from the namespace.

- **Confirmation**: The script will prompt you to confirm whether you want to proceed with the namespace removal. Ensure that you want to delete the specified namespace before confirming.

- **Execution**: Upon confirmation, the script will execute the removal process. All data associated with the specified namespace will be deleted from the Pinecone index.

- **Statistics**: After the removal process is complete, the script provides statistics and insights about the Pinecone index to give you an overview of the changes.

## Script Details

- **Author**: Positive
- **Date**: 07.09.2023
- **License**: MIT

Please note that this script is designed for use with Pinecone, and it's essential to have Pinecone credentials and permissions set up correctly.
