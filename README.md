# OpenAI Fine-Tuning Toolkit

Welcome to the OpenAI Fine-Tuning Toolkit! This repository contains a collection of Python scripts (.py files) designed to streamline and simplify the fine-tuning process for OpenAI's GPT-3.5 Turbo model. Whether you're looking to fine-tune the model for specific tasks, monitor job progress, or manage fine-tuned models, this toolkit has you covered.

## Overview

Fine-tuning the GPT-3.5 Turbo model allows you to customize it for various natural language understanding and generation tasks. OpenAI's fine-tuning capabilities empower you to create chatbots, provide recommendations, answer questions, and more, all tailored to your specific application.

This toolkit includes several Python scripts, each serving a specific purpose in the fine-tuning workflow. Below is an overview of the key components:

1. **Data Preparation:** Ensure that your training data is in the right format, within token limits, and ready for fine-tuning.

2. **Model Creation:** Upload your data, initiate fine-tuning, and specify model details, such as the name and organization.

3. **Job Management:** Keep track of fine-tuning jobs, retrieve job states, cancel ongoing jobs, and list events associated with a particular job.

4. **Model Deletion:** Delete fine-tuned models that are no longer needed.

5. **Model Listing:** List available models for reference.

## Getting Started

To get started with fine-tuning using this toolkit, follow these steps:

1. Clone this repository to your local machine.

2. Set up your OpenAI API key by defining it as an environment variable (`OPENAI_API_KEY`). You can obtain an API key by signing up on the OpenAI platform.

3. Install the required Python dependencies specified in each .py file.

4. Navigate to the specific .py file that corresponds to the task you want to perform and follow the instructions provided.

5. Refer to the individual .py file descriptions for detailed usage guidelines.

## Script Descriptions

Below, you'll find descriptions and usage guidelines for each Python script included in this toolkit:

---

Feel free to explore each script based on your specific fine-tuning needs. Whether you're new to fine-tuning or an experienced user, this toolkit aims to simplify the process and make it accessible to all.

For additional information and updates, please refer to the [OpenAI documentation](https://beta.openai.com/docs/).

---

## Fine-Tuning GPT-3.5 Turbo Model - `Fine_tuning_turbo.py`

This Python script is designed to prepare and run fine-tuning for the GPT-3.5 Turbo model provided by OpenAI. It is integrated into a Streamlit web application for ease of use. The script performs various tasks related to fine-tuning, including data verification, model creation, monitoring job status, and more.

### Features and Functionality:

1. **Data Verification:**
   - Allows users to upload a JSONL file containing question-answer pairs for data verification.
   - Checks the data structure to ensure it complies with the Chat completions message structure.
   - Verifies the token count to ensure it does not exceed the 4096 token limit.
   - Provides pricing and default epoch estimates based on the dataset.

2. **Create Fine-Tuned Model:**
   - Users can upload a JSONL file for creating a fine-tuned model.
   - Validates the uploaded training and validation data files.
   - Allows users to specify a suffix for the model's name.
   - Initiates the fine-tuning process using the specified data and model.

3. **List Fine-Tuning Jobs:**
   - Displays a list of up to 10 fine-tuning jobs.

4. **Retrieve Fine-Tuning Job State:**
   - Allows users to retrieve the state of a specific fine-tuning job using its ID.

5. **Cancel Fine-Tuning Job:**
   - Provides an option to cancel a fine-tuning job by specifying its ID.

6. **List Events from Fine-Tuning Job:**
   - Lists up to 50 events from a specific fine-tuning job.

7. **Delete Fine-Tuned Model:**
   - Allows users to delete a fine-tuned model using its ID (requires ownership privileges).

8. **List Available Models:**
   - Lists available models for reference.

### How to Use:

1. Clone this repository to your local machine.
2. Set up your OpenAI API key by defining it as an environment variable (`OPENAI_API_KEY`).
3. Install the required dependencies specified in the code.
4. Run the script, and it will launch a Streamlit web application.
5. Follow the Streamlit interface to perform various fine-tuning tasks.

Make sure to replace `mojafunkcija`, `positive_login`, and other placeholders with relevant functions or libraries according to your project's structure.

For additional information and updates, please refer to the [OpenAI documentation](https://beta.openai.com/docs/).
## Script Details

- **Author**: Positive
- **Date**: 07.09.2023
- **License**: MIT
---

# Data Preparation for Fine-Tuning - `priprema_podataka_za_ft.py`

This Python script, `priprema_podataka_za_ft.py`, is part of the OpenAI Fine-Tuning Toolkit. It serves a crucial role in the fine-tuning process for OpenAI's GPT-3.5 Turbo model by preparing the training and verification data files.

## Overview

Fine-tuning a language model like GPT-3.5 Turbo requires well-structured and curated datasets. This script streamlines the data preparation process, allowing you to:

- Upload a file containing questions, which may have been generated using the ChatGPT model or obtained from various sources.
- Optionally, upload a source file if the answers need to be based on specific text.
- Define a system message that describes the model's behavior and style, including its name.
- Generate answers to questions using the specified system message and data.
- Save the answers to an output file.
- Create a JSONL file for fine-tuning with user and assistant messages.

## Getting Started

To utilize this script and prepare your data for fine-tuning, follow these steps:

1. Clone this repository to your local machine.

2. Set up your OpenAI API key by defining it as an environment variable (`OPENAI_API_KEY`). Ensure you have an API key from OpenAI's platform.

3. Install the required Python dependencies specified in the script.

4. Run the script, and it will launch a Streamlit web application.

5. Follow the Streamlit interface to perform data preparation for fine-tuning.

## Usage Guidelines

1. Upload a file containing questions that require answers.

2. Optionally, upload a source file if answers should be based on specific text.

3. Define a system message that sets the behavior and style of the model, including its name.

4. Click the "Submit" button to generate answers to questions.

5. Review and, if necessary, make corrections to the generated answers.

6. The script saves the answers to an output file, which you can later use for fine-tuning.

7. A JSONL file is created, incorporating user and assistant messages, facilitating the fine-tuning process.

8. The processed data is saved and ready for use in training a fine-tuned model.

## Script Details

- **Author**: Positive
- **Date**: 07.09.2023
- **License**: MIT

For additional information and updates, please refer to the [OpenAI documentation](https://beta.openai.com/docs/).

---
