# Medical Marijuana Strain Chatbot

This application is a chatbot that provides detailed information about various medical marijuana strains using Retrieval Augmented Generation (RAG) architecture.

## Features

- **Interactive Chatbot:** Ask questions about different marijuana strains and receive detailed answers.
- **Source Transparency:** View the sources of the information provided by the chatbot.
- **Customizable Settings:** Adjust the creativity of responses and the length of answers.
- **Persistent Vector Store:** Efficiently retrieves relevant information without recomputing embeddings.

## Setup Instructions

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/yourusername/ai-cannabis-chatbot.git
    cd ai-cannabis-chatbot
    ```

2. **Create and Activate a Virtual Environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Configure Environment Variables:**

    - Create a `.env` file in the project directory.
    - Add your OpenAI API key and CSV file path.

    ```
    OPENAI_API_KEY=sk-your_openai_api_key_here
    CSV_FILE_PATH=/path/to/your/cannabis.csv
    ```

5. **Run the Application:**

    ```bash
    streamlit run app.py
    ```

6. **Interact with the Chatbot:**

    - Open the URL provided by Streamlit (usually `http://localhost:8501`) in your web browser.
    - Ask questions about medical marijuana strains and explore the features.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License.

