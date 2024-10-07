# Medical Marijuana Strain Chatbot

## Overview

This project implements an intelligent chatbot specialized in providing information about medical marijuana strains. Leveraging the power of large language models and retrieval-augmented generation (RAG), the chatbot offers detailed, accurate, and helpful answers to user queries about various aspects of medical marijuana strains.

## Features

- **Interactive Q&A**: Users can ask questions about medical marijuana strains and receive informative responses.
- **RAG-based Knowledge**: Utilizes a pre-processed database of strain information for accurate and context-aware answers.
- **Source Attribution**: Provides sources for the information given, enhancing transparency and reliability.
- **Customizable AI Parameters**: Users can adjust the AI's temperature and token limit to fine-tune responses.
- **Conversation History**: Maintains a chat history for context-aware follow-up questions.
- **User-Friendly Interface**: Built with Streamlit for an intuitive and responsive user experience.

## Technology Stack

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **LangChain**: For building LLM-powered applications
- **OpenAI**: Language model provider
- **FAISS**: Vector store for efficient similarity search
- **Pandas**: Data manipulation and analysis

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/medical-marijuana-chatbot.git
   cd medical-marijuana-chatbot
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   Create a `.env` file in the project root and add the following:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   CSV_FILE_PATH=path_to_your_strain_data.csv
   ```

4. Prepare your dataset:
   Ensure your CSV file with strain data is correctly formatted and located at the path specified in your `.env` file.

5. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

1. Start the application using the command above.
2. Open your web browser and navigate to the provided local URL (typically `http://localhost:8501`).
3. Use the text input to ask questions about medical marijuana strains.
4. Adjust the AI parameters in the sidebar if desired.
5. View the chatbot's responses and the sources of information.
6. Use the "Clear Chat History" button to start a new conversation.

## Project Structure

- `app.py`: Main application file containing the Streamlit UI and core logic.
- `utils.py`: Utility functions for data loading and processing.
- `requirements.txt`: List of Python dependencies.
- `.env`: Configuration file for environment variables (not included in the repository).

## Contributing

Contributions to improve the chatbot are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This chatbot is for informational purposes only and should not be considered medical advice. Always consult with a healthcare professional before making any decisions regarding medical marijuana use.

## Acknowledgments

- OpenAI for providing the language model capabilities.
- The LangChain community for their excellent tools and documentation.
- Contributors to the medical marijuana strain dataset used in this project.
