# Multimodal Fact-Checker

This project is a multimodal fact-checker that can determine whether a claim is "True" or "Fake" based on an image and a caption.

## Project Structure

- `app.py`: A Streamlit web application for interacting with the fact-checker.
- `backend.py`: A FastAPI backend that serves the fact-checking model.
- `inference.py`:  Handles model loading and prediction logic.

## Getting Started

### Prerequisites

- Python 3.10.11
- uv (https://github.com/astral-sh/uv)

### 1. Clone the Repository

```bash
git clone https://github.com/anshulsc/MFD.git
cd MFD
```

### 2. Set Up the Environment

```bash
uv venv -n .venv --python 3.10.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 3. Download the Model Checkpoint

You will need to download the pre-trained model checkpoint (`best_model.pt`) and place it in the `checkpoints_pt` directory.

### 4. Running the Application

The application consists of a backend server and a frontend web interface. You'll need to run them in separate terminals.

**Terminal 1: Start the Backend**

```bash
python backend.py
```

The backend will be running at `http://127.0.0.1:8000`.

**Terminal 2: Start the Frontend**

```bash
streamlit run app.py
```

The Streamlit application will open in your browser, usually at `http://localhost:8501`.

## How to Use the App

1.  **Upload an Image:** Click the "Choose image files" button to upload an image.
2.  **Upload a Caption:** Click the "Choose caption (.txt) files" button to upload a corresponding text file with the caption.
3.  **Analyze:** Click the "Analyze Samples" button to see the prediction.

## Running Inference from the Command Line

You can also run inference directly from the command line using `inference.py`.

```bash
python inference.py --test-data-path "Test Data"
```

This will run the model on the sample data in the `Test Data` directory and print the results.
