# Chatbot-
Here’s an all-in-one **note** that summarizes everything you need to set up and run your chatbot:

---

##  ChatBot Setup Guide (DialoGPT + Gradio)

###  Step 1: Install Required Packages

Use the following command to install all dependencies:

```bash
pip install torch transformers gradio python-dotenv
```

>  If you're using a GPU, install the CUDA-compatible version of PyTorch from [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally)

---

###  What Each Package Does

| Package         | Description                                                       |
| --------------- | ----------------------------------------------------------------- |
| `torch`         | Core library for running machine learning models                  |
| `transformers`  | Hugging Face library for loading pre-trained models like DialoGPT |
| `gradio`        | Makes a simple web interface for chatting with the bot            |
| `python-dotenv` | Loads environment variables from a `.env` file (if needed)        |

---

###  Optional: `requirements.txt` File

Create a file named `requirements.txt` and paste this:

```
torch
transformers
gradio
python-dotenv
```

Then install with:

```bash
pip install -r requirements.txt
```

---

###  Run Your ChatBot

Save your chatbot code in a Python file (e.g. `chatbot.py`) and run it:

```bash
python chatbot.py
```

This will launch a local Gradio interface in your browser.

---
