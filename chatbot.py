import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
from dotenv import load_dotenv
import os

class ChatBot:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        load_dotenv()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.chat_history_ids = None
        self.step = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def respond(self, user_input):
        new_user_input_ids = self.tokenizer.encode(
            user_input + self.tokenizer.eos_token,
            return_tensors='pt'
        ).to(self.device)

        bot_input_ids = torch.cat(
            [self.chat_history_ids, new_user_input_ids],
            dim=-1
        ) if self.step > 0 else new_user_input_ids

        self.chat_history_ids = self.model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=100,
            top_p=0.7,
            temperature=0.8
        )

        response = self.tokenizer.decode(
            self.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )

        self.step += 1
        return response

# Create a chatbot instance
chatbot = ChatBot()

# Define Gradio interface
def chat_interface(user_input):
    if user_input.lower() in ['reset', 'clear']:
        chatbot.chat_history_ids = None
        chatbot.step = 0
        return "Chat history cleared. Let's start again!"
    return chatbot.respond(user_input)

# Launch Gradio UI
iface = gr.Interface(
    fn=chat_interface,
    inputs=gr.Textbox(lines=2, placeholder="Type your message here..."),
    outputs="text",
    title="ChatBot",
    description="A conversational AI chatbot.",
    flagging_mode="never"

)

if __name__ == "__main__":
    iface.launch()
