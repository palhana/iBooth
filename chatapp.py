import gradio as gr
import os
import openai

openai.api_key = os.environ.get('OPENAI_API_KEY')

message_history = [{"role": "user", "content": f"You are a joke bot, i will specify ... "},
                   {"role": "assistant", "content": f"yon can make this whatever you want"},
                   {"role": "assistant", "content": f"OK"}]

def predict(input):
    global message_history
    message_history.append({"role": "user", "content": input})
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message_history
    )

    reply_content = completion.choices[0].message.content
    print(reply_content)
    message_history.append({"role": "assistant", "content": reply_content})
    response = [(message_history[i]["content"], message_history[i+1]["content"]) for i in range(2, len(message_history)-1, 2)]
    return response

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    with gr.Row():
        txt = gr.Textbox(show_lable=False, placeholder="Type your message here").style(container=False)
        txt.submit(predict, txt, chatbot)
        #txt.submit(lambda: "", None, txt)
        txt.submit(None, None, txt, _js="() => {''}")

demo.launch(share=True)
