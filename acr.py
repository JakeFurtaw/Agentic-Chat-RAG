import gradio as gr

css = """
.gradio-container{
background:radial-gradient(#416e8a, #000000);
}
#button{
background:#06354d
}
"""

with gr.Blocks(title="Agentic Chat RAG", fill_width=True, css=css) as demo:
    gr.Markdown("# Agentic Chat RAG: Interactive Coding Assistant")
    with gr.Row():
        with gr.Column(scale=7, variant="compact"):
            chatbot = gr.Chatbot(label="Agentic Chat RAG", height='80vh',
                                 autoscroll=True,
                                 type='messages')
            msg = gr.Textbox(placeholder="Enter your query here and hit enter when you're done...",
                             interactive=True,
                             container=True)
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        tavily_use = gr.Checkbox(label="Use Tavily Search", interactive=True)
                with gr.Column(scale=9):
                    with gr.Row():
                        clear = gr.ClearButton([msg, chatbot],
                                               value="Clear Chat Window",
                                               elem_id="button")
                        clear_chat_mem = gr.Button(value="Clear Chat Window and Chat Memory",
                                                   elem_id="button")
        with gr.Column(scale=3):
            gr.Markdown("## <center>Add your data source(s)</center>")
            with gr.Tab("Chat With Files"):
                files = gr.Files(interactive=True,
                                 file_count="multiple",
                                 file_types=["text", ".pdf", ".xlsx", ".py", ".txt", ".dart", ".c", ".jsx", ".xml",
                                             ".css", ".cpp", ".html", ".docx", ".doc", ".js", ".json", ".csv"])
                with gr.Row():
                    upload = gr.Button(value="Upload Data to Knowledge Base",
                                       interactive=True,
                                       size="sm",
                                       elem_id="button")
                    clear_db = gr.Button(value="Clear Knowledge Base",
                                         interactive=True,
                                         size="sm",
                                         elem_id="button")

            with gr.Tab("Chat With a GitHub Repository"):
                repoOwnerUsername = gr.Textbox(label="GitHub Repository Owners Username:",
                                               placeholder="Enter GitHub Repository Owners Username Here....",
                                               interactive=True)
                repoName = gr.Textbox(label="GitHub Repository Name:",
                                      placeholder="Enter Repository Name Here....",
                                      interactive=True)
                repoBranch = gr.Textbox(label="GitHub Repository Branch Name:",
                                        placeholder="Enter Branch Name Here....",
                                        interactive=True)
                with gr.Row():
                    getRepo = gr.Button(value="Load Repository to Model",
                                        size="sm",
                                        interactive=True,
                                        elem_id="button")
                    removeRepo = gr.Button(value="Reset Info and Remove Repository from Model",
                                           size="sm",
                                           interactive=True,
                                           elem_id="button")
    # Set up event handlers
    # msg.submit(chat.stream_response, [msg, chatbot], [msg, chatbot])
    # clear_chat_mem.click(clear_all_memory, [], [chatbot, msg])

    # # File upload handlers
    # upload.click(lambda: chat.reset_chat_engine())
    # clear_db.click(clear_local_docs())

    # # GitHub repository handlers
    # getRepo.click(chat.set_github_info,
    #               [repoOwnerUsername, repoName, repoBranch])
    # removeRepo.click(chat.reset_github_info,
    #                  outputs=[repoOwnerUsername, repoName, repoBranch])
    demo.launch(inbrowser=True) #, share=True