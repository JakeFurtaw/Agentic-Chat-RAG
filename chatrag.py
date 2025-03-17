import gradio as gr
from chat_utils import stream_response
from doc_utils import clear_local_docs

css = """
.gradio-container{
background:radial-gradient(#416e8a, #000000);
}
#button{
background:#06354d
}
"""

with gr.Blocks(title="Chat RAG 2.0", fill_width=True, css=css) as demo:
    gr.Markdown("# Chat RAG 2.0: Interactive Coding Assistant")
    with gr.Row():
        with gr.Column(scale=7, variant="compact"):
            chatbot = gr.Chatbot(label="ChatRAG 2.0", height='80vh',
                                 autoscroll=True,
                                 type='messages')
            msg = gr.Textbox(placeholder="Enter your query here and hit enter when you're done...",
                             interactive=True,
                             container=True)
            with gr.Row():
                clear = gr.ClearButton([msg, chatbot],
                                       value="Clear Chat Window",
                                       elem_id="button")
                clear_chat_mem = gr.Button(value="Clear Chat Window and Chat Memory",
                                           elem_id="button")

        with gr.Column(scale=3):
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
                upload_status = gr.Textbox(label="Upload Status", interactive=False)

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
                github_status = gr.Textbox(label="GitHub Status", interactive=False)

    # Set up event handlers
    msg.submit(stream_response, [msg, chatbot], [msg, chatbot])
    # clear_chat_mem.click(clear_all_memory, [], [chatbot, msg])
    #
    # # File upload handlers
    # upload.click(load_local_docs(), [files], [upload_status])
    clear_db.click(clear_local_docs())
    #
    # # GitHub repository handlers
    # getRepo.click(load_github_repository,
    #               [repoOwnerUsername, repoName, repoBranch],
    #               [github_status])
    # removeRepo.click(reset_github_info,
    #                  [],
    #                  [repoOwnerUsername, repoName, repoBranch, github_status])
    demo.launch(inbrowser=True)