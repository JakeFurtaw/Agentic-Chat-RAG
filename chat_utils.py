import json
import os
from agent_utils import AgentTools

class ChatEngine:
    def __init__(self):
        self.owner = None
        self.branch = None
        self.repo = None
        self.agent_tools = None
        self.history_file = "chat_history.json"
        self.create_agent()

    def create_agent(self):
        """Create or retrieve agent tools with document loading."""
        if self.agent_tools is None:
            self.agent_tools = AgentTools(owner=self.owner, repo=self.repo, branch=self.branch)
        return self.agent_tools

    def process_input(self, message):
        """Process user input using the agent."""
        if self.agent_tools is None:
            self.create_agent()
        return self.agent_tools.run_agent(message)

    def stream_response(self, message, history):
        """Stream response and update chat history."""
        try:
            response = self.process_input(message)
            full_response = ""
            for token in response.response_gen:
                full_response += token
                chat_history = history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": full_response}
                ]
                yield "", chat_history
            self.save_history(chat_history)
        except Exception as e:
            error_msg = f"Error streaming response: {str(e)}"
            chat_history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": error_msg}
            ]
            yield "", chat_history
            self.save_history(chat_history)

    def save_history(self, history):
        """Save chat history to a file."""
        try:
            with open(self.history_file, "w") as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            print(f"Failed to save history: {str(e)}")

    def set_github_info(self, owner, repo, branch):
        """Set GitHub repository information and update agent."""
        self.owner, self.repo, self.branch = owner, repo, branch
        self.agent_tools = None  # Reset agent to reload documents
        self.create_agent()

    def reset_github_info(self):
        """Reset GitHub repository information and update agent."""
        self.owner = self.repo = self.branch = ""
        self.set_github_info(self.owner, self.repo, self.branch)
        return self.owner, self.repo, self.branch