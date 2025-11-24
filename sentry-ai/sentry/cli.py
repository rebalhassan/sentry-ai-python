# sentry/cli.py
"""
CLI Interface for Sentry-AI
Rich terminal UI using Textual
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Input, Static, Button, DataTable
from textual.binding import Binding

from .services.rag import get_rag_service
from .core.database import db
from .core.models import ChatMessage


class ChatView(Static):
    """Display chat messages"""
    
    def __init__(self):
        super().__init__()
        self.messages = []
    
    def add_message(self, role: str, content: str):
        self.messages.append((role, content))
        self.update_display()
    
    def update_display(self):
        lines = []
        for role, content in self.messages:
            if role == "user":
                lines.append(f"[bold cyan]You:[/] {content}")
            else:
                lines.append(f"[bold green]Sentry:[/] {content}")
            lines.append("")
        
        self.update("\n".join(lines))


class SentryApp(App):
    """Sentry-AI TUI Application"""
    
    CSS = """
    #chat-view {
        height: 80%;
        border: solid green;
        padding: 1;
    }
    
    #input-area {
        height: 10%;
        dock: bottom;
    }
    
    #stats {
        height: 10%;
        border: solid blue;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("s", "show_sources", "Sources"),
        Binding("i", "show_stats", "Stats"),
    ]
    
    def __init__(self):
        super().__init__()
        self.rag = None  # Will be initialized lazily
        self.chat_view = None
        self.is_initializing = False
        self.is_ready = False
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        # Chat area
        self.chat_view = ChatView()
        yield self.chat_view
        
        # Input area
        with Vertical(id="input-area"):
            yield Input(placeholder="Ask about your logs...", id="query-input")
            with Horizontal():
                yield Button("Send", variant="primary", id="send-btn")
                yield Button("Clear", id="clear-btn")
        
        # Stats area
        yield Static(id="stats")
        
        yield Footer()
    
    def on_mount(self):
        """Called when app starts"""
        # Show loading message
        self.chat_view.add_message(
            "assistant",
            "🔄 Initializing Sentry-AI... Loading models, please wait..."
        )
        
        # Initialize RAG service in background
        self.run_worker(self._initialize_rag(), exclusive=True)
    
    def on_button_pressed(self, event):
        """Handle button clicks"""
        if event.button.id == "send-btn":
            self.send_query()
        elif event.button.id == "clear-btn":
            self.chat_view.messages = []
            self.chat_view.update_display()
    
    def on_input_submitted(self, event):
        """Handle Enter key in input"""
        if event.input.id == "query-input":
            self.send_query()
    
    async def _initialize_rag(self):
        """Initialize RAG service in background"""
        try:
            self.is_initializing = True
            
            # Load RAG service (this is the heavy operation)
            self.rag = get_rag_service()
            
            # Mark as ready
            self.is_ready = True
            self.is_initializing = False
            
            # Remove loading message
            self.chat_view.messages.pop()
            
            # Show welcome message
            self.chat_view.add_message(
                "assistant",
                "✅ Sentry-AI ready! Ask me about your logs."
            )
            
            # Update stats now that RAG is loaded
            self.update_stats()
            
        except Exception as e:
            self.is_initializing = False
            self.is_ready = False
            
            # Remove loading message
            if self.chat_view.messages:
                self.chat_view.messages.pop()
            
            # Show error
            self.chat_view.add_message(
                "assistant",
                f"❌ Failed to initialize: {str(e)}\\n\\nPlease check your configuration and try again."
            )
    
    def send_query(self):
        """Send query to RAG"""
        input_widget = self.query_one("#query-input", Input)
        query = input_widget.value.strip()
        
        if not query:
            return
        
        # Check if RAG is ready
        if not self.is_ready:
            self.chat_view.add_message(
                "assistant",
                "⚠️ Please wait for initialization to complete..."
            )
            return
        
        # Clear input
        input_widget.value = ""
        
        # Show user message
        self.chat_view.add_message("user", query)
        
        # Show "thinking" message
        self.chat_view.add_message("assistant", "Searching logs...")
        
        # Query RAG (in background to avoid blocking UI)
        self.run_worker(self._query_rag(query), exclusive=True)
    
    async def _query_rag(self, query: str):
        """Query RAG in background"""
        result = self.rag.query(query)
        
        # Remove "thinking" message
        self.chat_view.messages.pop()
        
        # Show answer
        answer = result.answer
        if result.sources:
            answer += f"\n\n[dim]({len(result.sources)} sources, confidence: {result.confidence:.2f})[/]"
        
        self.chat_view.add_message("assistant", answer)
        
        # Save to database
        db.add_chat_message(ChatMessage(role="user", content=query))
        db.add_chat_message(ChatMessage(
            role="assistant",
            content=result.answer,
            sources=[c.id for c in result.sources]
        ))
    
    def update_stats(self):
        """Update stats display"""
        if not self.rag or not self.is_ready:
            stats_text = "[bold]Statistics[/]\\nInitializing..."
        else:
            stats = self.rag.get_stats()
            
            stats_text = f"""
[bold]Statistics[/]
Total Chunks: {stats['total_chunks']}
Sources: {stats['database_stats'].get('total_sources', 0)}
Model: {stats['llm_model']}
"""
        
        self.query_one("#stats", Static).update(stats_text)
    
    def action_show_sources(self):
        """Show sources screen"""
        # TODO: Implement sources management screen
        pass
    
    def action_show_stats(self):
        """Show detailed stats"""
        # TODO: Implement stats screen
        pass


def main():
    """Entry point"""
    app = SentryApp()
    app.run()


if __name__ == "__main__":
    main()
