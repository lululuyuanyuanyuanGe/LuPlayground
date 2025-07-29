# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Structure

This is a Python playground repository with two main projects:

1. **MCP Chat Application** (`/mcp/`) - A command-line chat interface using the Model Control Protocol (MCP) architecture with Anthropic's Claude API
2. **CrewAI Project** (`/crewAI/`) - A minimal CrewAI project setup (currently empty)

## Development Commands

### MCP Chat Application

**Setup:**
```bash
cd mcp
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

**Run the application:**
```bash
uv run main.py
# or without uv:
python main.py
```

**Environment setup:**
- Create `.env` file with `ANTHROPIC_API_KEY` and `CLAUDE_MODEL`
- Required environment variables are validated at startup

### CrewAI Project

**Setup:**
```bash
cd crewAI
# Dependencies to be added to pyproject.toml
```

## Architecture Overview

### MCP Chat Application

**Core Components:**
- `main.py` - Application entry point with async context management
- `core/cli.py` - Command-line interface with auto-completion and suggestions
- `core/cli_chat.py` - Chat logic and message handling
- `core/claude.py` - Anthropic API integration
- `mcp_client.py` - MCP client implementation
- `mcp_server.py` - MCP server with document management tools

**Key Features:**
- Document retrieval using `@document_id` syntax
- Command execution using `/command` syntax with tab completion
- Multiple MCP client support via command-line arguments
- Windows-specific event loop policy handling

**MCP Server Capabilities:**
- Document storage in-memory dictionary
- Tools: `read_doc_contents`, `edit_document`
- Planned features (TODOs): resources for doc listing, prompts for summarization and markdown conversion

**CLI Features:**
- Auto-suggestion for commands after `/`
- Tab completion for commands and document IDs
- Unified completer supporting both prompts and resources

## Development Notes

- Uses `uv` for package management (recommended) with fallback to pip
- No linting or type checking currently implemented
- Async architecture throughout the MCP application
- Server scripts can be passed as command-line arguments to main.py