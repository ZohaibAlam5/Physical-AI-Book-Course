# Physical AI & Humanoid Robotics Book - Chatbot Frontend

This directory contains the frontend components for the RAG chatbot that answers questions about the Physical AI & Humanoid Robotics book.

## Components

- `ChatbotWidget.js`: Main chatbot component that appears as a floating icon on all website pages
- `ChatbotWidget.css`: Styling for the chatbot widget
- `config.js`: Configuration for API endpoints and other settings

## Features

- Floating chat widget that appears on all website pages
- Collapsible/minimizable interface
- Real-time chat with the RAG-powered AI assistant
- Source citations for responses
- Context-aware responses based on current page
- Responsive design for mobile devices

## Integration

The chatbot is integrated into the Docusaurus website through the Root component (`website/src/components/Root.js`). It uses React's lazy loading to avoid SSR issues.

## Configuration

The chatbot can be configured using the `config.js` file:

- `API_BASE_URL`: Backend API endpoint (default: http://localhost:8000)
- `REQUEST_TIMEOUT`: API request timeout in milliseconds
- `MAX_HISTORY_MESSAGES`: Maximum number of messages to keep in chat history
- Feature flags for various capabilities

## Development

The frontend components are built as part of the Docusaurus website build process. No separate build step is required.

## API Communication

The chatbot communicates with the backend API at `/chat` endpoint, sending:
- Question text
- Query type (global or page-specific)
- Context information (current page URL, selected text, etc.)

The backend responds with:
- Answer text
- Source citations
- Confidence scores