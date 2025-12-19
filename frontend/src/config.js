// Configuration for the chatbot API
const config = {
  // API endpoint for the chatbot backend
  // In production, this would be your deployed backend URL
  // For development, it defaults to localhost:8000
  API_BASE_URL: process.env.REACT_APP_API_URL || 'http://localhost:8000',

  // Timeout for API requests (in milliseconds)
  REQUEST_TIMEOUT: 30000,

  // Maximum number of messages to keep in chat history
  MAX_HISTORY_MESSAGES: 50,

  // Feature flags
  ENABLE_SOURCE_CITATIONS: true,
  ENABLE_RICH_TEXT_RESPONSES: true,
  ENABLE_CONTEXT_AWARENESS: true
};

export default config;