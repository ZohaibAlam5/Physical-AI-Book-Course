// API Configuration for Docusaurus
// This configuration allows setting the backend API URL for different environments

// Access the custom field from docusaurus config
const getApiBaseUrl = () => {
  // Check if we're in the browser environment
  if (typeof window !== 'undefined') {
    // Access the custom field that was passed from docusaurus config
    // Docusaurus makes custom fields available via window.__DOCUSAURUS__
    if (window.__DOCUSAURUS__) {
      const docusaurusConfig = window.__DOCUSAURUS__;
      if (docusaurusConfig.customFields && docusaurusConfig.customFields.CHATBOT_API_URL) {
        return docusaurusConfig.customFields.CHATBOT_API_URL;
      }
    }

    // Fallback: check for environment variable in browser (set via Docusaurus config)
    if (window.env && window.env.CHATBOT_API_URL) {
      return window.env.CHATBOT_API_URL;
    }
  }

  // Default to localhost for development
  // In production, this should be updated to the actual backend URL
  return 'http://localhost:8000';
};

const apiConfig = {
  API_BASE_URL: getApiBaseUrl(),
  REQUEST_TIMEOUT: 30000,
  ENABLE_SOURCE_CITATIONS: true,
  ENABLE_RICH_TEXT_RESPONSES: true,
  ENABLE_CONTEXT_AWARENESS: true
};

export default apiConfig;