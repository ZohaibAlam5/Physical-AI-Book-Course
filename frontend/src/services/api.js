// frontend/src/services/api.js
// API client for backend communication

class ApiClient {
  constructor(baseURL, apiKey) {
    this.baseURL = baseURL;
    this.apiKey = apiKey;
    this.defaultHeaders = {
      'Content-Type': 'application/json',
      'X-API-Key': this.apiKey,
    };
  }

  async chat(queryData) {
    try {
      const response = await fetch(`${this.baseURL}/v1/chat`, {
        method: 'POST',
        headers: this.defaultHeaders,
        body: JSON.stringify(queryData),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('API call failed:', error);
      throw error;
    }
  }

  async healthCheck() {
    try {
      const response = await fetch(`${this.baseURL}/v1/health`, {
        method: 'GET',
        headers: this.defaultHeaders,
      });

      return response.ok;
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  }
}

export default ApiClient;