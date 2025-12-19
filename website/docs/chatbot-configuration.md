# Configuring the Chatbot API URL

The RAG chatbot frontend can be configured to connect to different backend environments (development, staging, production).

## Environment Configuration

The API URL can be configured in multiple ways:

### 1. Build-time Configuration (Recommended for Production)

Set the environment variable when building the Docusaurus site:

```bash
CHATBOT_API_URL=https://your-backend-domain.com npm run build
```

### 2. Development Configuration

For local development, you can set the environment variable:

```bash
# Linux/Mac
export CHATBOT_API_URL=http://localhost:8000
npm run start

# Windows
set CHATBOT_API_URL=http://localhost:8000
npm run start
```

### 3. Default Value

If no environment variable is set, the chatbot will default to `http://localhost:8000`.

## Configuration Priority

The configuration follows this priority order:
1. Custom field from Docusaurus config (`CHATBOT_API_URL`)
2. Environment variable during build time
3. Default value (`http://localhost:8000`)

## Deployment

When deploying to GitHub Pages or other platforms, ensure that the correct API URL is set during the build process to connect to your deployed backend service.