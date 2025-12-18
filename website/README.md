# Technical Book Website

This is the documentation website for the Technical Book, built with [Docusaurus 3](https://docusaurus.io/).

## Local Development

```bash
npm start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

## Build

```bash
npm run build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

## Deployment

### Automatic Deployment (Recommended)

The website is automatically deployed to GitHub Pages when changes are pushed to the `main` branch. The deployment is handled by the GitHub Actions workflow defined in `.github/workflows/deploy.yml`.

### Manual Deployment

To deploy manually:

```bash
npm run gh-deploy
```

This command builds the website and deploys it to the `gh-pages` branch, which GitHub Pages uses to serve the site.

## GitHub Pages Configuration

To enable GitHub Pages for this repository:

1. Go to the repository settings on GitHub
2. Navigate to the "Pages" section
3. Select "Deploy from a branch" as the source
4. Choose the `gh-pages` branch
5. Select `/ (root)` as the folder
6. Click "Save"

The website will be available at `https://your-username.github.io/book-website/`

## Features

- Responsive design optimized for all device sizes
- Light/dark mode toggle
- Full-text search functionality
- WCAG 2.1 AA accessibility compliance
- Docusaurus documentation features
- Module-based content organization
- Syntax highlighting for code examples

## Project Structure

- `/docs` - Contains all the book content organized by modules
- `/src` - Custom React components and CSS
- `/static` - Static assets like images and favicon
- `docusaurus.config.js` - Main configuration file
- `sidebars.js` - Navigation sidebar configuration