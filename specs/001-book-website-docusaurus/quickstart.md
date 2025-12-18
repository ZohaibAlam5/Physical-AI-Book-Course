# Quickstart: Physical AI & Humanoid Robotics Book Website

## Overview
This guide will help you set up, run, and contribute to the Physical AI & Humanoid Robotics book website. The website is built with Docusaurus and contains comprehensive content organized into 4 modules with 10-12 chapters each.

## Prerequisites
- Node.js LTS (version 18 or higher)
- npm or yarn package manager
- Git for version control
- Basic knowledge of Markdown/MDX for content creation

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Navigate to the website directory**
   ```bash
   cd website
   ```

3. **Install dependencies**
   ```bash
   npm install
   ```

## Running the Development Server

1. **Start the development server**
   ```bash
   npm start
   ```
   This command starts a local development server and opens the website in your default browser at `http://localhost:3000`.

2. **View the site**
   The website will automatically reload when you make changes to content or configuration files.

## Project Structure

```
website/
├── docs/                    # All book content (modules and chapters)
│   ├── intro.md            # Introduction page
│   ├── module-1/           # Module 1: Physical AI Foundations
│   │   ├── chapter-1.md    # Individual chapter files
│   │   ├── chapter-2.md
│   │   └── ...
│   ├── module-2/           # Module 2: Digital Twins & Simulation
│   │   ├── chapter-1.md
│   │   └── ...
│   ├── module-3/           # Module 3: Perception & Navigation
│   │   └── ...
│   └── module-4/           # Module 4: Vision-Language-Action
│       └── ...
├── src/                    # Custom React components and pages
├── static/                 # Static assets (images, etc.)
├── docusaurus.config.js    # Main Docusaurus configuration
├── sidebars.js             # Navigation sidebar configuration
└── package.json            # Project dependencies and scripts
```

## Adding New Content

### Adding a New Chapter
1. Create a new Markdown file in the appropriate module directory:
   ```bash
   # For example, to add chapter 3 to module 1:
   # Create website/docs/module-1/chapter-3.md
   ```

2. Add the required frontmatter to your chapter:
   ```markdown
   ---
   title: Chapter Title
   description: Brief description of the chapter content
   sidebar_position: 3
   ---

   # Chapter Title

   Your chapter content here...
   ```

3. Update the sidebar configuration in `sidebars.js` to include your new chapter.

### Adding a New Module
1. Create a new directory under `docs/`:
   ```bash
   mkdir website/docs/module-5
   ```

2. Add chapter files to the new module directory.

3. Update `sidebars.js` to include the new module in the navigation.

## Configuration Files

### docusaurus.config.js
Main configuration file containing:
- Site metadata (title, tagline, favicon)
- Theme settings
- Plugin configurations
- Navigation items

### sidebars.js
Defines the sidebar navigation structure, organizing modules and chapters hierarchically.

## Building for Production

To build the website for deployment:

```bash
npm run build
```

This creates a `build/` directory with a production-ready version of the site.

## Deployment

The website is configured for GitHub Pages deployment. To deploy:

1. Run the build command: `npm run build`
2. The site will be automatically deployed to GitHub Pages if configured properly

## Content Guidelines

### Writing Style
- Use clear, instructional language appropriate for senior undergraduate and graduate-level students
- Include practical examples and implementation logic
- Explain complex robotics concepts using intuitive analogies
- Assume readers know basic Python and AI fundamentals

### Technical Requirements
- Content must be written in Markdown/MDX format
- Each chapter should include learning objectives
- Include diagrams and code snippets where appropriate
- Maintain WCAG 2.1 AA accessibility compliance

## Troubleshooting

### Common Issues
- **Page not loading**: Ensure all dependencies are installed with `npm install`
- **Broken links**: Run `npm run build` to check for broken links
- **Styling issues**: Check that custom CSS is properly loaded in `src/css/custom.css`

### Development Commands
- `npm start` - Start development server
- `npm run build` - Build for production
- `npm run serve` - Serve production build locally
- `npm run deploy` - Deploy to GitHub Pages