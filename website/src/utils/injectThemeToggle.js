import React from 'react';
import { createRoot } from 'react-dom/client';
import ThemeToggle from '../components/ThemeToggle';

// Function to inject the theme toggle into the navbar
export const injectThemeToggle = () => {
  // Wait for the DOM to be ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', injectThemeToggleIntoContainer);
  } else {
    injectThemeToggleIntoContainer();
  }
};

// Function to inject the theme toggle into the designated container
const injectThemeToggleIntoContainer = () => {
  const container = document.getElementById('theme-toggle-container');
  if (container) {
    // Clear the container
    container.innerHTML = '';

    // Create a React root and render the theme toggle
    const root = createRoot(container);
    root.render(<ThemeToggle />);
  }
};

// Export as default to be used in Root component
export default injectThemeToggle;