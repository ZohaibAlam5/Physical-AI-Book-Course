import React from 'react';
import { ThemeProvider } from '../contexts/ThemeContext';
import injectThemeToggle from '../utils/injectThemeToggle';
import { initAccessibilityFeatures } from '../utils/accessibilityUtils';

// This is the client-side entry point that wraps the entire app
export default function Root({ children }) {
  // Inject the theme toggle into the navbar when the component mounts
  React.useEffect(() => {
    injectThemeToggle();
    initAccessibilityFeatures();
  }, []);

  return <ThemeProvider>{children}</ThemeProvider>;
}