import React from 'react';
import { ThemeProvider } from '@site/src/contexts/ThemeContext';
import injectThemeToggle from '@site/src/utils/injectThemeToggle';
import { initAccessibilityFeatures } from '@site/src/utils/accessibilityUtils';
import apiConfig from '@site/src/utils/apiConfig';

// Dynamically import the chatbot widget to avoid SSR issues
const ChatbotWidget = React.lazy(() => import('@site/src/components/ChatbotWidget'));

// This is the client-side entry point that wraps the entire app
export default function Root({ children }) {
  // Inject the theme toggle into the navbar when the component mounts
  React.useEffect(() => {
    injectThemeToggle();
    initAccessibilityFeatures();
  }, []);

  return (
    <ThemeProvider>
      {children}
      <React.Suspense fallback={process.env.NODE_ENV === 'development' ? <div>Chatbot loading...</div> : null}>
        <ChatbotWidget apiBaseUrl={apiConfig.API_BASE_URL} />
      </React.Suspense>
    </ThemeProvider>
  );
}