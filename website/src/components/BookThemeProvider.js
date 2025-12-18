import React from 'react';
import { ThemeProvider } from '../contexts/ThemeContext';

// This component wraps the entire app with the theme context
const BookThemeProvider = ({ children }) => {
  return <ThemeProvider>{children}</ThemeProvider>;
};

export default BookThemeProvider;