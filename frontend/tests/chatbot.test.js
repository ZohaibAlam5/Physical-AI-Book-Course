// Simple test to verify the chatbot component structure
import React from 'react';
import { render, screen } from '@testing-library/react';
import ChatbotWidget from '../src/components/ChatbotWidget';

describe('ChatbotWidget', () => {
  test('renders chatbot widget with default props', () => {
    // Basic render test
    const { container } = render(<ChatbotWidget />);

    // Check if the initial chat toggle button is present
    const toggleButton = container.querySelector('.chat-toggle-button');
    expect(toggleButton).toBeInTheDocument();
  });

  test('accepts custom apiBaseUrl prop', () => {
    const customUrl = 'https://test-api.example.com';
    render(<ChatbotWidget apiBaseUrl={customUrl} />);

    // The component should accept the prop without errors
    const toggleButton = screen.getByRole('button');
    expect(toggleButton).toBeInTheDocument();
  });
});