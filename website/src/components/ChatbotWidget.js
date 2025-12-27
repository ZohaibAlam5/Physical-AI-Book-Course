import React, { useState, useEffect, useRef } from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext'; 
import './ChatbotWidget.css';

const ChatbotWidget = () => {
  // --- FINAL FIX: HARDCODED URL ---
  // We are forcing the widget to use your live Railway server.
  // This removes any chance of it falling back to localhost.
  const finalBaseUrl = 'https://physical-ai-book-course-production.up.railway.app'; 

  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);
    setIsTyping(true);

    try {
      // Remove any potential trailing slash
      const cleanUrl = finalBaseUrl.replace(/\/$/, '');
      
      console.log("Sending request to:", `${cleanUrl}/chat`); // Debug log

      const response = await fetch(`${cleanUrl}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: inputValue,
          query_type: 'global',
          selected_text: null,
          page_context: {
            module: '', 
            chapter: '',
            url: typeof window !== 'undefined' ? window.location.pathname : ''
          }
        })
      });

      if (response.ok) {
        const data = await response.json();
        const botMessage = {
          id: Date.now() + 1,
          text: data.answer,
          sender: 'bot',
          sources: data.sources || [],
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        };
        setMessages(prev => [...prev, botMessage]);
      } else {
        const errorData = await response.json();
        const errorMessage = {
          id: Date.now() + 1,
          text: `Error: ${errorData.detail || 'Failed to get response'}`,
          sender: 'bot',
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    } catch (error) {
      console.error("Fetch error:", error);
      const errorMessage = {
        id: Date.now() + 1,
        text: `Network Error: Could not connect to server.`,
        sender: 'bot',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      setIsTyping(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  return (
    <div className={`chatbot-widget ${isOpen ? 'chat-open' : 'chat-closed'}`}>
      {isOpen ? (
        <div className="chat-window">
          <div className="chat-header">
            <div className="chat-title">Physical AI & Humanoid Robotics Assistant</div>
            <div className="chat-controls">
              <button onClick={clearChat} className="clear-btn" title="Clear chat">
                ✕
              </button>
              <button onClick={toggleChat} className="minimize-btn" title="Minimize">
                −
              </button>
            </div>
          </div>

          <div className="chat-messages">
            {messages.length === 0 ? (
              <div className="welcome-message">
                <h3>Hello! I'm your Physical AI & Humanoid Robotics Assistant</h3>
                <p>Ask me anything about the book content:</p>
                <ul>
                  <li>• Concepts from specific modules or chapters</li>
                  <li>• Technical details and explanations</li>
                  <li>• Cross-references between topics</li>
                </ul>
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={`message ${message.sender}-message`}
                >
                  <div className="message-content">
                    <span className="message-text">{message.text}</span>
                    <span className="message-time">{message.timestamp}</span>
                  </div>
                  {message.sources && message.sources.length > 0 && (
                    <div className="message-sources">
                      <details>
                        <summary>Sources</summary>
                        <ul>
                          {message.sources.map((source, idx) => (
                            <li key={idx}>
                              <a href={source.url} target="_blank" rel="noopener noreferrer">
                                {source.title || source.url}
                              </a>
                            </li>
                          ))}
                        </ul>
                      </details>
                    </div>
                  )}
                </div>
              ))
            )}
            {isTyping && (
              <div className="message bot-message">
                <div className="message-content">
                  <span className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="chat-input-area">
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about the book content..."
              rows="1"
              disabled={isLoading}
              className="chat-input"
            />
            <button
              onClick={sendMessage}
              disabled={isLoading || !inputValue.trim()}
              className="send-button"
            >
              {isLoading ? 'Sending...' : '→'}
            </button>
          </div>
        </div>
      ) : (
        <button className="chat-toggle-button" onClick={toggleChat}>
          <div className="chat-icon">🤖</div>
          <div className="chat-label">Ask AI</div>
        </button>
      )}
    </div>
  );
};

export default ChatbotWidget;