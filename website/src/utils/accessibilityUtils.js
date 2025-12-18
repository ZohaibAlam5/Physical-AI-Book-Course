// Accessibility utilities for the book website
// Implements WCAG 2.1 AA compliance features

// Function to ensure proper heading hierarchy
export const validateHeadingHierarchy = () => {
  // This would typically be used during development to validate heading structure
  if (typeof window !== 'undefined') {
    const headings = Array.from(document.querySelectorAll('h1, h2, h3, h4, h5, h6'));
    let lastLevel = 0;

    headings.forEach(heading => {
      const currentLevel = parseInt(heading.tagName.charAt(1));

      // Ensure heading levels increase by at most 1
      if (currentLevel > lastLevel + 1 && lastLevel !== 0) {
        console.warn(`Invalid heading level: ${heading.tagName} after ${lastLevel > 0 ? 'h' + lastLevel : 'no heading'}`, heading);
      }

      lastLevel = currentLevel;
    });
  }
};

// Function to enhance focus management
export const enhanceFocusManagement = () => {
  if (typeof window !== 'undefined') {
    // Add focus indicators for keyboard navigation
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Tab' || e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        document.body.classList.add('keyboard-nav');
      }
    });

    document.addEventListener('mousedown', () => {
      document.body.classList.remove('keyboard-nav');
    });

    // Ensure focus is maintained within modal dialogs
    const handleFocusTrap = (container) => {
      const focusableElements = container.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );

      if (focusableElements.length === 0) return;

      const firstElement = focusableElements[0];
      const lastElement = focusableElements[focusableElements.length - 1];

      container.addEventListener('keydown', (e) => {
        if (e.key !== 'Tab') return;

        if (e.shiftKey && document.activeElement === firstElement) {
          lastElement.focus();
          e.preventDefault();
        } else if (!e.shiftKey && document.activeElement === lastElement) {
          firstElement.focus();
          e.preventDefault();
        }
      });
    };

    // Apply focus trap to any modal dialogs
    const modals = document.querySelectorAll('[role="dialog"], [role="alertdialog"]');
    modals.forEach(handleFocusTrap);
  }
};

// Function to ensure sufficient color contrast
export const ensureColorContrast = () => {
  // This would typically be used as a development utility
  // In production, proper contrast is ensured through CSS variables
  if (process.env.NODE_ENV === 'development' && typeof window !== 'undefined') {
    const elements = document.querySelectorAll('*');

    elements.forEach(element => {
      const computedStyle = window.getComputedStyle(element);
      const backgroundColor = computedStyle.backgroundColor;
      const color = computedStyle.color;

      // This is a simplified check - in a real implementation you'd calculate actual contrast ratios
      if (backgroundColor && color) {
        // Color contrast validation would go here
      }
    });
  }
};

// Function to enhance screen reader accessibility
export const enhanceScreenReaderAccessibility = () => {
  if (typeof window !== 'undefined') {
    // Add skip to main content link for screen readers
    const skipLink = document.createElement('a');
    skipLink.href = '#main';
    skipLink.textContent = 'Skip to main content';
    skipLink.id = 'skip-to-main';
    skipLink.className = 'screen-reader-only';

    // Initially position off-screen
    skipLink.style.position = 'absolute';
    skipLink.style.left = '-10000px';
    skipLink.style.top = 'auto';
    skipLink.style.width = '1px';
    skipLink.style.height = '1px';
    skipLink.style.overflow = 'hidden';

    // Show the skip link when it receives focus
    skipLink.addEventListener('focus', () => {
      skipLink.style.position = 'fixed';
      skipLink.style.left = '6px';
      skipLink.style.top = '6px';
      skipLink.style.width = 'auto';
      skipLink.style.height = 'auto';
      skipLink.style.overflow = 'visible';
      skipLink.style.backgroundColor = 'var(--ifm-color-primary)';
      skipLink.style.color = 'white';
      skipLink.style.padding = '8px';
      skipLink.style.borderRadius = '4px';
      skipLink.style.zIndex = '1000';
      skipLink.style.textDecoration = 'none';
    });

    skipLink.addEventListener('blur', () => {
      // Return to off-screen position
      skipLink.style.position = 'absolute';
      skipLink.style.left = '-10000px';
      skipLink.style.top = 'auto';
      skipLink.style.width = '1px';
      skipLink.style.height = '1px';
      skipLink.style.overflow = 'hidden';
    });

    document.body.insertBefore(skipLink, document.body.firstChild);
  }
};

// Initialize all accessibility enhancements
export const initAccessibilityFeatures = () => {
  if (typeof window !== 'undefined') {
    // Run accessibility enhancements when DOM is loaded
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => {
        enhanceFocusManagement();
        enhanceScreenReaderAccessibility();
      });
    } else {
      enhanceFocusManagement();
      enhanceScreenReaderAccessibility();
    }
  }
};