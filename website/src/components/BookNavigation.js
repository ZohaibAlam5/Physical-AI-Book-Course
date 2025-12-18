import React from 'react';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './BookNavigation.module.css';

// Simple book navigation component that provides chapter navigation
const BookNavigation = ({ currentPath }) => {
  // Define the book structure for navigation
  const bookStructure = {
    'intro': {
      title: 'Introduction',
      prev: null,
      next: 'module-1/chapter-1'
    },
    'module-1/chapter-1': {
      title: 'Chapter 1 - Introduction to Core Concepts',
      prev: 'intro',
      next: 'module-1/chapter-2'
    },
    'module-1/chapter-2': {
      title: 'Chapter 2 - Advanced Core Concepts',
      prev: 'module-1/chapter-1',
      next: 'module-2/chapter-1'
    },
    'module-2/chapter-1': {
      title: 'Chapter 1 - Introduction to Module 2 Concepts',
      prev: 'module-1/chapter-2',
      next: 'module-3/chapter-1'
    },
    'module-3/chapter-1': {
      title: 'Chapter 1 - Introduction to Advanced Concepts',
      prev: 'module-2/chapter-1',
      next: 'module-4/chapter-1'
    },
    'module-4/chapter-1': {
      title: 'Chapter 1 - Introduction to Practical Applications',
      prev: 'module-3/chapter-1',
      next: null
    }
  };

  const currentPage = bookStructure[currentPath] || {};

  return (
    <div className={styles.bookNavigation}>
      <div className={styles.navigationContainer}>
        {currentPage.prev && (
          <div className={styles.navLink}>
            <Link to={useBaseUrl(`docs/${currentPage.prev}`)} className={styles.prevLink}>
              ← Previous: {bookStructure[currentPage.prev]?.title || 'Previous Chapter'}
            </Link>
          </div>
        )}

        <div className={styles.navCenter}>
          <Link to={useBaseUrl('docs/')} className={styles.homeLink}>
            Book Home
          </Link>
        </div>

        {currentPage.next && (
          <div className={styles.navLink}>
            <Link to={useBaseUrl(`docs/${currentPage.next}`)} className={styles.nextLink}>
              Next: {bookStructure[currentPage.next]?.title || 'Next Chapter'} →
            </Link>
          </div>
        )}
      </div>
    </div>
  );
};

export default BookNavigation;