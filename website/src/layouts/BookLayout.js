import React from 'react';
import Layout from '@theme/Layout';
import BookNavigation from '../components/BookNavigation';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { useLocation } from '@docusaurus/router';

// BookLayout wraps content with book-specific navigation
const BookLayout = ({ children, ...props }) => {
  const location = useLocation();
  const { siteConfig } = useDocusaurusContext();

  // Extract the doc path from the URL to determine current chapter
  const currentPath = location.pathname.replace('/docs/', '').replace(/\/$/, '') || 'intro';

  return (
    <Layout {...props}>
      <div className="container margin-vert--lg">
        <div className="row">
          <div className="col col--12">
            <BookNavigation currentPath={currentPath} />
            <main>{children}</main>
            <BookNavigation currentPath={currentPath} />
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default BookLayout;