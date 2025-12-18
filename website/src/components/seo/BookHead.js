import React from 'react';
import Head from '@docusaurus/Head';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

// Component to add book-specific SEO enhancements to the head
const BookHead = ({ title, description, image, type = 'website', url }) => {
  const { siteConfig } = useDocusaurusContext();

  const siteTitle = siteConfig.title || 'Technical Book Website';
  const siteDescription = siteConfig.tagline || 'A comprehensive guide to technical concepts';
  const siteUrl = siteConfig.url || 'https://your-book.github.io';
  const siteImage = `${siteUrl}/img/docusaurus-social-card.jpg`;

  // Use provided values or fall back to defaults
  const pageTitle = title ? `${title} | ${siteTitle}` : siteTitle;
  const pageDescription = description || siteDescription;
  const pageImage = image || siteImage;
  const pageUrl = url || siteUrl;

  return (
    <Head>
      {/* Standard SEO */}
      <title>{pageTitle}</title>
      <meta name="description" content={pageDescription} />

      {/* Open Graph / Facebook */}
      <meta property="og:type" content={type} />
      <meta property="og:url" content={pageUrl} />
      <meta property="og:title" content={pageTitle} />
      <meta property="og:description" content={pageDescription} />
      <meta property="og:image" content={pageImage} />

      {/* Twitter */}
      <meta property="twitter:card" content="summary_large_image" />
      <meta property="twitter:url" content={pageUrl} />
      <meta property="twitter:title" content={pageTitle} />
      <meta property="twitter:description" content={pageDescription} />
      <meta property="twitter:image" content={pageImage} />

      {/* Canonical URL */}
      <link rel="canonical" href={pageUrl} />

      {/* Structured Data for Book */}
      <script type="application/ld+json">
        {JSON.stringify({
          "@context": "https://schema.org",
          "@type": "Book",
          "name": siteTitle,
          "author": {
            "@type": "Person",
            "name": "Technical Book Authors"
          },
          "publisher": {
            "@type": "Organization",
            "name": "Technical Book Publishers"
          },
          "description": siteDescription,
          "url": siteUrl,
          "image": siteImage,
          "datePublished": "2025-01-01",
          "genre": "Technical Education",
          "inLanguage": "English",
          "bookFormat": "EBook",
          "educationalUse": "Learning",
          "learningResourceType": "Textbook",
          "audience": {
            "@type": "Audience",
            "audienceType": ["Students", "Developers", "Educators"]
          },
          "hasPart": [
            {
              "@type": "Chapter",
              "name": "Module 1: Foundational Concepts",
              "url": `${siteUrl}/docs/module-1/chapter-1`
            },
            {
              "@type": "Chapter",
              "name": "Module 2: Intermediate Topics",
              "url": `${siteUrl}/docs/module-2/chapter-1`
            },
            {
              "@type": "Chapter",
              "name": "Module 3: Advanced Concepts",
              "url": `${siteUrl}/docs/module-3/chapter-1`
            },
            {
              "@type": "Chapter",
              "name": "Module 4: Practical Applications",
              "url": `${siteUrl}/docs/module-4/chapter-1`
            }
          ]
        })}
      </script>
    </Head>
  );
};

export default BookHead;