// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

const {themes: prismThemes} = require('prism-react-renderer');


/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'A comprehensive guide to embodied intelligence and humanoid robotics',
  favicon: 'img/favicon.png',

  // Set the production url of your site here
  url: 'https://your-username.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub Pages deployment, it's often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub Pages, you don't need these.
  organizationName: 'your-username', // Usually your GitHub org/user name.
  projectName: 'physical-ai-book', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/your-username/physical-ai-book/edit/main/website/',
        },
        blog: false, // Disable blog functionality
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: 'Physical AI & Humanoid Robotics',
        logo: {
          alt: 'Physical AI & Humanoid Robotics Book Logo',
          src: 'img/book-logo.svg', // This will be created later
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Book',
          },
          {
            to: '/toc',
            label: 'Table of Contents',
            position: 'left',
          },
          {
            to: '/about',
            label: 'About',
            position: 'left',
          },
          {
            to: '/book-search',
            label: 'Search',
            position: 'left',
          },
          {
            type: 'html',
            position: 'right',
            value: '<div id="theme-toggle-container"></div>',
          },
          {
            href: 'https://github.com/your-username/physical-ai-book',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Book Sections',
            items: [
              {
                label: 'Module 1: Physical AI Foundations',
                to: '/docs/module-1/chapter-1',
              },
              {
                label: 'Module 2: Digital Twins & Simulation',
                to: '/docs/module-2/chapter-1',
              },
              {
                label: 'Module 3: Perception & Navigation',
                to: '/docs/module-3/chapter-1',
              },
              {
                label: 'Module 4: Vision-Language-Action',
                to: '/docs/module-4/chapter-1',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Robotics Stack Exchange',
                href: 'https://robotics.stackexchange.com/',
              },
              {
                label: 'ROS Discourse',
                href: 'https://discourse.ros.org/',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/your-username/physical-ai-book',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Book. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
      // Comprehensive metadata for SEO
      metadata: [
        {
          name: 'keywords',
          content: 'physical ai, humanoid robotics, embodied intelligence, ros 2, robot simulation, perception, navigation, vision-language-action, ai robotics, machine learning, computer vision, robotics engineering'
        },
        {
          name: 'author',
          content: 'Physical AI & Humanoid Robotics Authors'
        },
        {
          name: 'og:locale',
          content: 'en_US'
        },
        {
          name: 'og:type',
          content: 'website'
        },
        {
          name: 'og:site_name',
          content: 'Physical AI & Humanoid Robotics Book'
        },
        {
          name: 'twitter:card',
          content: 'summary_large_image'
        },
        {
          name: 'twitter:site',
          content: '@physicalairobotics'
        },
        {
          name: 'description',
          content: 'A comprehensive guide to Physical AI and Humanoid Robotics covering embodied intelligence, ROS 2, simulation, perception, navigation, and vision-language-action systems.'
        },
        {
          name: 'theme-color',
          content: '#2e8555'
        },
        {
          name: 'msapplication-TileColor',
          content: '#2e8555'
        },
        {
          name: 'application-name',
          content: 'Physical AI & Humanoid Robotics'
        },
        {
          name: 'apple-mobile-web-app-title',
          content: 'Physical AI & Humanoid Robotics'
        },
        {
          name: 'robots',
          content: 'index, follow'
        },
        {
          name: 'googlebot',
          content: 'index, follow, max-video-preview:-1, max-image-preview:large, max-snippet:-1'
        },
      ],
    }),

  themes: [
    [
      require.resolve("@easyops-cn/docusaurus-search-local"),
      {
        // ... your options
        hashed: true,
        // For Docs using Chinese, The `language` is recommended to be 'zh'
        language: ["en"],
        // Optional: also index docs' descriptions
        indexDocs: true,
        indexBlog: false, // We don't have a blog
        indexPages: false,
        searchResultLimits: 8,
        // ... other options
      },
    ],
  ],

  // Use the Root component to wrap the app with the theme provider
  clientModules: [
    require.resolve('./src/components/Root.js'),
  ],
};

module.exports = config;
