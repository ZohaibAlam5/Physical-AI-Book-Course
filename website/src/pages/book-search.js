import React, {useState, useEffect} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import useSearchQuery from '@theme/hooks/useSearchQuery';
import BookHead from '@site/src/components/seo/BookHead';

import styles from './search.module.css';

function SearchPage() {
  const {siteConfig} = useDocusaurusContext();
  const {searchQuery, setQuery} = useSearchQuery();
  const [query, setQueryState] = useState('');
  const [selectedModules, setSelectedModules] = useState(['all']);
  const [searchResults, setSearchResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  // Mock data for demonstration - in a real implementation, this would come from the search API
  const mockResults = [
    {
      id: 1,
      title: 'Chapter 1 - Embodied Intelligence Concepts',
      url: '/docs/module-1/chapter-1',
      excerpt: 'Welcome to the first chapter of our Physical AI & Humanoid Robotics book. This chapter introduces the fundamental concepts of embodied intelligence, which forms the foundation for understanding how artificial intelligence can be integrated with physical systems to create truly intelligent robots...',
      module: 'module1',
      chapter: 'Chapter 1'
    },
    {
      id: 2,
      title: 'Chapter 2 - Digital vs Physical AI Differences',
      url: '/docs/module-1/chapter-2',
      excerpt: 'This chapter explores the fundamental differences between digital AI and physical AI systems. While digital AI operates in virtual environments with perfect information, physical AI must deal with real-world uncertainties, sensor noise, and physical constraints...',
      module: 'module1',
      chapter: 'Chapter 2'
    },
    {
      id: 3,
      title: 'Chapter 3 - Embodied Cognition Principles',
      url: '/docs/module-1/chapter-3',
      excerpt: 'Embodied cognition is a key principle in Physical AI that emphasizes the role of the body in shaping cognitive processes. This chapter explores how physical interaction with the environment influences learning and decision-making in robotic systems...',
      module: 'module1',
      chapter: 'Chapter 3'
    },
    {
      id: 4,
      title: 'Chapter 1 - Role of Simulation in Physical AI',
      url: '/docs/module-2/chapter-1',
      excerpt: 'Simulation plays a pivotal role in Physical AI and humanoid robotics development. Unlike traditional AI that operates in digital-only environments, Physical AI systems must interact with the physical world, making simulation an essential tool for development, testing, and validation...',
      module: 'module2',
      chapter: 'Chapter 1'
    },
    {
      id: 5,
      title: 'Chapter 5 - NVIDIA Isaac Sim Integration',
      url: '/docs/module-3/chapter-5',
      excerpt: 'NVIDIA Isaac Sim and Isaac ROS represent a significant advancement in robotics simulation technology, specifically designed for complex robotic systems like humanoid robots. Unlike traditional simulation environments, Isaac Sim provides photorealistic rendering, advanced physics simulation, and seamless integration with NVIDIA\'s GPU-accelerated computing ecosystem...',
      module: 'module3',
      chapter: 'Chapter 5'
    },
    {
      id: 6,
      title: 'Chapter 1 - Vision-Language-Action Paradigm',
      url: '/docs/module-4/chapter-1',
      excerpt: 'Welcome to Module 4 of our Physical AI & Humanoid Robotics book. This chapter introduces the Vision-Language-Action (VLA) paradigm, which represents a unified approach to integrating perception, communication, and physical action in humanoid robots...',
      module: 'module4',
      chapter: 'Chapter 1'
    },
    {
      id: 7,
      title: 'Chapter 8 - Autonomous Humanoid Execution',
      url: '/docs/module-4/chapter-8',
      excerpt: 'This chapter focuses on the execution layer of autonomous humanoid systems, where high-level plans are translated into low-level motor commands. The execution layer must handle real-time constraints, sensor feedback, and dynamic adaptation to environmental changes...',
      module: 'module4',
      chapter: 'Chapter 8'
    }
  ];

  useEffect(() => {
    if (searchQuery) {
      setQueryState(searchQuery);
    }
  }, [searchQuery]);

  useEffect(() => {
    if (query.trim()) {
      performSearch(query);
    } else {
      setSearchResults([]);
    }
  }, [query, selectedModules]);

  const handleInputChange = (e) => {
    setQueryState(e.target.value);
    setQuery(e.target.value);
  };

  const handleModuleFilterChange = (module) => {
    setSelectedModules(prev => {
      if (module === 'all') {
        // If 'All Modules' is being checked, select only 'all'
        if (!prev.includes('all')) {
          return ['all'];
        } else {
          // If 'All Modules' is being unchecked, clear all selections
          return [];
        }
      } else {
        // If a specific module is being checked
        if (!prev.includes(module)) {
          // Add the module to the selection
          const newSelection = [...prev];
          // If 'all' was selected, remove it since we're now selecting specific modules
          const filteredSelection = newSelection.filter(m => m !== 'all');
          return [...filteredSelection, module];
        } else {
          // If a module is being unchecked, remove it from the selection
          const newSelection = prev.filter(m => m !== module);
          // If no modules are selected, default to 'all'
          return newSelection.length > 0 ? newSelection : ['all'];
        }
      }
    });
  };

  const performSearch = (searchQuery) => {
    setIsLoading(true);

    // Simulate search delay for better UX
    setTimeout(() => {
      let results = mockResults;

      // Filter by search query if there's a query
      if (searchQuery.trim()) {
        results = results.filter(result =>
          result.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
          result.excerpt.toLowerCase().includes(searchQuery.toLowerCase())
        );
      }

      // Filter by selected modules
      if (!selectedModules.includes('all')) {
        results = results.filter(result =>
          selectedModules.includes(result.module)
        );
      }

      setSearchResults(results);
      setIsLoading(false);
    }, 300);
  };

  return (
    <>
      <BookHead
        title="Search"
        description="Search the Physical AI & Humanoid Robotics book content across all modules and chapters."
        url={`${siteConfig.url}/book-search`}
      />
      <Layout
        title={`Search | ${siteConfig.title}`}
        description="Search the Physical AI & Humanoid Robotics book content">
        <header className={clsx('hero hero--primary', styles.heroBanner)}>
          <div className="container">
            <h1 className="hero__title">Search the Book</h1>
            <p className="hero__subtitle">Find specific topics, concepts, or examples in Physical AI & Robotics</p>

            <div className={styles.searchContainer}>
              <div className="row">
                <div className="col col--8 col--offset-2">
                  <div className={styles.searchBox}>
                    <input
                      type="text"
                      placeholder="Search Physical AI & Robotics content..."
                      value={query}
                      onChange={handleInputChange}
                      className={clsx('form-control', styles.searchInput)}
                      autoFocus
                    />
                    <button className={styles.searchButton}>
                      <svg
                        className={styles.searchIcon}
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                        xmlns="http://www.w3.org/2000/svg">
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                        />
                      </svg>
                    </button>
                  </div>

                  <div className={styles.searchFilters}>
                    <h3>Filter by Module</h3>
                    <div className={styles.filterOptions}>
                      <label className={styles.filterOption}>
                        <input
                          type="checkbox"
                          checked={selectedModules.includes('all')}
                          onChange={() => handleModuleFilterChange('all')}
                        />
                        All Modules
                      </label>
                      <label className={styles.filterOption}>
                        <input
                          type="checkbox"
                          checked={selectedModules.includes('module1')}
                          onChange={() => handleModuleFilterChange('module1')}
                        />
                        Module 1: Physical AI Foundations & the Robotic Nervous System
                      </label>
                      <label className={styles.filterOption}>
                        <input
                          type="checkbox"
                          checked={selectedModules.includes('module2')}
                          onChange={() => handleModuleFilterChange('module2')}
                        />
                        Module 2: Digital Twins & Robot Simulation
                      </label>
                      <label className={styles.filterOption}>
                        <input
                          type="checkbox"
                          checked={selectedModules.includes('module3')}
                          onChange={() => handleModuleFilterChange('module3')}
                        />
                        Module 3: Perception, Navigation & the AI Robot Brain
                      </label>
                      <label className={styles.filterOption}>
                        <input
                          type="checkbox"
                          checked={selectedModules.includes('module4')}
                          onChange={() => handleModuleFilterChange('module4')}
                        />
                        Module 4: Vision-Language-Action & Autonomous Humanoids
                      </label>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </header>

        <main className={styles.main}>
          <div className="container margin-vert--lg">
            <div className="row">
              <div className="col col--8 col--offset-2">
                {query ? (
                  <div className={styles.searchResults}>
                    <h2 className={styles.resultsHeader}>
                      Search Results for "{query}"
                    </h2>
                    <p className={styles.resultsInfo}>
                      {selectedModules.includes('all')
                        ? `Showing ${searchResults.length} results from across all Physical AI and Humanoid Robotics modules and chapters`
                        : `Showing ${searchResults.length} results from ${selectedModules.map(m =>
                            m === 'module1' ? 'Module 1' :
                            m === 'module2' ? 'Module 2' :
                            m === 'module3' ? 'Module 3' :
                            m === 'module4' ? 'Module 4' : ''
                          ).filter(Boolean).join(', ')}`}
                      }
                    </p>

                    {isLoading ? (
                      <div className={styles.loading}>
                        <p>Searching...</p>
                      </div>
                    ) : searchResults.length > 0 ? (
                      <div className={styles.resultsList}>
                        {searchResults.map((result) => (
                          <div key={result.id} className={styles.resultItem}>
                            <h3>
                              <Link to={result.url} className={styles.resultLink}>
                                {result.title}
                              </Link>
                            </h3>
                            <div className={styles.resultMeta}>
                              <span className={styles.moduleTag}>{result.module.replace('module', 'Module ')}</span>
                              <span className={styles.chapterTag}>{result.chapter}</span>
                            </div>
                            <p className={styles.resultExcerpt}>
                              {result.excerpt}
                            </p>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className={styles.noResults}>
                        <p>No results found for "{query}". Try different keywords or remove some filters.</p>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className={styles.searchInfo}>
                    <h2 className={styles.infoHeader}>How to Search</h2>
                    <p className={styles.infoText}>
                      Enter keywords in the search box above to find specific topics, concepts, or examples across all modules and chapters of the Physical AI & Humanoid Robotics book.
                    </p>
                    <div className={styles.searchTips}>
                      <h3>Search Tips</h3>
                      <ul>
                        <li>Use specific keywords related to robotics, AI, or physical systems for better results</li>
                        <li>Try different terms if you don't find what you're looking for</li>
                        <li>Use the filters to narrow results by module</li>
                        <li>Search for specific concepts like "embodied intelligence", "ROS 2", "humanoid control", or "simulation"</li>
                        <li>Combine multiple keywords with AND/OR for complex searches</li>
                      </ul>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </main>
      </Layout>
    </>
  );
}

export default SearchPage;