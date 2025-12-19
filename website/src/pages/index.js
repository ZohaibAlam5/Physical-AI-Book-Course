import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import BookHead from '@site/src/components/seo/BookHead';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            Read the Book - 5 min ⏱️
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <>
      <BookHead
        title="Home"
        description="A comprehensive book on Physical AI and Humanoid Robotics covering embodied intelligence, simulation, perception, and autonomous systems with practical examples."
        url={`${siteConfig.url}/`}
      />
      <Layout
        title={`Hello from ${siteConfig.title}`}
        description="A comprehensive guide to Physical AI and Humanoid Robotics">
        <HomepageHeader />
        <main>
          <section className={styles.features}>
            <div className="container">
              <div className="row">
                <div className="col col--3 col--offset-0">
                  <h2>Physical AI Foundations</h2>
                  <p>Start with Module 1 to build a solid foundation in embodied intelligence and ROS 2 fundamentals.</p>
                  <Link className="button button--primary" to="/docs/module-1/chapter-1">
                    Begin Module 1
                  </Link>
                </div>
                <div className="col col--3 col--offset-0">
                  <h2>Digital Twins & Simulation</h2>
                  <p>Explore advanced simulation techniques and digital twin technologies in Module 2.</p>
                  <Link className="button button--primary" to="/docs/module-2/chapter-1">
                    Explore Module 2
                  </Link>
                </div>
                <div className="col col--3 col--offset-0">
                  <h2>Perception & Navigation</h2>
                  <p>Master perception systems, navigation, and AI robot brain concepts in Module 3.</p>
                  <Link className="button button--primary" to="/docs/module-3/chapter-1">
                    Explore Module 3
                  </Link>
                </div>
                <div className="col col--3 col--offset-0">
                  <h2>Vision-Language-Action</h2>
                  <p>Build complete autonomous humanoid systems with VLA paradigms in Module 4.</p>
                  <Link className="button button--primary" to="/docs/module-4/chapter-1">
                    Explore Module 4
                  </Link>
                </div>
              </div>
            </div>
          </section>

          <section className={styles.about}>
            <div className="container padding-vert--lg">
              <div className="row">
                <div className="col col--8 col--offset-2">
                  <h2>About This Book</h2>
                  <p>
                    This comprehensive guide to Physical AI and Humanoid Robotics is designed to help students, researchers,
                    engineers, and robotics enthusiasts understand the principles and practices of creating intelligent physical systems.
                    The book bridges the gap between traditional digital AI and embodied intelligence, focusing on how AI systems can
                    interact with and operate in the physical world through humanoid robots.
                  </p>
                  <p>
                    The book is organized into 4 main modules, each covering essential aspects of Physical AI and Humanoid Robotics.
                    Each module contains 12 chapters that build upon each other to provide a complete understanding of embodied intelligence,
                    simulation, perception, and autonomous systems.
                  </p>
                </div>
              </div>
            </div>
          </section>
        </main>
      </Layout>
    </>
  );
}