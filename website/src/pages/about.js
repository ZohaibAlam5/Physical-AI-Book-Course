import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import BookHead from '@site/src/components/seo/BookHead';

import styles from './about.module.css';

function AboutPage() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <>
      <BookHead
        title="About"
        description="Learn more about the Physical AI & Humanoid Robotics book, its structure, target audience, and how to use it effectively."
        url={`${siteConfig.url}/about`}
      />
      <Layout
        title={`About | ${siteConfig.title}`}
        description="Learn more about the Physical AI & Humanoid Robotics book and its purpose">
        <header className={clsx('hero hero--primary', styles.heroBanner)}>
          <div className="container">
            <h1 className="hero__title">About This Book</h1>
            <p className="hero__subtitle">A comprehensive guide to Physical AI and Humanoid Robotics</p>
          </div>
        </header>
        <main className={styles.main}>
          <div className="container margin-vert--lg">
            <div className="row">
              <div className="col col--8 col--offset-2">
                <section className={styles.aboutSection}>
                  <h2>Book Overview</h2>
                  <p className={styles.aboutIntro}>
                    Welcome to our comprehensive guide to Physical AI and Humanoid Robotics, designed to help students, researchers,
                    engineers, and robotics enthusiasts understand the principles and practices of creating intelligent physical systems.
                    This book bridges the gap between traditional digital AI and embodied intelligence, focusing on how AI systems can
                    interact with and operate in the physical world through humanoid robots.
                  </p>

                  <div className={styles.bookStats}>
                    <div className={styles.statItem}>
                      <div className={styles.statNumber}>4</div>
                      <div className={styles.statLabel}>Main Modules</div>
                    </div>
                    <div className={styles.statItem}>
                      <div className={styles.statNumber}>48</div>
                      <div className={styles.statLabel}>Total Chapters</div>
                    </div>
                    <div className={styles.statItem}>
                      <div className={styles.statNumber}>1000+</div>
                      <div className={styles.statLabel}>Concepts Covered</div>
                    </div>
                  </div>
                </section>

                <section className={styles.aboutSection}>
                  <h2>Book Structure</h2>
                  <p>
                    This book is organized into 4 main modules, each covering essential aspects of Physical AI and Humanoid Robotics:
                  </p>
                  <div className={styles.moduleGrid}>
                    <div className={styles.moduleCard}>
                      <h3>Module 1: Physical AI Foundations & the Robotic Nervous System</h3>
                      <p>
                        Covers the core principles of embodied intelligence, ROS 2 as the robotic nervous system,
                        distributed systems, data flow from sensors to actuators, and practical Python AI agents.
                        Includes URDF modeling, simulation basics, and best practices for Physical AI.
                      </p>
                    </div>
                    <div className={styles.moduleCard}>
                      <h3>Module 2: Digital Twins & Robot Simulation</h3>
                      <p>
                        Explores the role of simulation in Physical AI, physics modeling, Gazebo simulation,
                        sensor simulation, whole-body control for humanoids, and control systems.
                        Focuses on simulation-to-reality transfer and advanced simulation techniques.
                      </p>
                    </div>
                    <div className={styles.moduleCard}>
                      <h3>Module 3: Perception, Navigation & the AI Robot Brain</h3>
                      <p>
                        Delves into NVIDIA Isaac Sim integration, synthetic data generation, photorealistic simulation,
                        visual SLAM, navigation stacks, perception pipelines, and multi-sensor fusion.
                        Covers sim-to-real transfer concepts and deep learning for perception.
                      </p>
                    </div>
                    <div className={styles.moduleCard}>
                      <h3>Module 4: Vision-Language-Action & Autonomous Humanoids</h3>
                      <p>
                        Focuses on Vision-Language-Action paradigms, voice-to-action pipelines, natural language processing,
                        LLM-driven cognitive planning, multi-modal interaction, and complete autonomous humanoid systems.
                        Integrates all concepts into a capstone autonomous humanoid project.
                      </p>
                    </div>
                  </div>
                </section>

                <section className={styles.aboutSection}>
                  <h2>Target Audience</h2>
                  <div className={styles.audienceGrid}>
                    <div className={styles.audienceCard}>
                      <h3>Students</h3>
                      <p>
                        Undergraduate and graduate students in robotics, AI, computer science, and engineering
                        seeking to understand Physical AI concepts and humanoid robotics through clear explanations
                        and practical examples.
                      </p>
                    </div>
                    <div className={styles.audienceCard}>
                      <h3>Researchers</h3>
                      <p>
                        Academic and industry researchers looking to deepen their understanding of embodied intelligence,
                        Physical AI systems, and advanced humanoid robotics techniques and methodologies.
                      </p>
                    </div>
                    <div className={styles.audienceCard}>
                      <h3>Engineers</h3>
                      <p>
                        Robotics engineers, AI developers, and technical professionals who need comprehensive resources
                        for developing and implementing Physical AI systems and humanoid robots.
                      </p>
                    </div>
                  </div>
                </section>

                <section className={styles.aboutSection}>
                  <h2>Learning Approach</h2>
                  <p>
                    Our book follows a structured learning approach designed to maximize comprehension and retention of Physical AI concepts:
                  </p>
                  <ul className={styles.learningApproach}>
                    <li><strong>Embodied Learning</strong>: Concepts are grounded in physical reality and real-world applications</li>
                    <li><strong>Progressive Complexity</strong>: Builds from foundational concepts to advanced autonomous systems</li>
                    <li><strong>Practical Examples</strong>: Each concept is demonstrated with robotics code examples and implementations</li>
                    <li><strong>Simulation to Reality</strong>: Emphasizes the transfer from simulation to real-world robotics</li>
                    <li><strong>Interdisciplinary Focus</strong>: Integrates AI, robotics, control systems, and perception</li>
                  </ul>
                </section>

                <section className={styles.aboutSection}>
                  <h2>Technical Details</h2>
                  <div className={styles.techGrid}>
                    <div className={styles.techCard}>
                      <h3>ROS 2 Integration</h3>
                      <p>
                        Throughout the book, we use ROS 2 as the foundational framework for building robotic systems,
                        emphasizing best practices for distributed computing and real-time control.
                      </p>
                    </div>
                    <div className={styles.techCard}>
                      <h3>Simulation First</h3>
                      <p>
                        We leverage advanced simulation environments like Gazebo and NVIDIA Isaac Sim for safe,
                        efficient development and testing before real-world deployment.
                      </p>
                    </div>
                    <div className={styles.techCard}>
                      <h3>Open Source Focus</h3>
                      <p>
                        All examples and implementations use open-source tools and frameworks to ensure accessibility
                        and reproducibility for learners and practitioners.
                      </p>
                    </div>
                  </div>
                </section>

                <section className={styles.aboutSection}>
                  <h2>How to Use This Book</h2>
                  <p>
                    For the best learning experience in Physical AI and Humanoid Robotics, we recommend:
                  </p>
                  <ol className={styles.usageGuidelines}>
                    <li>Start with Module 1 if you're new to Physical AI concepts</li>
                    <li>Work through chapters sequentially for progressive skill building</li>
                    <li>Implement the code examples and experiment with parameters</li>
                    <li>Use simulation environments to test concepts safely</li>
                    <li>Refer to the table of contents for an overview of all content</li>
                    <li>Connect theoretical concepts to practical implementations</li>
                    <li>Explore the relationships between perception, action, and cognition</li>
                  </ol>
                </section>

                <div className={styles.ctaSection}>
                  <Link to="/docs/intro" className="button button--primary button--lg">
                    Start Reading
                  </Link>
                  <Link to="/toc" className="button button--secondary button--lg">
                    View Table of Contents
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </main>
      </Layout>
    </>
  );
}

export default AboutPage;