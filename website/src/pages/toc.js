import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import BookHead from '@site/src/components/seo/BookHead';

import styles from './toc.module.css';

function TableOfContents() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <>
      <BookHead
        title="Table of Contents"
        description="Complete table of contents for the Technical Book, organized by modules and chapters."
        url={`${siteConfig.url}/toc`}
      />
      <Layout
        title={`Table of Contents | ${siteConfig.title}`}
        description="Complete table of contents for the Technical Book">
        <header className={clsx('hero hero--primary', styles.heroBanner)}>
          <div className="container">
            <h1 className="hero__title">Table of Contents</h1>
            <p className="hero__subtitle">Complete overview of all modules and chapters</p>
          </div>
        </header>
        <main className={styles.main}>
          <div className="container margin-vert--lg">
            <div className="row">
              <div className="col col--8 col--offset-2">
                <section className={styles.tocSection}>
                  <h2>Module 1: Physical AI Foundations & the Robotic Nervous System</h2>
                  <ul className={styles.chapterList}>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-1/chapter-1" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>1.1</span>
                        <span className={styles.chapterTitle}>Chapter 1 - Embodied Intelligence Concepts</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-1/chapter-2" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>1.2</span>
                        <span className={styles.chapterTitle}>Chapter 2 - Digital vs Physical AI Differences</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-1/chapter-3" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>1.3</span>
                        <span className={styles.chapterTitle}>Chapter 3 - ROS 2 as Robotic Nervous System</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-1/chapter-4" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>1.4</span>
                        <span className={styles.chapterTitle}>Chapter 4 - Distributed Systems: Nodes, Topics, Services, Actions</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-1/chapter-5" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>1.5</span>
                        <span className={styles.chapterTitle}>Chapter 5 - Data Flow from Sensors to Actuators</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-1/chapter-6" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>1.6</span>
                        <span className={styles.chapterTitle}>Chapter 6 - Python AI Agents with rclpy</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-1/chapter-7" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>1.7</span>
                        <span className={styles.chapterTitle}>Chapter 7 - Humanoid Robot Structure and URDF</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-1/chapter-8" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>1.8</span>
                        <span className={styles.chapterTitle}>Chapter 8 - Practical ROS 2 Examples for Physical AI</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-1/chapter-9" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>1.9</span>
                        <span className={styles.chapterTitle}>Chapter 9 - URDF Modeling Exercises</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-1/chapter-10" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>1.10</span>
                        <span className={styles.chapterTitle}>Chapter 10 - Simulation Basics with Gazebo</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-1/chapter-11" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>1.11</span>
                        <span className={styles.chapterTitle}>Chapter 11 - ROS 2 Best Practices for Physical AI</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-1/chapter-12" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>1.12</span>
                        <span className={styles.chapterTitle}>Chapter 12 - Module 1 Capstone Project</span>
                      </Link>
                    </li>
                  </ul>
                </section>

                <section className={styles.tocSection}>
                  <h2>Module 2: Digital Twins & Robot Simulation</h2>
                  <ul className={styles.chapterList}>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-2/chapter-1" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>2.1</span>
                        <span className={styles.chapterTitle}>Chapter 1 - Role of Simulation in Physical AI</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-2/chapter-2" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>2.2</span>
                        <span className={styles.chapterTitle}>Chapter 2 - Physics, Gravity, Collisions, and Constraints in Simulation</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-2/chapter-3" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>2.3</span>
                        <span className={styles.chapterTitle}>Chapter 3 - Gazebo for Robot Simulation</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-2/chapter-4" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>2.4</span>
                        <span className={styles.chapterTitle}>Chapter 4 - Sensor Simulation: Cameras, LiDAR, IMU</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-2/chapter-5" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>2.5</span>
                        <span className={styles.chapterTitle}>Chapter 5 - Whole-Body Control for Humanoids</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-2/chapter-6" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>2.6</span>
                        <span className={styles.chapterTitle}>Chapter 6 - Control Systems for Humanoid Robots</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-2/chapter-7" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>2.7</span>
                        <span className={styles.chapterTitle}>Chapter 7 - URDF/SDF Integration with Simulators</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-2/chapter-8" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>2.8</span>
                        <span className={styles.chapterTitle}>Chapter 8 - Physics Simulation Optimization</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-2/chapter-9" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>2.9</span>
                        <span className={styles.chapterTitle}>Chapter 9 - Multi-Robot Simulation Scenarios</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-2/chapter-10" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>2.10</span>
                        <span className={styles.chapterTitle}>Chapter 10 - Simulation-to-Reality Transfer</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-2/chapter-11" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>2.11</span>
                        <span className={styles.chapterTitle}>Chapter 11 - Advanced Simulation Techniques</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-2/chapter-12" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>2.12</span>
                        <span className={styles.chapterTitle}>Chapter 12 - Module 2 Capstone Project</span>
                      </Link>
                    </li>
                  </ul>
                </section>

                <section className={styles.tocSection}>
                  <h2>Module 3: Perception, Navigation & the AI Robot Brain</h2>
                  <ul className={styles.chapterList}>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-3/chapter-1" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>3.1</span>
                        <span className={styles.chapterTitle}>Chapter 1 - NVIDIA Isaac Sim and Isaac ROS Integration</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-3/chapter-2" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>3.2</span>
                        <span className={styles.chapterTitle}>Chapter 2 - Synthetic Data Generation</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-3/chapter-3" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>3.3</span>
                        <span className={styles.chapterTitle}>Chapter 3 - Photorealistic Simulation</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-3/chapter-4" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>3.4</span>
                        <span className={styles.chapterTitle}>Chapter 4 - Visual SLAM and Localization</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-3/chapter-5" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>3.5</span>
                        <span className={styles.chapterTitle}>Chapter 5 - Navigation Stacks and Path Planning</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-3/chapter-6" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>3.6</span>
                        <span className={styles.chapterTitle}>Chapter 6 - Perception Pipelines for Object Detection</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-3/chapter-7" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>3.7</span>
                        <span className={styles.chapterTitle}>Chapter 7 - Scene Understanding</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-3/chapter-8" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>3.8</span>
                        <span className={styles.chapterTitle}>Chapter 8 - Sim-to-Real Transfer Concepts</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-3/chapter-9" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>3.9</span>
                        <span className={styles.chapterTitle}>Chapter 9 - Deep Learning for Perception</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-3/chapter-10" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>3.10</span>
                        <span className={styles.chapterTitle}>Chapter 10 - Path Planning Algorithms</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-3/chapter-11" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>3.11</span>
                        <span className={styles.chapterTitle}>Chapter 11 - Multi-Sensor Fusion</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-3/chapter-12" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>3.12</span>
                        <span className={styles.chapterTitle}>Chapter 12 - Module 3 Capstone Project</span>
                      </Link>
                    </li>
                  </ul>
                </section>

                <section className={styles.tocSection}>
                  <h2>Module 4: Vision-Language-Action & Autonomous Humanoids</h2>
                  <ul className={styles.chapterList}>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-4/chapter-1" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>4.1</span>
                        <span className={styles.chapterTitle}>Chapter 1 - Vision-Language-Action Paradigm</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-4/chapter-2" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>4.2</span>
                        <span className={styles.chapterTitle}>Chapter 2 - Voice-to-Action Pipelines</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-4/chapter-3" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>4.3</span>
                        <span className={styles.chapterTitle}>Chapter 3 - Natural Language to Task Plans</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-4/chapter-4" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>4.4</span>
                        <span className={styles.chapterTitle}>Chapter 4 - LLM-driven Cognitive Planning</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-4/chapter-5" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>4.5</span>
                        <span className={styles.chapterTitle}>Chapter 5 - Multi-modal Interaction Systems</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-4/chapter-6" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>4.6</span>
                        <span className={styles.chapterTitle}>Chapter 6 - Autonomous Humanoid Execution</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-4/chapter-7" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>4.7</span>
                        <span className={styles.chapterTitle}>Chapter 7 - Vision-Language Models for Robotics</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-4/chapter-8" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>4.8</span>
                        <span className={styles.chapterTitle}>Chapter 8 - Speech Recognition Integration</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-4/chapter-9" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>4.9</span>
                        <span className={styles.chapterTitle}>Chapter 9 - Motion Planning from Language</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-4/chapter-10" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>4.10</span>
                        <span className={styles.chapterTitle}>Chapter 10 - Human-Robot Communication</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-4/chapter-11" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>4.11</span>
                        <span className={styles.chapterTitle}>Chapter 11 - Autonomous System Integration</span>
                      </Link>
                    </li>
                    <li className={styles.chapterItem}>
                      <Link to="/docs/module-4/chapter-12" className={styles.chapterLink}>
                        <span className={styles.chapterNumber}>4.12</span>
                        <span className={styles.chapterTitle}>Chapter 12 - Capstone - Complete Autonomous Humanoid</span>
                      </Link>
                    </li>
                  </ul>
                </section>

                <div className={styles.tocNavigation}>
                  <Link to="/docs/intro" className="button button--primary button--lg">
                    Start Reading
                  </Link>
                  <Link to="/" className="button button--secondary button--lg">
                    Back to Home
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

export default TableOfContents;