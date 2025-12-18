import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';

const FeatureList = [
  {
    title: 'Easy to Understand',
    description: (
      <>
        Complex technical concepts explained in simple, accessible language
        with practical examples that anyone can follow.
      </>
    ),
  },
  {
    title: 'Comprehensive Coverage',
    description: (
      <>
        Four modules covering foundational concepts, intermediate topics,
        advanced concepts, and practical applications.
      </>
    ),
  },
  {
    title: 'Real-World Applications',
    description: (
      <>
        Learn with practical examples and real-world applications that
        demonstrate how concepts work in practice.
      </>
    ),
  },
];

function Feature({ Svg, title, description }) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}