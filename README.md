## Banking Apps Sentiment Analysis Project

## Project Overview
This project conducts a comprehensive sentiment analysis of customer reviews for mobile banking applications from three major Ethiopian 

banks: Commercial Bank of Ethiopia (CBE), Bank of Abyssinia (BOA), and Dashen Bank. Utilizing reviews scraped from the Google Play Store,

 the project implements a full data science pipeline, encompassing data collection, preprocessing, sentiment and thematic analysis, 

database implementation, and derivation of actionable insights.

## Project Objectives

The project is divided into four key tasks, all of which have been successfully completed:

Task 1: Data Collection and Preprocessing – Completed

Task 2: Sentiment and Thematic Analysis – Completed

Task 3: PostgreSQL Database Implementation – Completed

Task 4: Insights and Recommendations – Completed

## Project Structure

sentiment-analysis-week2/

├── data/

│   ├── raw/                  # Raw scraped review data

│   └── processed/            # Cleaned and analyzed datasets (e.g., banking_reviews.csv, reviews_with_sentiment.csv)

├── scripts/                  # Python scripts for each task

│   ├── task1_collect.py      # Data scraping and preprocessing

│   ├── task2_analyze.py      # Sentiment and thematic analysis

│   ├── task3_database.py     # Database setup and data insertion

│   └── task4_insights.py     # Insights generation and visualization

├── notebooks/                # Jupyter notebooks for interactive analysis

│   └── task4_insights.ipynb  # Exploratory analysis for Task 4

├── reports/                  # Generated reports and visualizations

│   ├── final_report.md       # Comprehensive 10+ page project report

│   ├── visualizations/       # Charts and plots (e.g., sentiment distribution, word clouds)

│   └── task4_insights.json   # Raw insights data

├── src/                      # Source code modules

│   ├── data/                 # Data handling utilities (scraper.py, preprocessor.py)

│   └── nlp/                  # NLP modules (sentiment.py, themes.py)

├── .env.example              # Example environment configuration

├── requirements.txt          # Core dependencies

├── requirements_task4.txt    # Additional dependencies for Task 4

├── bank_reviews_schema.sql   # SQL dump for database schema

├── setup_database.py         # Comprehensive database setup script

└── README.md                 # This file

## Quick Start

## Prerequisites

Python 3.8 or higher

PostgreSQL 14 or higher

## Installation Steps

1. Clone the repository:

git clone https://github.com/Saronzeleke/sentiment-analysis-week2.git

cd sentiment-analysis-week2

2. Create and activate a virtual environment

  python -m venv my_env

source my_env/bin/activate  # On macOS/Linux

# or

my_env\Scripts\activate     # On Windows

3. Install dependencies

pip install -r requirements.txt

pip install -r requirements_task4.txt  # For visualization tools in Task 4

4. Configure the environment:

Copy .env.example to .env and update with your PostgreSQL credentials.

5. Run the tasks sequentially

python scripts/task1_collect.py    # Task 1: Data collection

python scripts/task2_analyze.py    # Task 2: Sentiment analysis

python scripts/task3_database.py   # Task 3: Database implementation

python scripts/task4_insights.py   # Task 4: Insights and recommendations

## Task Details

**Task 1: Data Collection and Preprocessing (Completed)**

## Objectives:

Scrape at least 400 reviews per bank (totaling over 1,200 reviews) from the Google Play Store.

Clean the data by removing duplicates, handling missing values, and normalizing dates.

Save the processed data as a CSV file for further analysis.

## Implementation:

Utilizes src.data.scraper.GooglePlayScraper for data collection and src.data.preprocessor.DataCleaner for preprocessing. The cleaned 

dataset is stored in data/processed/banking_reviews.csv.

## Outputs:

Raw data in data/raw/.

Cleaned dataset with over 1,200 reviews, meeting the requirement of 400+ per bank.

**Task 2: Sentiment and Thematic Analysis (Completed)**

## Objectives:

Compute sentiment scores using NLP models (DistilBERT and VADER).

Extract keywords and themes using TF-IDF and spaCy.

Cluster feedback into actionable categories.

Save the results as a CSV file.

## Implementation:

Employs src.nlp.sentiment.SentimentAnalyzer for sentiment scoring and src.nlp.themes.ThemeExtractor for theme identification. Results are

 combined and saved in data/processed/reviews_with_sentiment.csv.

## Outputs:

Sentiment labels (positive, neutral, negative) and scores.

3-5 identified themes per bank.

Keyword clusters and complete analyzed dataset.

**Task 3: PostgreSQL Database Implementation (Completed)**

## Objectives:

Design and implement a relational database schema in PostgreSQL.

Insert cleaned review data using Python (psycopg2).

Verify data integrity with SQL queries.

Generate an SQL dump for deployment.

## Database Schema:
-- Banks Table
CREATE TABLE banks (
    bank_id SERIAL PRIMARY KEY,
    bank_name VARCHAR(100) NOT NULL,
    app_name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Reviews Table
CREATE TABLE reviews (
    review_id SERIAL PRIMARY KEY,
    bank_id INTEGER REFERENCES banks(bank_id),
    review_text TEXT NOT NULL,
    rating NUMERIC(3,1) CHECK (rating >= 1 AND rating <= 5),
    review_date DATE,
    sentiment_label VARCHAR(20) CHECK (sentiment_label IN ('positive', 'neutral', 'negative')),
    sentiment_score NUMERIC(4,3) CHECK (sentiment_score >= -1 AND sentiment_score <= 1),
    source VARCHAR(50) DEFAULT 'Google Play Store',
    cleaned_text TEXT,
    keywords TEXT[],
    theme VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

## Implementation:

Run python scripts/task3_database.py or python setup_database.py to create the database and insert data.

## Verification Queries:

Total reviews: SELECT COUNT(*) FROM reviews; (Expected: 1,200+).

Reviews per bank: SELECT b.bank_name, COUNT(r.review_id) FROM banks b LEFT JOIN reviews r ON b.bank_id = r.bank_id GROUP BY b.bank_name; 

(Expected: 400+ per bank).

Average rating: SELECT b.bank_name, ROUND(AVG(r.rating)::numeric, 2) FROM banks b LEFT JOIN reviews r ON b.bank_id = r.bank_id GROUP BY b.

bank_name ORDER BY avg_rating DESC;.

## Outputs:

Database bank_reviews created with over 1,200 reviews inserted.

SQL dump: bank_reviews_schema.sql.

Verification report confirming data integrity.

**Task 4: Insights and Recommendations (Completed)**

## Objectives:


Identify at least 2 satisfaction drivers and pain points per bank.

Compare performance across banks.

Generate actionable recommendations.

Create at least 5 professional visualizations.

Address ethical considerations and biases.

Compile a comprehensive report (10+ pages).

## Key Insights:

Bank,Satisfaction Drivers,Pain Points

CBE,"User-friendly interface, Reliable transactions, Good customer support","App crashes, Slow loading, Complex interface"

BOA,"Fast performance, Easy navigation, Helpful features","Login issues, Transaction errors, Notification problems"

Dashen,"Secure platform, Consistent uptime, Basic functionality","Frequent crashes, Very slow, Update complications"


## Visualizations:

Sentiment distribution (pie/bar charts).

Rating comparison (box plots).

Drivers and pain points (horizontal bar charts).

Word clouds (frequent terms).

Time series (sentiment trends).

Additional: Recommendations priority matrix.

## Actionable Recommendations:

High Priority: Fix app stability, simplify authentication, optimize performance.

## Bank-Specific:

CBE: Simplify interface, add progress indicators.

BOA: Fix transaction validation, enhance error messages.

Dashen: Resolve crashes, improve update process.


## Implementation:
Run python scripts/task4_insights.py or

explore notebooks/task4_insights.ipynb.

## Outputs:

Comprehensive report: reports/final_report.md.

Visualizations in reports/visualizations/.

Raw insights: reports/task4_insights.json.

Ethical considerations documented.

## Results Summary

**Metric        ,CBE    ,BOA,       Dashen,           Overall**

AvgRating,    3.81/5, 4.08/5,      3.52/5,            3.80/5

Positive %,    42.5%,   48.3%,      35.8%,           42.2%

Review Count,   400+,    400+,       400+            ,"1,200+"

Key Strength,   Reliability,  Speed,  Security,-  StrengthReliabilitySpeedSecurity-



## Requirements Met

Task      ,Requirement,      Status,           Evidence

1,400+ reviews perbank,        ✅,             data/processed/banking_reviews.csv

2,Sentiment scores + themes,   ✅,             reviews_with_sentiment.csv

3,"PostgreSQL with 1,000+ reviews",✅,         Database verification queries

4,2+ drivers/pain points per bank,✅,          Final report sections 3-4

4,5+ visualizations,        ✅,                        reports/visualizations/

4,10+ page report,        ✅,                          reports/final_report.md

## Generated Reports:

reports/final_report.md: Includes executive summary, performance comparison, drivers/pain points, recommendations, ethical 

considerations, and implementation roadmap.

database_verification_report.json: Data integrity checks, review counts, and quality metrics.

## Code Documentation:

All scripts include docstrings and comments.

SQL schema is documented with constraints.

Notebooks feature markdown explanations.

This README provides setup and usage instructions.


## Contributing

Fork the repository.

Create a feature branch: git checkout -b feature/improvement.

Commit changes: git commit -am 'Add some feature'.

Push to the branch: git push origin feature/improvement.

Create a Pull Request.


## License

This project is intended for educational purposes as part of a data science coursework.

## Authors

Saron Zeleke – Complete project implementation

## Acknowledgments

Google Play Store for providing review data.

PostgreSQL community for the robust database system.

Open-source libraries including pandas, scikit-learn, and transformers.

Course facilitators and reviewers.

## Project Status

All tasks (1-4) are complete.

Last Updated: December 2025

Repository: https://github.com/Saronzeleke/sentiment-analysis-week2

Contact: For questions regarding implementation or analysis, please open an issue on GitHub.