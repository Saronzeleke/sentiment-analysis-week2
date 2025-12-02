-- Save as: database/schema.sql

-- Create database
CREATE DATABASE bank_reviews;

-- Connect to database
\c bank_reviews;

-- Create Banks table
CREATE TABLE banks (
    bank_id SERIAL PRIMARY KEY,
    bank_name VARCHAR(100) NOT NULL,
    app_name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(bank_name, app_name)
);

-- Create Reviews table
CREATE TABLE reviews (
    review_id SERIAL PRIMARY KEY,
    bank_id INTEGER NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
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

-- Create indexes for performance
CREATE INDEX idx_reviews_bank_id ON reviews(bank_id);
CREATE INDEX idx_reviews_rating ON reviews(rating);
CREATE INDEX idx_reviews_sentiment ON reviews(sentiment_score);
CREATE INDEX idx_reviews_date ON reviews(review_date);

-- Create view for common queries
CREATE VIEW bank_performance AS
SELECT 
    b.bank_name,
    b.app_name,
    COUNT(r.review_id) AS total_reviews,
    ROUND(AVG(r.rating)::numeric, 2) AS avg_rating,
    ROUND(AVG(r.sentiment_score)::numeric, 3) AS avg_sentiment,
    SUM(CASE WHEN r.sentiment_label = 'positive' THEN 1 ELSE 0 END) AS positive_count,
    SUM(CASE WHEN r.sentiment_label = 'negative' THEN 1 ELSE 0 END) AS negative_count,
    SUM(CASE WHEN r.sentiment_label = 'neutral' THEN 1 ELSE 0 END) AS neutral_count,
    MIN(r.review_date) AS earliest_review,
    MAX(r.review_date) AS latest_review
FROM banks b
LEFT JOIN reviews r ON b.bank_id = r.bank_id
GROUP BY b.bank_id, b.bank_name, b.app_name
ORDER BY avg_rating DESC;