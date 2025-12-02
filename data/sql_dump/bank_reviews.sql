--
-- PostgreSQL database dump
--

\restrict 0BLgGdQrtmCpe9Tx2J40x8PvEi1I1pARVYkAtFFDPaYqHrEBTerfC4gL8PmVpCZ

-- Dumped from database version 18.0
-- Dumped by pg_dump version 18.0

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: banks; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.banks (
    bank_id integer NOT NULL,
    bank_name character varying(100) NOT NULL,
    app_name character varying(100) NOT NULL,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.banks OWNER TO postgres;

--
-- Name: reviews; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.reviews (
    review_id integer NOT NULL,
    bank_id integer,
    review_text text,
    rating double precision,
    review_date date,
    sentiment_label character varying(50),
    sentiment_score double precision,
    source character varying(50),
    cleaned_text text,
    keywords text[],
    theme character varying(100),
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.reviews OWNER TO postgres;

--
-- Name: bank_sentiment_summary; Type: VIEW; Schema: public; Owner: postgres
--

CREATE VIEW public.bank_sentiment_summary AS
 SELECT b.bank_name,
    count(r.review_id) AS total_reviews,
    avg(r.rating) AS avg_rating,
    avg(r.sentiment_score) AS avg_sentiment,
    sum(
        CASE
            WHEN ((r.sentiment_label)::text = 'positive'::text) THEN 1
            ELSE 0
        END) AS positive_count,
    sum(
        CASE
            WHEN ((r.sentiment_label)::text = 'negative'::text) THEN 1
            ELSE 0
        END) AS negative_count,
    sum(
        CASE
            WHEN ((r.sentiment_label)::text = 'neutral'::text) THEN 1
            ELSE 0
        END) AS neutral_count
   FROM (public.banks b
     LEFT JOIN public.reviews r ON ((b.bank_id = r.bank_id)))
  GROUP BY b.bank_name;


ALTER VIEW public.bank_sentiment_summary OWNER TO postgres;

--
-- Name: banks_bank_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.banks_bank_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.banks_bank_id_seq OWNER TO postgres;

--
-- Name: banks_bank_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.banks_bank_id_seq OWNED BY public.banks.bank_id;


--
-- Name: reviews_review_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.reviews_review_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.reviews_review_id_seq OWNER TO postgres;

--
-- Name: reviews_review_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.reviews_review_id_seq OWNED BY public.reviews.review_id;


--
-- Name: banks bank_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.banks ALTER COLUMN bank_id SET DEFAULT nextval('public.banks_bank_id_seq'::regclass);


--
-- Name: reviews review_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.reviews ALTER COLUMN review_id SET DEFAULT nextval('public.reviews_review_id_seq'::regclass);


--
-- Name: banks banks_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.banks
    ADD CONSTRAINT banks_pkey PRIMARY KEY (bank_id);


--
-- Name: reviews reviews_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.reviews
    ADD CONSTRAINT reviews_pkey PRIMARY KEY (review_id);


--
-- Name: idx_banks_name; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_banks_name ON public.banks USING btree (bank_name);


--
-- Name: idx_reviews_bank_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_reviews_bank_id ON public.reviews USING btree (bank_id);


--
-- Name: idx_reviews_date; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_reviews_date ON public.reviews USING btree (review_date);


--
-- Name: idx_reviews_sentiment; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_reviews_sentiment ON public.reviews USING btree (sentiment_score);


--
-- Name: reviews reviews_bank_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.reviews
    ADD CONSTRAINT reviews_bank_id_fkey FOREIGN KEY (bank_id) REFERENCES public.banks(bank_id) ON DELETE CASCADE;


--
-- PostgreSQL database dump complete
--

\unrestrict 0BLgGdQrtmCpe9Tx2J40x8PvEi1I1pARVYkAtFFDPaYqHrEBTerfC4gL8PmVpCZ

