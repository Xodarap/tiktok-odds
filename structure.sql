--
-- PostgreSQL database dump
--

-- Dumped from database version 12.3
-- Dumped by pg_dump version 12.3

-- Started on 2020-06-10 13:47:17

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- TOC entry 8 (class 2615 OID 33378)
-- Name: tiktok; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA tiktok;


ALTER SCHEMA tiktok OWNER TO postgres;

--
-- TOC entry 223 (class 1259 OID 35713)
-- Name: text_extra; Type: VIEW; Schema: tiktok; Owner: postgres
--

CREATE VIEW tiktok.text_extra AS
 SELECT q.id,
    (q.te ->> 'hashtagName'::text) AS hashtag_name,
    (NULLIF((q.te ->> 'userId'::text), ''::text))::bigint AS tagged_user,
    ((q.te ->> 'isCommerce'::text))::boolean AS is_commerce
   FROM ( SELECT ((tiktok.json ->> 'id'::text))::bigint AS id,
            jsonb_array_elements((tiktok.json -> 'textExtra'::text)) AS te
           FROM public.tiktok) q;


ALTER TABLE tiktok.text_extra OWNER TO postgres;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- TOC entry 219 (class 1259 OID 33385)
-- Name: trending_music; Type: TABLE; Schema: tiktok; Owner: postgres
--

CREATE TABLE tiktok.trending_music (
    fetch_time timestamp with time zone NOT NULL,
    json jsonb
);


ALTER TABLE tiktok.trending_music OWNER TO postgres;

--
-- TOC entry 218 (class 1259 OID 33379)
-- Name: trending_tags; Type: TABLE; Schema: tiktok; Owner: postgres
--

CREATE TABLE tiktok.trending_tags (
    fetch_time timestamp with time zone NOT NULL,
    json jsonb
);


ALTER TABLE tiktok.trending_tags OWNER TO postgres;

--
-- TOC entry 220 (class 1259 OID 33391)
-- Name: users; Type: TABLE; Schema: tiktok; Owner: postgres
--

CREATE TABLE tiktok.users (
    fetch_time timestamp with time zone NOT NULL,
    json jsonb
);


ALTER TABLE tiktok.users OWNER TO postgres;

--
-- TOC entry 221 (class 1259 OID 33415)
-- Name: users_normalized; Type: VIEW; Schema: tiktok; Owner: postgres
--

CREATE VIEW tiktok.users_normalized AS
 SELECT all_data.author,
    all_data.total_likes,
    all_data.video_count,
    all_data.follower_count,
    all_data.following_count,
    all_data.bio,
    concat('https://www.tiktok.com/@', all_data.author) AS url
   FROM (( SELECT users.fetch_time,
            (((users.json -> 'userInfo'::text) -> 'user'::text) ->> 'uniqueId'::text) AS author,
            ((((users.json -> 'userInfo'::text) -> 'stats'::text) ->> 'heartCount'::text))::integer AS total_likes,
            ((((users.json -> 'userInfo'::text) -> 'stats'::text) ->> 'videoCount'::text))::integer AS video_count,
            ((((users.json -> 'userInfo'::text) -> 'stats'::text) ->> 'followerCount'::text))::integer AS follower_count,
            ((((users.json -> 'userInfo'::text) -> 'stats'::text) ->> 'followingCount'::text))::integer AS following_count,
            (((users.json -> 'userInfo'::text) -> 'user'::text) ->> 'signature'::text) AS bio
           FROM tiktok.users) all_data
     JOIN ( SELECT (((users.json -> 'userInfo'::text) -> 'user'::text) ->> 'uniqueId'::text) AS author,
            max(users.fetch_time) AS latest_fetch
           FROM tiktok.users
          GROUP BY (((users.json -> 'userInfo'::text) -> 'user'::text) ->> 'uniqueId'::text)) latest ON (((latest.author = all_data.author) AND (latest.latest_fetch = all_data.fetch_time))));


ALTER TABLE tiktok.users_normalized OWNER TO postgres;

--
-- TOC entry 222 (class 1259 OID 35692)
-- Name: videos_normalized; Type: VIEW; Schema: tiktok; Owner: postgres
--

CREATE VIEW tiktok.videos_normalized AS
 SELECT all_data.id,
    all_data.author,
    all_data.play_count,
    all_data.share_count,
    all_data.comment_count,
    all_data.like_count,
    all_data.create_time,
    all_data.fetch_time,
    all_data.representative,
    concat('https://www.tiktok.com/@', all_data.author, '/video/', all_data.id) AS url,
    te.any_hashtags,
    te.any_tagged_users,
    te.is_commerce
   FROM ((( SELECT ((tiktok.json ->> 'id'::text))::bigint AS id,
            ((tiktok.json -> 'author'::text) ->> 'uniqueId'::text) AS author,
            (((tiktok.json -> 'stats'::text) -> 'playCount'::text))::integer AS play_count,
            (((tiktok.json -> 'stats'::text) -> 'shareCount'::text))::integer AS share_count,
            (((tiktok.json -> 'stats'::text) -> 'commentCount'::text))::integer AS comment_count,
            (((tiktok.json -> 'stats'::text) -> 'diggCount'::text))::integer AS like_count,
            to_timestamp((((tiktok.json -> 'createTime'::text))::integer)::double precision) AS create_time,
            tiktok."time" AS fetch_time,
            tiktok.representative
           FROM public.tiktok) all_data
     JOIN ( SELECT ((tiktok.json ->> 'id'::text))::bigint AS id,
            max(tiktok."time") AS latest_fetch
           FROM public.tiktok
          GROUP BY ((tiktok.json ->> 'id'::text))::bigint) latest ON (((latest.id = all_data.id) AND (latest.latest_fetch = all_data.fetch_time))))
     LEFT JOIN ( SELECT text_extra.id,
            bool_or((text_extra.hashtag_name <> ''::text)) AS any_hashtags,
            bool_or((text_extra.tagged_user IS NOT NULL)) AS any_tagged_users,
            bool_or(text_extra.is_commerce) AS is_commerce
           FROM tiktok.text_extra
          GROUP BY text_extra.id) te ON ((te.id = all_data.id)));


ALTER TABLE tiktok.videos_normalized OWNER TO postgres;

--
-- TOC entry 224 (class 1259 OID 35718)
-- Name: videos_materialized; Type: MATERIALIZED VIEW; Schema: tiktok; Owner: postgres
--

CREATE MATERIALIZED VIEW tiktok.videos_materialized AS
 SELECT videos_normalized.id,
    videos_normalized.author,
    videos_normalized.play_count,
    videos_normalized.share_count,
    videos_normalized.comment_count,
    videos_normalized.like_count,
    videos_normalized.create_time,
    videos_normalized.fetch_time,
    videos_normalized.representative,
    videos_normalized.url,
    videos_normalized.any_hashtags,
    videos_normalized.any_tagged_users,
    videos_normalized.is_commerce
   FROM tiktok.videos_normalized
  WITH NO DATA;


ALTER TABLE tiktok.videos_materialized OWNER TO postgres;

--
-- TOC entry 2853 (class 1259 OID 35727)
-- Name: videos_materialized_author_idx; Type: INDEX; Schema: tiktok; Owner: postgres
--

CREATE INDEX videos_materialized_author_idx ON tiktok.videos_materialized USING btree (author);


--
-- TOC entry 2854 (class 1259 OID 35725)
-- Name: videos_materialized_create_time_idx; Type: INDEX; Schema: tiktok; Owner: postgres
--

CREATE INDEX videos_materialized_create_time_idx ON tiktok.videos_materialized USING btree (create_time);


--
-- TOC entry 2855 (class 1259 OID 35726)
-- Name: videos_materialized_id_idx; Type: INDEX; Schema: tiktok; Owner: postgres
--

CREATE INDEX videos_materialized_id_idx ON tiktok.videos_materialized USING btree (id);


-- Completed on 2020-06-10 13:47:18

--
-- PostgreSQL database dump complete
--

