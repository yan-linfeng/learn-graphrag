CREATE TABLE IF NOT EXISTS "GRAPH_RAG" (
    "CHUNK_ID"         VARCHAR2(200) NOT NULL,
    "CHUNK_TEXT"       VARCHAR2(4000) NOT NULL,
    "CHUNK_VECTOR"     VECTOR(1024, FLOAT32) NOT NULL,
    "ATTRIBUTES"       JSON NOT NULL
);

--------------------------------------------------------
--  DDL for Table LANGCHAIN_ORACLE_COLLECTION
--------------------------------------------------------

CREATE TABLE "LANGCHAIN_ORACLE_COLLECTION"
(
"ID" VARCHAR2(200 BYTE),
"DATA" BLOB,
"CMETADATA" CLOB
);

--------------------------------------------------------
--  DDL for Table LANGCHAIN_ORACLE_EMBEDDING
--------------------------------------------------------

CREATE TABLE "VECTOR100"."LANGCHAIN_ORACLE_EMBEDDING"
(
"DOC_ID" VARCHAR2(200 BYTE),
"EMBED_ID" NUMBER,
"EMBED_DATA" VARCHAR2(2000 BYTE),
"EMBED_VECTOR" VECTOR(1024, FLOAT32),
"CMETADATA" CLOB
);