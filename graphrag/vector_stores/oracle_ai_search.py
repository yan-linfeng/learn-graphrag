# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""The LanceDB vector storage implementation package."""
import json
import os
from typing import Any

import oracledb
from dotenv import load_dotenv, find_dotenv
from oracledb import DatabaseError

from graphrag.model.types import TextEmbedder
from .base import (
    BaseVectorStore,
    VectorStoreDocument,
    VectorStoreSearchResult,
)

# read local .env file
load_dotenv(find_dotenv())

# if platform.system() == 'Linux':
#     oracledb.init_oracle_client(lib_dir="/u01/aipoc/instantclient_23_5")

# 初始化一个数据库连接
pool = oracledb.create_pool(
    dsn=os.environ["ORACLE_AI_CONNECTION_STRING"],
    min=1,
    max=5,
    increment=1
)


class OracleAIVectorSearch(BaseVectorStore):
    """The LanceDB vector storage implementation."""

    def connect(self, **kwargs: Any) -> Any:
        """Connect to the vector storage."""
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                try:
                    cursor.execute("SELECT 1 FROM DUAL")
                    print(cursor.fetchone())
                except DatabaseError as de:
                    print(f"Database error: {de}")

    def load_documents(
        self, documents: list[VectorStoreDocument], overwrite: bool = True
    ) -> None:
        """Load documents into vector storage."""
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                if overwrite:
                    pass
                    # cursor.execute("DELETE FROM langchain_oracle_embedding WHERE doc_id = :1", "1")
                # cursor.execute("TRUNCATE TABLE graph_rag")
                data = [
                    {
                        "chunk_id": document.id,
                        "chunk_text": document.text,
                        "chunk_vector": document.vector,
                        "attributes": json.dumps(document.attributes),
                    }
                    for document in documents
                    if document.vector is not None
                ]

                if len(data) == 0:
                    data = None
                else:
                    # use foor position
                    # cursor.setinputsizes(None, None, oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_JSON)
                    # use for dict
                    cursor.setinputsizes(chunk_vector=oracledb.DB_TYPE_VECTOR, attributes=oracledb.DB_TYPE_JSON)
                    insert_sql = """
                        INSERT INTO graph_rag (
                            chunk_id,
                            chunk_text,
                            chunk_vector,
                            attributes
                        ) VALUES (
                            :chunk_id,
                            :chunk_text,
                            to_vector(:chunk_vector),
                            :attributes
                        )        
                    """
                    cursor.executemany(insert_sql, data)
                    conn.commit()

    def filter_by_id(self, include_ids: list[str] | list[int]) -> Any:
        """Build a query filter to filter documents by id."""
        if len(include_ids) == 0:
            self.query_filter = None
        else:
            if isinstance(include_ids[0], str):
                id_filter = ", ".join([f"'{id}'" for id in include_ids])
                self.query_filter = f"id in ({id_filter})"
            else:
                self.query_filter = (
                    f"id in ({', '.join([str(id) for id in include_ids])})"
                )
        return self.query_filter

    def similarity_search_by_vector(
        self, query_embedding: list[float], k: int = 10, **kwargs: Any
    ) -> list[VectorStoreSearchResult]:
        """Perform a vector-based similarity search."""
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                select_sql = """
                    SELECT 
                        chunk_id,
                        chunk_text,
                        chunk_vector,
                        attributes,
                        vector_distance(chunk_vector, TO_VECTOR(:1, 1024, FLOAT32)) as distance
                    FROM graph_rag
                    ORDER BY distance
                    FETCH FIRST :2 ROWS ONLY
                """
                cursor.setinputsizes(oracledb.DB_TYPE_VECTOR)
                cursor.execute(select_sql, [query_embedding, k])
                rows = cursor.fetchall()
                # print(f"{rows=}")

                return [
                    VectorStoreSearchResult(
                        document=VectorStoreDocument(
                            id=row[0],
                            text=row[1],
                            vector=row[2],
                            attributes=json.loads(row[3]),
                        ),
                        score=2.0 - float(row[4]),
                    )
                    for row in rows
                ]

    def similarity_search_by_text(
        self, text: str, text_embedder: TextEmbedder, k: int = 10, **kwargs: Any
    ) -> list[VectorStoreSearchResult]:
        """Perform a similarity search using a given input text."""
        query_embedding = text_embedder(text)
        # print(f"{query_embedding=}")
        if query_embedding:
            return self.similarity_search_by_vector(query_embedding, k)
        return []
