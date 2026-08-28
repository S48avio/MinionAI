import os
import json
import hashlib
import logging
import time

import numpy as np
import valkey
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


load_dotenv()
logger = logging.getLogger(__name__)


# =========================================================
# CONFIGURATION
# =========================================================

VALKEY_HOST = os.getenv("VALKEY_HOST", "localhost")
VALKEY_PORT = int(os.getenv("VALKEY_PORT", "6379"))
VALKEY_DB = int(os.getenv("VALKEY_DB", "0"))

CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))

# Semantic similarity threshold
SEMANTIC_THRESHOLD = float(
    os.getenv("SEMANTIC_THRESHOLD", "0.90")
)


# =========================================================
# EMBEDDING MODEL
# =========================================================

EMBEDDING_MODEL_NAME = os.getenv(
    "CACHE_EMBEDDING_MODEL",
    "BAAI/bge-small-en-v1.5",
)


embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)


# bge-small-en-v1.5 = 384 dimensions
EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()


# =========================================================
# VALKEY CLIENT
# =========================================================

cache_client = valkey.Valkey(
    host=VALKEY_HOST,
    port=VALKEY_PORT,
    db=VALKEY_DB,

    # IMPORTANT:
    # Vector data is binary, therefore do NOT use
    # decode_responses=True.
    decode_responses=False,

    socket_connect_timeout=2,
    socket_timeout=2,
)


# =========================================================
# VALKEY KEY NAMES
# =========================================================

EXACT_PREFIX = "minion:exact:"
SEMANTIC_PREFIX = "minion:semantic:"

SEMANTIC_INDEX = "minion:semantic:index"


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_question(question: str) -> str:
    """
    Normalize user question for exact caching.
    """

    return " ".join(
        question
        .strip()
        .lower()
        .split()
    )


# =========================================================
# EXACT CACHE KEY
# =========================================================

def make_exact_cache_key(question: str) -> str:

    normalized = normalize_question(question)

    question_hash = hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()

    return f"{EXACT_PREFIX}{question_hash}"


# =========================================================
# EMBEDDINGS
# =========================================================

def generate_embedding(question: str) -> np.ndarray:
    """
    Generate normalized FLOAT32 embedding.
    """

    embedding = embedding_model.encode(
        question,
        normalize_embeddings=True,
    )

    return np.asarray(
        embedding,
        dtype=np.float32,
    )


# =========================================================
# CREATE SEMANTIC INDEX
# =========================================================

def create_semantic_index():
    """
    Create Valkey Search HNSW vector index.

    Safe to call when application starts.
    """

    try:

        cache_client.execute_command(
            "FT.CREATE",
            SEMANTIC_INDEX,

            "ON",
            "HASH",

            "PREFIX",
            "1",
            SEMANTIC_PREFIX,

            "SCHEMA",

            "question",
            "TEXT",

            "answer",
            "TEXT",

            "route",
            "TAG",

            "embedding",
            "VECTOR",
            "HNSW",
            "6",

            "TYPE",
            "FLOAT32",

            "DIM",
            str(EMBEDDING_DIM),

            "DISTANCE_METRIC",
            "COSINE",
        )

        logger.info(
            "Created semantic cache index"
        )

    except Exception as error:

        error_message = str(error)

        # Index already exists
        if (
            "already exists" in error_message.lower()
            or
            "index already exists" in error_message.lower()
        ):

            logger.info(
                "Semantic cache index already exists"
            )

        else:

            logger.exception(
                "Failed to create semantic cache index"
            )


# =========================================================
# EXACT CACHE LOOKUP
# =========================================================

def get_exact_cached_response(
    question: str,
):

    try:

        key = make_exact_cache_key(
            question
        )

        cached = cache_client.get(key)

        if cached is None:
            return None

        data = json.loads(
            cached.decode("utf-8")
        )

        logger.info(
            "Exact cache HIT"
        )

        return data

    except Exception:

        logger.exception(
            "Exact cache lookup failed"
        )

        return None


# =========================================================
# SEMANTIC CACHE LOOKUP
# =========================================================

def get_semantic_cached_response(
    question: str,
):

    try:

        embedding = generate_embedding(
            question
        )

        vector_bytes = embedding.tobytes()

        # Search nearest cached question
        results = cache_client.execute_command(

            "FT.SEARCH",

            SEMANTIC_INDEX,

            "*=>[KNN 1 @embedding $query_vector]",

            "PARAMS",
            "2",
            "query_vector",
            vector_bytes,

            "DIALECT",
            "2",
        )


        # Expected structure:
        #
        # [
        #     total_results,
        #     key,
        #     [
        #         field,
        #         value,
        #         ...
        #     ]
        # ]


        if not results or results[0] == 0:

            logger.info(
                "Semantic cache MISS"
            )

            return None


        fields = results[2]


        cache_data = {}

        for i in range(
            0,
            len(fields),
            2,
        ):

            field = fields[i]

            value = fields[i + 1]

            if isinstance(
                field,
                bytes,
            ):
                field = field.decode(
                    "utf-8"
                )

            cache_data[field] = value


        # Valkey Search normally returns the computed vector distance as
        # __<vector-field>_score. Some compatible implementations return a
        # custom alias such as "distance", so accept either response shape.
        score_field = next(
            (
                field
                for field in (
                    "distance",
                    "__embedding_score",
                )
                if field in cache_data
            ),
            None,
        )

        if score_field is None:
            score_field = next(
                (
                    field
                    for field in cache_data
                    if field.endswith("_score")
                ),
                None,
            )

        if score_field is None:
            logger.warning(
                "Semantic cache result did not contain a vector score; fields=%s",
                sorted(cache_data),
            )
            return None

        distance = float(cache_data[score_field])


        # Valkey cosine metric returns:
        #
        # distance = 1 - cosine_similarity
        #
        # therefore:
        similarity = 1.0 - distance


        logger.info(
            "Semantic candidate similarity=%.4f",
            similarity,
        )


        if similarity < SEMANTIC_THRESHOLD:

            logger.info(
                "Semantic cache MISS "
                "(similarity %.4f < threshold %.4f)",
                similarity,
                SEMANTIC_THRESHOLD,
            )

            return None


        answer = cache_data[
            "answer"
        ]

        route = cache_data[
            "route"
        ]


        if isinstance(
            answer,
            bytes,
        ):
            answer = answer.decode(
                "utf-8"
            )


        if isinstance(
            route,
            bytes,
        ):
            route = route.decode(
                "utf-8"
            )


        logger.info(
            "Semantic cache HIT "
            "similarity=%.4f",
            similarity,
        )


        return {
            "answer": answer,
            "route": route,
            "semantic_similarity": similarity,
        }


    except Exception:

        logger.exception(
            "Semantic cache lookup failed"
        )

        return None


# =========================================================
# CACHE LOOKUP
# =========================================================

def get_cached_response(
    question: str,
):
    """
    Cache lookup order:

    1. Exact cache
    2. Semantic cache
    3. None
    """

    # -----------------------------------------------------
    # Exact cache
    # -----------------------------------------------------

    exact_result = get_exact_cached_response(
        question
    )

    if exact_result is not None:

        exact_result["cache_type"] = "exact"

        return exact_result


    # -----------------------------------------------------
    # Semantic cache
    # -----------------------------------------------------

    semantic_result = get_semantic_cached_response(
        question
    )

    if semantic_result is not None:

        semantic_result[
            "cache_type"
        ] = "semantic"

        return semantic_result


    return None

# =========================================================
# STORE CACHE
# =========================================================

def cache_response(
    question: str,
    answer: str,
    route: str,
):
    """
    Store:

    Exact cache:
        question hash -> final LLM response

    Semantic cache:
        question
        embedding
        final LLM response
        route

    RAG chunks/context are NOT stored.
    """

    try:

        # =================================================
        # EXACT CACHE
        # =================================================

        exact_key = make_exact_cache_key(
            question
        )

        exact_data = {
            "answer": answer,
            "route": route,
        }


        cache_client.setex(
            exact_key,
            CACHE_TTL,
            json.dumps(
                exact_data
            ),
        )


        # =================================================
        # SEMANTIC CACHE
        # =================================================

        embedding = generate_embedding(
            question
        )


        normalized = normalize_question(
            question
        )


        semantic_hash = hashlib.sha256(
            normalized.encode(
                "utf-8"
            )
        ).hexdigest()


        semantic_key = (
            f"{SEMANTIC_PREFIX}"
            f"{semantic_hash}"
        )


        # HASH values
        cache_client.hset(
            semantic_key,
            mapping={
                "question": question,
                "answer": answer,
                "route": route,
                "embedding": embedding.tobytes(),
                "created_at": str(
                    int(time.time())
                ),
            },
        )


        # TTL for semantic cache
        cache_client.expire(
            semantic_key,
            CACHE_TTL,
        )


        logger.info(
            "Stored exact + semantic cache entry"
        )


    except Exception:

        # Cache should never break your chatbot.
        logger.exception(
            "Failed to store cache response"
        )


# =========================================================
# HEALTH CHECK
# =========================================================

def check_valkey() -> bool:

    try:

        return bool(
            cache_client.ping()
        )

    except Exception:

        return False
