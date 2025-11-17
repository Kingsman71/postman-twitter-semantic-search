#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import faiss  # faiss-cpu
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


# -----------------------------
# Config
# -----------------------------

DEFAULT_COLLECTION_PATH = "Twitter API v2.postman_collection.json"
DEFAULT_INDEX_DIR = ".twitter_semantic_index"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# -----------------------------
# Utilities
# -----------------------------

def die(msg: str) -> None:
    """Print error to stderr and exit with non-zero code."""
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def ensure_deps():
    if SentenceTransformer is None:
        die(
            "sentence-transformers is not installed.\n"
            "Install with: pip install sentence-transformers"
        )
    if faiss is None:
        die(
            "faiss is not installed.\n"
            "Install with: pip install faiss-cpu"
        )


# -----------------------------
# Chunking the Postman collection
# -----------------------------

def load_postman_collection(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        die(f"Postman collection not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_url_from_postman_url(url_obj: Any) -> str:
    """
    Build a human-readable URL string from a Postman url object.
    """
    if isinstance(url_obj, str):
        return url_obj

    protocol = url_obj.get("protocol", "https")
    host = url_obj.get("host", [])
    path = url_obj.get("path", [])
    query = url_obj.get("query", [])

    host_str = ".".join(host) if isinstance(host, list) else str(host)
    path_str = "/".join(path) if isinstance(path, list) else str(path)
    base = f"{protocol}://{host_str}"
    if path_str:
        base += f"/{path_str}"

    if query:
        # query is list of {key, value, description}
        parts = []
        for q in query:
            k = q.get("key", "")
            v = q.get("value", "")
            if k:
                parts.append(f"{k}={v}")
        if parts:
            base += "?" + "&".join(parts)
    return base


def params_to_text(params: List[Dict[str, Any]], header: str) -> str:
    """
    Render query / header / body params into a readable block.
    """
    if not params:
        return ""

    lines = [header]
    for p in params:
        key = p.get("key") or p.get("name") or ""
        value = p.get("value", "")
        desc = p.get("description", "")
        optional = p.get("disabled", False)
        opt_str = " (optional)" if optional else ""
        line = f"- {key}{opt_str}"
        if value:
            line += f" = {value}"
        if desc:
            line += f" — {desc}"
        lines.append(line)
    return "\n".join(lines)


def body_to_text(body: Dict[str, Any]) -> str:
    if not body:
        return ""
    mode = body.get("mode", "")
    block = ["Body:"]
    block.append(f"mode: {mode}")
    if mode == "raw":
        raw = body.get("raw", "")
        if raw:
            block.append("raw:")
            block.append(raw)
    elif mode in ("urlencoded", "formdata"):
        block.append(params_to_text(body.get(mode, []), header=f"{mode} params:"))
    elif mode == "graphql":
        gql = body.get("graphql", {})
        query = gql.get("query", "")
        variables = gql.get("variables", "")
        if query:
            block.append("GraphQL query:")
            block.append(query)
        if variables:
            block.append("GraphQL variables:")
            block.append(json.dumps(variables, indent=2))
    return "\n".join(block)


def chunk_requests_from_items(
    items: List[Dict[str, Any]],
    folder_path: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Recursively traverse Postman collection items and build chunks.
    Each request becomes one chunk with rich context (folder, method, url, params, description).
    """
    if folder_path is None:
        folder_path = []

    chunks: List[Dict[str, Any]] = []

    for item in items:
        name = item.get("name", "").strip()
        current_path = folder_path + [name] if name else folder_path

        # Folder
        if "item" in item:
            chunks.extend(chunk_requests_from_items(item["item"], current_path))
            continue

        # Actual request
        req = item.get("request")
        if not req:
            continue

        method = req.get("method", "").upper()
        url_obj = req.get("url", {})
        url_str = build_url_from_postman_url(url_obj)

        description = req.get("description", "")
        if isinstance(description, dict):
            description = description.get("content", "")

        headers = req.get("header", [])
        query_params = url_obj.get("query", []) if isinstance(url_obj, dict) else []
        body = req.get("body", {})

        folder_str = " / ".join(folder_path) if folder_path else ""
        headers_text = params_to_text(headers, header="Headers:")
        query_text = params_to_text(query_params, header="Query params:")
        body_text = body_to_text(body)

        # Construct chunk text (this is what we embed)
        parts = []
        parts.append(f"Folder: {folder_str}" if folder_str else "Folder: (root)")
        parts.append(f"Name: {name}")
        parts.append(f"Method: {method}")
        parts.append(f"URL: {url_str}")
        if description:
            parts.append("Description:")
            parts.append(description)
        if headers_text:
            parts.append(headers_text)
        if query_text:
            parts.append(query_text)
        if body_text:
            parts.append(body_text)

        chunk_text = "\n".join(parts)

        chunk = {
            "id": len(chunks),  # will be overwritten but kept for debugging
            "name": name,
            "method": method,
            "url": url_str,
            "description": description,
            "folder": folder_str,
            "text": chunk_text,
        }
        chunks.append(chunk)

    # Normalize IDs to index order
    for idx, c in enumerate(chunks):
        c["id"] = idx

    return chunks


# -----------------------------
# Semantic index builder
# -----------------------------

class SemanticSearchEngine:
    def __init__(
        self,
        index_dir: str = DEFAULT_INDEX_DIR,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        ensure_deps()
        self.index_dir = index_dir
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None  # type: ignore
        self.chunks: List[Dict[str, Any]] = []

    # ---------- paths ----------

    @property
    def index_path(self) -> str:
        return os.path.join(self.index_dir, "faiss.index")

    @property
    def chunks_path(self) -> str:
        return os.path.join(self.index_dir, "chunks.json")

    @property
    def meta_path(self) -> str:
        return os.path.join(self.index_dir, "meta.json")

    def has_index(self) -> bool:
        return os.path.exists(self.index_path) and os.path.exists(self.chunks_path)

    # ---------- build ----------

    def build_index(self, collection_path: str) -> None:
        os.makedirs(self.index_dir, exist_ok=True)

        collection = load_postman_collection(collection_path)
        items = collection.get("item", [])
        chunks = chunk_requests_from_items(items)
        if not chunks:
            die("No request items found in the Postman collection.")

        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # use cosine similarity via inner product
        )
        embeddings = embeddings.astype("float32")
        dim = embeddings.shape[1]

        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        faiss.write_index(index, self.index_path)
        with open(self.chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        meta = {"model_name": self.model_name}
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        self.index = index
        self.chunks = chunks

    # ---------- load ----------

    def load_index(self) -> None:
        if not self.has_index():
            die(
                f"No index found in '{self.index_dir}'. "
                f"Run with --build_index or delete the directory and try again."
            )

        self.index = faiss.read_index(self.index_path)
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    # ---------- query ----------

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None or not self.chunks:
            self.load_index()

        query_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(query_emb, top_k)
        scores = scores[0]
        indices = indices[0]

        results: List[Dict[str, Any]] = []
        for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "name": chunk.get("name"),
                    "method": chunk.get("method"),
                    "url": chunk.get("url"),
                    "folder": chunk.get("folder"),
                    "description": chunk.get("description"),
                    "text": chunk.get("text"),
                }
            )
        return results


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semantic search over Twitter API Postman documentation."
    )
    parser.add_argument(
        "--query",
        type=str,
        required=False,
        help='Search query, e.g. "How do I fetch tweets with expansions?"',
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top results to return (default: 5)",
    )
    parser.add_argument(
        "--build_index",
        action="store_true",
        help="Rebuild the FAISS index from the Postman collection.",
    )
    parser.add_argument(
        "--collection_path",
        type=str,
        default=DEFAULT_COLLECTION_PATH,
        help=f"Path to Postman collection JSON (default: {DEFAULT_COLLECTION_PATH})",
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        default=DEFAULT_INDEX_DIR,
        help=f"Directory to store index and metadata (default: {DEFAULT_INDEX_DIR})",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"SentenceTransformers model name (default: {DEFAULT_MODEL_NAME})",
    )

    args = parser.parse_args()

    # enforce invocation requirement: if not building index, require --query
    if not args.build_index and not args.query:
        die("You must provide --query (or use --build_index to only build the index).")

    return args


def main():
    args = parse_args()

    engine = SemanticSearchEngine(
        index_dir=args.index_dir,
        model_name=args.model_name,
    )

    # Build index if requested or if it doesn't exist yet
    if args.build_index or not engine.has_index():
        engine.build_index(args.collection_path)

        # If user only wanted to build index, exit
        if not args.query:
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "message": "Index built successfully.",
                        "index_dir": args.index_dir,
                        "collection_path": args.collection_path,
                        "model_name": args.model_name,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return

    # Search
    results = engine.search(args.query, top_k=args.top_k)
    output = {
        "query": args.query,
        "top_k": args.top_k,
        "results": results,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
