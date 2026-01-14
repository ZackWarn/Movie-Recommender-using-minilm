#!/usr/bin/env python3
"""
Upload movie_embeddings.pkl to HuggingFace Hub for cloud deployment.
This allows Render and other cloud platforms to fetch embeddings without git-lfs.
"""

import os
from huggingface_hub import HfApi, create_repo

def upload_embeddings(embeddings_file="movie_embeddings.pkl", hf_token=None):
    """
    Upload embeddings to HF Hub.
    
    Args:
        embeddings_file: Path to movie_embeddings.pkl
        hf_token: HuggingFace API token (reads from HF_API_TOKEN env if not provided)
    """
    if not os.path.exists(embeddings_file):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
    
    hf_token = hf_token or os.getenv("HF_API_TOKEN")
    if not hf_token:
        raise ValueError("HF_API_TOKEN env var not set. Get it from https://huggingface.co/settings/tokens")
    
    api = HfApi(token=hf_token)
    
    # Get your username
    user_info = api.whoami(token=hf_token)
    username = user_info["name"]
    
    repo_name = "cinematch-embeddings"
    repo_id = f"{username}/{repo_name}"
    
    print(f"Uploading to {repo_id}...")
    
    try:
        # Create repo if it doesn't exist
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=True,
            exist_ok=True,
            token=hf_token
        )
        print(f"✅ Repository created/exists: {repo_id}")
    except Exception as e:
        print(f"⚠️  Could not create repo (may already exist): {e}")
    
    # Upload file
    file_size_mb = os.path.getsize(embeddings_file) / (1024 * 1024)
    print(f"Uploading {embeddings_file} ({file_size_mb:.2f} MB)...")
    
    api.upload_file(
        path_or_fileobj=embeddings_file,
        path_in_repo=embeddings_file,
        repo_id=repo_id,
        repo_type="dataset",
        token=hf_token
    )
    
    print(f"✅ Successfully uploaded to https://huggingface.co/datasets/{repo_id}")
    print(f"\nSet this env var on Render:")
    print(f"  HF_EMBEDDINGS_REPO={repo_id}")
    print(f"  HF_API_TOKEN=<your_token>")

if __name__ == "__main__":
    upload_embeddings()
