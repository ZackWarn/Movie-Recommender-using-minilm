"""
Make the embeddings dataset public on HuggingFace Hub
"""
import os
from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError

hf_token = os.getenv("HF_API_TOKEN")
if not hf_token:
    print("❌ HF_API_TOKEN not set. Please set it as an environment variable.")
    exit(1)

api = HfApi()
repo_id = "VibinJethro/cinematch-embeddings"

try:
    # Update repo to be public (using new API)
    api.update_repo_settings(
        repo_id=repo_id,
        private=False,
        repo_type="dataset",
        token=hf_token
    )
    print(f"✅ Dataset {repo_id} is now public!")
    print(f"   URL: https://huggingface.co/datasets/{repo_id}")
except RepositoryNotFoundError:
    print(f"❌ Dataset not found: {repo_id}")
    print("   Make sure you've uploaded embeddings first with upload_embeddings_to_hub.py")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)
