from huggingface_hub import snapshot_download
import sys
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2], ignore_patterns=["*.bin", "*.h5", "*.msgpack", "*.pt"])
