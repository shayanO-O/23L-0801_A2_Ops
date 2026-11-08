import os
import sys
from dotenv import load_dotenv

# Ensure Workspace Root In Python Path
sys.path.insert(0, os.path.abspath("."))
os.environ["PYTHONIOENCODING"] = "utf-8"

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv(".env")

from huggingface_hub import HfApi

SPACE_REPO_ID = "shayanO-O/vg-23l-0801-a2-ops-visionguard-api"


def deploy_to_hf_space():
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN missing in .env")

    api = HfApi(token=token)

    print(f"Creating / verifying Hugging Face Docker Space: {SPACE_REPO_ID}...")
    try:
        api.create_repo(
            repo_id=SPACE_REPO_ID,
            repo_type="space",
            space_sdk="docker",
            private=False,
            exist_ok=True,
        )
        print("Space repository initialized.")
    except Exception as e:
        print(f"Note on create_repo: {e}")

    # Ensure Champion Model Exists In api/model/
    src_model = os.path.join("model", "champion.keras")
    dst_model = os.path.join("api", "model", "champion.keras")
    if os.path.exists(src_model) and not os.path.exists(dst_model):
        import shutil
        os.makedirs(os.path.dirname(dst_model), exist_ok=True)
        shutil.copyfile(src_model, dst_model)

    print(f"\nUploading 'api/' directory to Space {SPACE_REPO_ID}...")
    api.upload_folder(
        folder_path="api",
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        commit_message="Deploy VisionGuard Champion Cloud Inference API Container",
    )

    space_url = f"https://huggingface.co/spaces/{SPACE_REPO_ID}"
    app_url = f"https://{SPACE_REPO_ID.replace('/', '-').replace('_', '-')}.hf.space"

    print("\n========================================================")
    print(" Hugging Face Space Deployed Successfully!")
    print(f" Space Web Page: {space_url}")
    print(f" Direct API URL: {app_url}")
    print("========================================================")
    return space_url, app_url


if __name__ == "__main__":
    deploy_to_hf_space()
