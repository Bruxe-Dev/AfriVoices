"""
check_access.py
================
Quick standalone check: confirms whether your Hugging Face token has been
granted access to the two gated AfriVoices datasets.

Usage:
    python check_access.py
"""

from huggingface_hub import HfApi

api = HfApi()

datasets_to_check = [
    "DigitalUmuganda/Afrivoice_Swahili",
    "MCAA1-MSU/anv_data_ke",
]

for repo_id in datasets_to_check:
    try:
        files = api.list_repo_files(repo_id, repo_type="dataset")
        print(f"{repo_id}: ACCESS OK ({len(files)} files visible)")
        print("  First 5 files:", files[:5])
    except Exception as exc:
        print(f"{repo_id}: FAILED -> {exc}")
    print("-" * 70)