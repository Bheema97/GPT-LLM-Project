from datasets import load_dataset

print("Starting OpenWebText download...")

ds = load_dataset(
    "Skylion007/openwebtext",
    cache_dir=r"C:\Users\bheem\OneDrive\Documents\Programs\LLM-Practice\fcc-gpt-course"
)

print("Download/load complete.")
print(ds)