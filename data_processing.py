import os

from datasets import Dataset

from tqdm import tqdm


def arrow_files_in_dir(directory):

    files = []

    for filename in os.listdir(directory):

        if filename.endswith(".arrow") and os.path.isfile(
            os.path.join(directory, filename)
        ):
            files.append(filename)

    return files


folder_path = r"C:\Users\bheem\OneDrive\Documents\Programs\LLM-Practice\fcc-gpt-course\Skylion007___openwebtext\plain_text\0.0.0\79d93d786212f7344586290adb811d4ae6a1762c"


output_file_train = "output_train.txt"

output_file_val = "output_val.txt"

vocab_file = "vocab.txt"


files = arrow_files_in_dir(folder_path)

# Optional but recommended so the split is reproducible
files = sorted(files)

total_files = len(files)

print("Total Arrow files found:", total_files)


# 90% training, 10% validation
split_index = int(total_files * 0.9)

files_train = files[:split_index]

files_val = files[split_index:]


print("Training files:", len(files_train))
print("Validation files:", len(files_val))


vocab = set()


# ===============================
# Process training files
# ===============================

with open(output_file_train, "w", encoding="utf-8") as outfile:

    for filename in tqdm(
        files_train,
        total=len(files_train),
        desc="Processing training files"
    ):

        file_path = os.path.join(folder_path, filename)

        dataset = Dataset.from_file(file_path)

        for example in dataset:

            text = example["text"]

            outfile.write(text)
            outfile.write("\n")

            characters = set(text)

            vocab.update(characters)


# ===============================
# Process validation files
# ===============================

with open(output_file_val, "w", encoding="utf-8") as outfile:

    for filename in tqdm(
        files_val,
        total=len(files_val),
        desc="Processing validation files"
    ):

        file_path = os.path.join(folder_path, filename)

        dataset = Dataset.from_file(file_path)

        for example in dataset:

            text = example["text"]

            outfile.write(text)
            outfile.write("\n")

            characters = set(text)

            vocab.update(characters)


# ===============================
# Write vocabulary
# ===============================

with open(vocab_file, "w", encoding="utf-8") as vfile:

    for char in sorted(vocab):

        vfile.write(char + "\n")


print("Processing complete.")

print("Created:")
print(output_file_train)
print(output_file_val)
print(vocab_file)

print("Vocabulary size:", len(vocab))