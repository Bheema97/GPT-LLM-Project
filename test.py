from datasets import Dataset

test = Dataset.from_file(
    r"C:/Users/bheem/OneDrive/Documents/Programs/LLM-Practice/fcc-gpt-course/Skylion007___openwebtext/plain_text/0.0.0/79d93d786212f7344586290adb811d4ae6a1762c/openwebtext-train-00000-of-00080.arrow"
)

print(test)
print(test.column_names)
print(test[0])