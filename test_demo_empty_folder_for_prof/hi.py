from edu_segmentation import download, main
import time

download.download_models()

text = ""

# granularity_levels = ["default", "conjunction_words"]
# models = ["bart", "bert_uncased", "bert_cased"]
granularity_levels = ["default"]
models = ["bart"]
for m in models:
    for g in granularity_levels:
        start_time = time.time()
        output_var = main.run_segbot(text, g, m)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"### model: {m}, granularity level: {g}")
        print(output_var)
        print(f"elapsed time: {elapsed_time}s")
