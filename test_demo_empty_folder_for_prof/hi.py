from edu_segmentation import download, main
import time

download.download_models()

text = "Social media has revolutionized the way people connect and communicate in the digital age and it has become an integral part of modern society, impacting various aspects of our lives. With platforms like Facebook, Twitter, and Instagram, social media has provided individuals with unprecedented opportunities for self-expression, networking, and information sharing. It has bridged geographical barriers, allowing people from different corners of the world to interact and engage in real-time conversations. However, the widespread use of social media has also given rise to concerns regarding privacy, mental health, and the spread of misinformation."

granularity_levels = ["default", "conjunction_words"]
models = ["bart", "bert_uncased", "bert_cased"]
for m in models:
    for g in granularity_levels:
        start_time = time.time()
        output_var = main.run_segbot(text, g, m, device='cpu')
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"### model: {m}, granularity level: {g}")
        print(output_var)
        print(f"elapsed time: {elapsed_time}s")
