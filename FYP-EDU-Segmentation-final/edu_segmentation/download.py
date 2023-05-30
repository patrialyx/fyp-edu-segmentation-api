import requests
import os
def download_models():

    # print("[HEY] download_models has been executed.")
    # function is placed here because requirements.txt needs to be downloaded first to get requests
    def download_torch_models(model_folder, model_name):
        username = "patrialyx"
        repo_name = "edu-segmentation-models"
        tag = "v1.0.0"
        model_url = f"https://github.com/{username}/{repo_name}/releases/download/{tag}/{model_name}"
        model_path = os.path.join(os.path.dirname(__file__), f"{model_folder}/model_dependencies/{model_name}")

        response = requests.get(model_url)

        with open(model_path, "wb") as f:
            f.write(response.content)

    # just automatically download all models for the user

    # download bert-uncased
    try:
        model_name = "BERT_token_classification_final.pth"
        model_folder = "BERTTokenClassification"
        path_to_exist = os.path.join(os.path.dirname(__file__), f"{model_folder}/model_dependencies")
        if not os.path.exists(path_to_exist):
            os.makedirs(path_to_exist)
        if os.path.isfile(os.path.join(os.path.dirname(__file__), f"{model_folder}/model_dependencies/{model_name}")):
            print(f"Segbot BERT-uncased Model has already been downloaded.")
            pass
        else:
            print("Downloading Segbot BERT-uncased Model...")
            download_torch_models(model_folder, model_name)
            print("Segbot BERT-uncased Model downloaded successfully.")
    except:
        print("Failed to download BERT-uncased Segbot Model.")
    
    # download bert-cased
    try:
        model_name = "BERT_token_classification_final_cased.pth"
        model_folder = "BERTTokenClassification"
        path_to_exist = os.path.join(os.path.dirname(__file__), f"{model_folder}/model_dependencies")
        if not os.path.exists(path_to_exist):
            os.makedirs(path_to_exist)
        if os.path.isfile(os.path.join(os.path.dirname(__file__), f"{model_folder}/model_dependencies/{model_name}")):
            print(f"Segbot BERT-cased Model has already been downloaded.")
            pass
        else:
            print("Downloading Segbot BERT-cased Model...")
            download_torch_models(model_folder, model_name)
            print("Segbot BERT-cased Model downloaded successfully.")
    except:
        print("Failed to download BERT-cased Segbot Model.")
    
    # download original segbot model
    try:
        model_name = "model_segbot.torchsave"
        model_folder = "BARTTokenClassification"
        if os.path.isfile(os.path.join(os.path.dirname(__file__), f"{model_folder}/model_dependencies/{model_name}")):
            # print(f"Original Segbot Model has already been downloaded.")
            pass
        else:
            # print("Downloading Original Segbot Model...")
            download_torch_models(model_folder, model_name)
            # print("Original Segbot Model downloaded successfully.")
    except:
        print("Failed to download Original Segbot Model.")

    # download bart model
    try:
        model_name = "model_segbot_bart.torchsave"
        model_folder = "BARTTokenClassification"
        if os.path.isfile(os.path.join(os.path.dirname(__file__), f"{model_folder}/model_dependencies/{model_name}")):
            print(f"Segbot BART Model has already been downloaded.")
        else:
            print("Downloading Segbot BART Model...")
            download_torch_models(model_folder, model_name)
            print("Segbot BART Model downloaded successfully.")
    except:
        print("Failed to download Segbot BART Model.")