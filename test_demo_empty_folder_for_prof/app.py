from flask import Flask
from edu_segmentation import run_segbot_bart


app = Flask(__name__)

@app.route('/')
def print_display():
    text = "jamine has a friend named john and they like to buy ice cream together however she"
    output_var = run_segbot_bart.run_segbot_bart(text, "conjunction_words")
    # output_var = run_segbot_bart.run_segbot_bart(text)
    return output_var

if __name__ == '__main__':
    app.run()
