# Hotel Review Sentiment Analysis

## Overview

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/wildangunawan/hotel-review/app.py)

In this project I'm trying to deploy a trained model with Streamlit. This model already trained with ~4.000 sentences that is balanced (1:1 ratio) between each class. Language learned by this is Bahasa Indonesia and I'm using [IndoBERT Lite Base](https://huggingface.co/indobenchmark/indobert-lite-base-p1) for pretrained model.

I also published an article on how to implement this in my Medium [here](https://wildangunawan.medium.com/bert-serverless-deployment-with-streamlit-and-its-free-5d9f20154f24).

## Usage

It's easy to deploy this on your own. All you have to do is:
1. Clone this project
2. Inside that folder, run `pip install -r requirements.txt`
3. When done, run `streamlit run app.py`.

Streamlit will take care the rest and will automatically open a new tab in your favorite browser.

## License

This project is licensed under [MIT License](LICENSE).
