# Image Caption Generator (Flickr8k)

An image caption generator model using Deep Learning and Natural Language Processing (NLP). This project trains a captioning model on the Flickr8k dataset (8,000 images + captions) and can generate short natural-language captions for input images by combining a CNN-based feature extractor with an LSTM-based language model.

- `img(1).jpg` — sample image 1

## Project overview

- Dataset: Flickr8k (downloaded from Kaggle). The dataset contains images and multiple human-written captions per image.
- Feature extraction: DenseNet201 is used as the CNN backbone to extract image features. A feature extractor model is saved as `feature_extractor.keras`.
- Language modelling: Captions are tokenized and turned into sequences with `tokenizer.pkl`. The decoder is an embedding + LSTM network that uses image features to predict the next word at each timestep.
- Trained model: The complete captioning model (image + text inputs -> softmax over vocabulary) is saved as `model.keras`.

This repository contains a Jupyter notebook `ImgCapModel.ipynb` with data preprocessing, training, saving artifacts, and an inference helper `generate_and_display_caption` for visual testing.

## Quick highlights

- Tokenizer saved: `tokenizer.pkl`
- Feature extractor: `feature_extractor.keras`
- Final model: `model.keras`
- Notebook: `ImgCapModel.ipynb`
- Helper script / entry: `app.py` (if present)
- Environment and packages: `requirements.txt`

## Model architecture (short)

1. DenseNet201 (pretrained) used as a feature extractor. We take one of the final intermediate layers (bottleneck features) and save that as the image embedding.
2. Image features are passed through a small Dense layer (256 units) and reshaped to combine with the text embedding sequence.
3. Text input: an Embedding layer (vocab_size -> 256), concatenated with image features and processed by an LSTM (256 units).
4. Output: a fully-connected + softmax layer producing a probability over the vocabulary.

Loss: categorical crossentropy. Optimizer: Adam. Training used callbacks including ModelCheckpoint, EarlyStopping and ReduceLROnPlateau. Example hyperparameters from the notebook: batch_size=64, epochs up to 50 with patience callbacks.

## Files in this repo

- `ImgCapModel.ipynb` — Main notebook with full pipeline: data preprocessing, feature extraction (DenseNet201), generator, model creation, training and inference helper.
- `model.keras` — Saved trained captioning model (may be large). If you retrain, the notebook will overwrite this with the best model found.
- `feature_extractor.keras` — Saved DenseNet201-based feature extractor used at inference to compute image features.
- `tokenizer.pkl` — Pickled Keras Tokenizer that maps words <-> integer indices used at training & inference.
- `app.py` — (Optional) small runner script to load a model and generate captions programmatically. Check this file for a CLI or example usage.
- `requirements.txt` — Python dependencies used by the project (use to create a virtual environment).

## Example usage (Notebook / Colab)

1. Open `ImgCapModel.ipynb` in Colab or locally.
2. Make sure the Flickr8k images and captions file are available (in the notebook the images path used is `/content/Images/` in Colab).
3. Run the notebook cells in order. The notebook contains a helper function:

```python
# Example (taken from the notebook):
image_path = '/content/Images/1473618073_7db56a5237.jpg'
generate_and_display_caption(image_path, model_path='model.keras', tokenizer_path='tokenizer.pkl', feature_extractor_path='feature_extractor.keras')
```

This will load the saved models/tokenizer, generate a caption, and display the image with the predicted caption.

## Run locally (Windows)

1. Create and activate virtual environment (PowerShell):

```powershell
python -m venv myenv
# Activate (PowerShell)
.\myenv\Scripts\Activate.ps1
# Or use the cmd script if using cmd.exe
.\myenv\Scripts\activate.bat
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Start Jupyter (optional) and open the notebook:

```powershell
jupyter notebook ImgCapModel.ipynb
```

4. To run inference from `app.py` (if `app.py` contains an example CLI or runner), edit the paths inside to point to your local `Images` folder and the model/tokenizer files.


## Reproducibility & tips

- If you retrain the model, ensure `tokenizer.pkl` and `feature_extractor.keras` are saved after training to reuse during inference.
- Training the model end-to-end on CPU will be slow; use a GPU (Colab/Local GPU) for practical training times.
- If you run out of memory when extracting features, reduce batch size during feature extraction or compute features per-image and save them incrementally to disk.

## Next steps and possible improvements

- Fine-tune the DenseNet201 backbone on the captioning dataset rather than using fixed bottleneck features.
- Replace the decoder with an attention mechanism (Luong/Bahdanau) to improve caption quality.
- Use beam search during inference rather than greedy argmax to get better captions.
- Evaluate with BLEU / METEOR / CIDEr metrics and show examples of successes & failure cases.

## Credits & dataset

This project uses the Flickr8k dataset (available on Kaggle). Thanks to the dataset authors and Kaggle for hosting the dataset.

## License & Contact

This repo is provided as-is for educational purposes. If you'd like help improving the model, please open an issue or contact the project author.

---
