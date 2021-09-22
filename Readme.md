# Using word2vec for emojis
Training Word2Vec model for emojis using twitter data

### Data:-
We have 1 million sentences with emojis. This dataset is from twitter and you can download it from [here](https://github.com/AdiShirsath/Machine-Learning-Notes/tree/main/NLP/word2vec/word2vec_emoji/Data).

### Usage:- 
1. Training word2vec on our data
```
python3 src.train_word2vec.py --data_path="Data/corpus.txt"
```
* This will train word2vec and save trained models as word2vec.bin

2. Create instance of prediction (In testing_emoji_model.ipynb)
```
from src.prediction import Prediction
predictor = Prediction(model_path="word2vec.bin")
```
#### Methods in prediction

1. getPrediction 
* Get only emojis for given text from top 200 most similar embeddings.
```
predictor.getPrediction("cat",emoji_only=True, topn=200)
```
2. get_similarity
* How similar given two words are
```
predictor.get_similarity(w1="😘", w2="😙")
```
3. get_vector_embedding
```
predictor.get_vector_embedding("🐰")
```

### Model:-
Pre trained model can be found [here](https://drive.google.com/drive/folders/1-8zMyK-xHlf5-siX1ta3WCxJ2lTG8ev2?usp=sharing), Where woed2vec4 is final model.

### Results:-
1. Prediction

<img src="https://user-images.githubusercontent.com/75840165/133033855-74d460c8-4c39-45ac-b216-5e1e6f6f4257.png" height=300 width=500 />

2. Similarity:-

<img src="https://user-images.githubusercontent.com/75840165/133034111-cec6faa6-314f-46fe-b2c8-830f27ea4d76.png" height=100 width=500 />

