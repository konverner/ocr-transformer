# OCR Transformer

This is a transformer approach to OCR task. The architecture is composed of three parts: a feature extractor, encoder and decoder. The feature extractor transforms the input image into vectors of features; then temporal encoding is used to encode the order of vectors in a sequence. After that, the encoder transforms a sequence of vectors using the self-attention mechanism to get a context for the second sequence, which is target strings. The target strings also get temporal encoding and passed to the decoder that takes the context as well.

The feature extractor is a crucial part of ocr architecture. I present two models with two different extractors -- the heavy and light ones. The heavy is composed of ResNet50 layers whereas the light model uses 6 simple convolutions.

Fig. 1 -- Diagram of model1

<div>
<img src="https://github.com/conwerner/ocr-transformer/blob/master/diagram_model1.png" width="500"/>
</div>


Fig. 2 -- Diagram of model2


<div>
<img src="https://github.com/conwerner/ocr-transformer/blob/master/diagram_model2.png" width="500"/>
</div>

# Get Started

Firstly, set constants in ```const.py``` file, namely, paths to files. Secondly, edit the configurations of a model and traning processes ```config.py``` if needed.

Example of scripts launching see in [this notebook](https://github.com/conwerner/ocr-transformer/blob/master/notebooks/ocr_scripts.ipynb).

# Pretraining

Prior to training on real data, the model is pretrained on synthetic data. Crops are generated in real time using different fonts, backgrounds and Russian words; as fonts as backgrounds were created specifically for this task.

# Training

The training is done on the [Cyrillic Handwriting Dataset](https://www.kaggle.com/datasets/constantinwerner/cyrillic-handwriting-dataset), which was collected by us. It is composed of 33122 segments of handwriting texts (crops) in Russian and splited into train, and test sets with a split of 95%, 5%, respectively.

For training we used SGD and the increasing dropout from 0.1 to 0.6; the learning rate is constant and equals to 0.001, whereas the batch size is increased starting with 16.

# Metrics

Models are tested on the [Cyrillic Handwriting Dataset](https://www.kaggle.com/datasets/constantinwerner/cyrillic-handwriting-dataset)

## model1

**Word Error Rate**
|   | w/ CASE  | w/o CASE  |
|---|---|---|
|w/ PUNCT   | 0.46  | 0.44  |
|w/o PUNCT   | 0.42  | 0.41  |

**Character Error Rate**
|   | w/ CASE  | w/ PUNCT  |
|---|---|---|
|w/ CASE   |  0.09 | 0.09  |
|w/o PUNCT   | 0.09  | 0.09  |


## model2

**Word Error Rate**
|   | w/ CASE  | w/o CASE  |
|---|---|---|
|w/ PUNCT   | 0.43 | 0.41 |
|w/o PUNCT   | 0.4 | 0.38 |

**Character Error Rate**
|   | w/ CASE  | w/o CASE  |
|---|---|---|
|w/ PUNCT   | 0.1 | 0.09 |
|w/o PUNCT   | 0.09 | 0.09 |
