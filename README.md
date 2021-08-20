# OCR Transformer

This is a transformer approach to OCR task. The architecture is composed of three parts: a feature extractor, encoder and decoder. The feature extractor transforms the input image into vectors of features; then temporal encoding is used to encode the order of vectors in a sequence. After that, the encoder transforms a sequence of vectors using the self-attention mechanism to get a context for the second sequence, which is target strings. The target strings also get temporal encoding and passed to the decoder that takes the context as well.

![](https://github.com/constantin50/ocr-transformer/arch.png)

