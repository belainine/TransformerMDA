#  Multi-dimensional Attention- 3D - : A Pytorch Implementation

This is a PyTorch implementation of the Transformer MDA model


This is a multi-turn chatbot project using the a new multi-sequence to sequence framework uses the self-attention mechanism, instead of the convolution operation or the recurrent structure, and achieves the peak performance on the task of generation dialgues using Ubuntu Corpus, Cornell Movie and DailyDialog for Embeding metrics, BLEU Score and Perplexity compare to HRED, VHRED, VHCR, HRAN et ReCoSa.

> To learn more about self-attention mechanism, you could read "[A Structured Self-attentive Sentence Embedding](https://arxiv.org/abs/1703.03130)".

<p align="center">
<img src="https://github.com/belainine/TransformerMDA/blob/master/MDA.jpg" width="450">
</p>


The project support training and translation with trained model now.

Note that this project is still a work in progress.


# Requirement
- python 3.4+
- pytorch 1.3.1
- torchtext 0.4.0
- spacy 2.2.2+
- tqdm
- dill
- numpy


# Usage

### 0) Download the spacy language model.
```bash
# conda install -c conda-forge spacy 
python -m spacy download en
python -m spacy download de
```

### 1) Preprocess the data with torchtext and spacy.
```bash
python preprocess.py -lang_src de -lang_trg en -share_vocab -save_data m30k_deen_shr.pkl
```

### 2) Train the model
```bash
python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 256 -warmup 128000 -epoch 400
```

### 3) Test the model
```bash
python translate.py -data_pkl m30k_deen_shr.pkl -model trained.chkpt -output prediction.txt
```
### 1) Download and preprocess the data with bpe:

> Since the interfaces is not unified, you need to switch the main function call from `main_wo_bpe` to `main`.

```bash
python preprocess.py -raw_dir /tmp/raw_deen -data_dir ./bpe_deen -save_data bpe_vocab.pkl -codes codes.txt -prefix deen
```

### 2) Train the model
```bash
python train.py -data_pkl ./bpe_deen/bpe_vocab.pkl -train_path ./bpe_deen/deen-train -val_path ./bpe_deen/deen-val -log deen_bpe -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 256 -warmup 128000 -epoch 400
```

### 3) Test the model (not ready)
- TODO:
	- Load vocabulary.
	- Perform decoding after the translation.
---
# Performance
## Training

<p align="center">
<img src="https://imgur.com/rKeP1bb.png" width="400">
<img src="https://imgur.com/9je3X6U.png" width="400">
</p>

- Parameter settings:
  - default parameter and optimizer settings
  - label smoothing 
  - target embedding / pre-softmax linear layer weight sharing. 
  
---
# TODO
  - Evaluation on the generated text.
  - Attention weight plot.
---
# Acknowledgement
- The byte pair encoding parts are borrowed from [subword-nmt](https://github.com/rsennrich/subword-nmt/).
- The project structure, some scripts and the dataset preprocessing steps are heavily borrowed from (https://github.com/hyunwoongko/transformer).


### References

<a id="1">[1]</a> Zhang, H., Lan, Y., Pang, L., Guo, J., & Cheng, X. (2019). Recosa: Detecting the relevant contexts with self-attention for multi-turn dialogue generation. *arXiv preprint arXiv:1907.05339*. ([https://arxiv.org/abs/1907.05339](https://arxiv.org/abs/1907.05339))

<a id="2">[2]</a> Li, Y., Su, H., Shen, X., Li, W., Cao, Z., & Niu, S. (2017). Dailydialog: A manually labelled multi-turn dialogue dataset. *arXiv preprint arXiv:1710.03957*. ([https://arxiv.org/abs/1710.03957](https://arxiv.org/abs/1710.03957))

<a id="7">[7]</a> https://parl.ai/docs/index.html

