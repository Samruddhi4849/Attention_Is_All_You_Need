# Attention Is All You Need

## 1. Problem Statement

Traditional sequence-to-sequence models use Recurrent Neural
Networks (RNNs) and Convolutional Neural Networks (CNNs).

These models have limitations in parallel processing and
require sequential computation.

The paper proposes the Transformer, a model based entirely
on attention mechanisms.

## 2. Dataset Used

The Transformer was evaluated using:

- WMT 2014 English-German translation dataset
- WMT 2014 English-French translation dataset

The English-German dataset contains approximately
4.5 million sentence pairs.

## 3. Proposed Model

The paper introduces the Transformer architecture.

It consists of:

- Encoder
- Decoder
- Multi-Head Attention
- Feed-Forward Networks
- Positional Encoding

The model uses attention instead of recurrence and convolution.

## 4. Methodology

The Transformer uses:

1. Scaled Dot-Product Attention
2. Multi-Head Attention
3. Positional Encoding
4. Encoder-Decoder Architecture

Multi-head attention allows the model to focus on different
parts of the input sequence.

Positional encoding provides information about word positions.

## 5. Performance Evaluation

The Transformer achieved strong results in machine translation.

Reported results include:

- 28.4 BLEU score for English-German translation
- 41.8 BLEU score for English-French translation

The model also demonstrated faster training compared
with several previous approaches.

## 6. Key Findings

- Attention can replace recurrence and convolution.
- The model supports better parallelization.
- Training can be performed more efficiently.
- Multi-head attention improves the model's ability
  to learn relationships between words.

## 7. Conclusion

The paper introduces the Transformer architecture,
which relies entirely on attention mechanisms.

The Transformer achieves strong translation performance
while allowing more parallelized and efficient training.