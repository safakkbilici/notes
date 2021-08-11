# NLP Annotations

- [Natural Language Processing (Almost) from Scratch (Collobert et al., 2011)](https://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf)
	- neural networks on POS Tagging, Chuking, NER.
	- random initialized lookup word vectors.
 	- CoNLL challenge.
	- Multitask learning.

- [Better Word Representations with Recursive Neural Networks for Morphology (Luong et al., 2013)](https://www.aclweb.org/anthology/W13-3512/)
	- recursive neural networks.
	- nearly same idea with fasttext.
	- Context-insensitive Morphological RNN
	- Context-sensitive Morphological RNN: contextual embeddings from 2013.
	- morphological segmentation toolkit: Morfessor.
	- pre* stm suf* instead of (pre* stm suf*)+ which is handy for words in morphologically rich languages.
	- no starting training from scratch, but rather, initialize the models with existing word representations.

- [On The Difficulty Of Training Recurrent Neural Networs (Pascanu et al., 2013)](http://proceedings.mlr.press/v28/pascanu13.pdf)
	- definition of exploding/vanishing gradients.
	- backpropagation through time.
	- exmples on matrix norms and spectral radius.
	- dynamical systems.
	- L1/L2, teacher forcing, LSTM, hessian-free optimization, echo state networks and their deficiencies.
	- gradient clipping.

- [Distributed Representations of Words and Phrases and their Compositionality (Mikolov et al., 2013)](https://arxiv.org/abs/1310.4546)
	- word vectors without corpus statistics & contexualization.
	- skip-grams and cbow.
	- hierarchical softmax & negative sampling.

- [Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., 2014)](https://arxiv.org/abs/1409.0473)
	- soft-attention definition.
 	- RNNencdec vs. RNNsearch (proposed).
	- RNNsearch30 > RNNencdec50.
	- soft-attention vs. hard-attention.

- [Effective Approaches to Attention-based Neural Machine Translation (Luong et al., 2015)](https://arxiv.org/abs/1508.04025)
	- WMT'14 sota.
	- proposed method: local alignment.
	- scoring types in alignment (attention mechanisms).
	- explains which scoring is better for which type of attention.
	- ensemble rocks!

- [Neural Machine Translation of Rare Words with Subword Units (Sennrich et al., 2015)](https://arxiv.org/abs/1508.07909)
	- byte pair tokenization.
	- productive word information process: agglutination and compounding.
	- variance in the degree of morphological synthesis between languages.
	- SMT examples, must look at.
	- unigram performs poorly but bigram is unable to produce some tokens in test set.


- [GloVe: Global Vectors for Word Representation (Pennington et al., 2014)](https://www.aclweb.org/anthology/D14-1162/)
	- shortcomings of word2vec.
	- basic knowledge on matrix factorization methods (LSA, HAL, COALS, HPCA).
	- where does GloVe's loss fn come from?
	- tldr: complexity of GloVe.
	- semantics vs syntax -> benchmarks
	- symmetric vs. assymetric vs. dimension.
	- SVD-S and SVD-L

- [Enriching Word Vectors with Subword Information (Bojanowski et al., 2016)](https://arxiv.org/abs/1607.04606)
	- morphologically rich languages.
	- char n-grams.
	- morphological information significally improves syntactic task. And does not help semantic questions. But optional n-gram helps.
	- other morphological representations.
	- very good vectors on small datasets.

- [Generating Wikipedia By Summarizing Long Sequences (Liu et al., 2018)](https://arxiv.org/abs/1801.10198)
	- only transformer's decoder.
	- (i1, ..., 1n) -> (o1, ..., om): (i1, ..., in, sep, o1, ..., om).
	- monolingual text-to-text tasks redundant information is re-learned about language in the encoder and decoder.
	- local attention: sequence tokens are divided into blocks of similar length and attention is performed in each block independently.
	- memory compressed attention: reducing the number of keys and values by using a kernel 3 stride 3 convolution.
	- local attention layers, which only capture the local information within a block, the memory-compressed attention layers are able to exchange information globally on the entire sequence.

- [Deep contextualized word representations (Peters et al., 2018)](https://arxiv.org/abs/1802.05365?ref=hackernoon.com)
	- char convolutions as inputs.
	- one billion benchmark.
	- other context dependent papers.
	- sota on 6 benchmark.
	- first layer vs second layer representations.
	- word sense disambiguation.
	- GloVe vs. biLM.

- [Improving Language Understanding By Generative Pre-Training (Radford et al., 2018)](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
	- transformer decoder.
	- language modeling as objective function in pre-training.
	- NLI types.
	- good visualizations of fine-tuning tasks.
	- 12 decoder layer (as in BERT small).
	- comparison: language modeling as an auxiliary objective in fine-tuning.

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)
	- masked LM as objective fn in pre-training..
	- transformer encoder.
	- ELMO but transformer and deeper.
	- BERT large vs BERT base.
	- \[CLS\] and \[SEP\] tokens.
	- Wordpiece tokenizer.
	- Task specific input representations.
	- GPT vs. BERT.
	- GLUE Benchmarks.

- [ICON: Interactive Conversational Memory Network for Multimodal Emotion Detection (Hazarika et al., 2018)](https://aclanthology.org/D18-1280/)
	- multimodal features: visual & speech & text.
	- concatenating visual, speech and text representations at first level.
	- memory networks with GRU: read/write.


- [Cross-lingual Language Model Pretraining (Lample et al., 2019)](https://arxiv.org/abs/1901.07291)
	- novel unspuervised method for learning cross-lingual representation.
	- novel supervised method for cross-lingual pretraining.
	- CLM, MLM, TLM
	- XNLI
	- fine-tuning only with English on sequence classification.
	- shared subword vocab.
	- low-resource language modeling.

- [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context (Yang et al., 2019)](https://arxiv.org/abs/1901.02860)
	- context fragmentation problem.
	- model cannot capture any longer-term dependency beyond the predefined context length.
	- reusing the hidden states obtained in previous segments.
	- good illustrations.

- [XLNet: Generalized Autoregressive Pretraining for Language Understanding (Yang et al., 2019)](https://arxiv.org/abs/1906.08237)
	- defficiencies of MLM and autoregressive LM.
	- permutation language modeling
	- two-stream self-sttention for target-aware representations
	- relative segment encodings
	- using transformer-xl

- [RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al., 2019)](https://arxiv.org/abs/1907.11692)
	- training the model longer.
	- bigger batches over more data.
	- removing the next sentence prediction objective.
	- training on longer sequences.
	- dynamically changing the masking pattern applied to the training data.
	- larger byte-level BPE vocabulary.
	- segment-pairs.

- [SpanBERT: Improving Pre-training by Representing and Predicting Spans (Joshi et al., 2019)](https://arxiv.org/abs/1907.10529)
	- masking contiguous random spans, rather than random tokens.
	- the span-boundary objective encourages the model to store this span-level in- formation at the boundary tokens.
	- span length l ~ Geo(0.2), l_max = 10 -> mean(l) = 3.8
	- all the tokens in a span are re- placed with \[MASK\]or sampled tokens.
	- L(xi) = L\_MLM(xi) + L\_SBO(xi)
	- Good performance on QA and coreference resolution.

- [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations (Lan et al., 2019)](https://arxiv.org/abs/1909.11942)
	- sentence order prediction objective.
	- factorized embedding parameterization and cross-layer parameter sharing.
	- main reason behind NSP’s ineffectiveness is its lack of difficulty as a task, as compared to MLM.
	- sentence order prediction loss is based primarily on coherence. 
	- removing dropout to further increase their model capacity.

- [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension (Lewis et al., 2019)](https://arxiv.org/abs/1910.13461)
	- denoising autoencoder that maps a corrupted document to the original document it was derived from.
	- token masking, token deletion, text infilling, sentence permutation, document rotation.
	- ELI5, XSum, ConvAI2, CNN/DM.
	- the successful methods either use token deletion or masking, or self-attention masks.
	- just left-to-right decoder performs poorly on SQuAD.

- [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators (Clark et al., 2020)](https://arxiv.org/abs/2003.10555) 
	- replaced token detection.
	- the generator learns to predict the original identities of the masked-out tokens. The discriminator is trained to distinguish tokens in the data from tokens that have been replaced by generator samples.
	- no backpropagation for the discriminator loss through the generator.
	- comparison of smaller generator and smaller discriminator.
	- experiments on weight sharing of discriminator and generator.
	- comparison of larger electra and smaller electra.



- [Contextualized Emotion Recognition in Conversation as Sequence Tagging (Wang et al., 2020)](https://aclanthology.org/2020.sigdial-1.23/)
	- global context encoder with transformer encoder and individual context encoder with unidirectional LSTM for utterances
	- their approach is more like a NER task. Decoding emotions of utterances with a Conditional Random Field at the top of the model.
	- state-of-the-art on DailyDialog.

- [DCR-Net: A Deep Co-Interactive Relation Network for Joint Dialog Act Recognition and Sentiment Classification (Qin et al., 2020)](https://arxiv.org/abs/2008.06914)
	- utterance encoding with bi-LSTM at first layer, self-attention for dialogue and semantic representations.
	- different co-interactive relation layers.
	- different decoding layers for act and semantics of utterances.

- [COSMIC: COmmonSense knowledge for eMotion Identification in Conversations (Ghosal et al., 2020)](https://arxiv.org/abs/2010.02795)
	- RoBERTa for extracting utterances’ contextualized representations.
	- extracting commonsense features with commonsense transformer model COMET: intent of speaker, effect on speaker, reaction of speaker, effect of listeners, reaction of listeners.

