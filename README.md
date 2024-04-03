# pPre-training a BERT Transformer Neural Network for Swedish political discourse


**The problem** :  The objective of this project is to train a dedicated BERT model tailored to the Swedish parliamentary corpus. The issue is that existing BERT models both lack vocabulary from the 19th century and are trained on a more general corpus, thereby restricting the scope of analysis for such a historical corpus and the performance on political data.
Objective : Train a BERT model specifically for Swedish political discourse over a longer time period. 

**Proposition of Models** : 

Continue Pre-training using KB-bert on the swedish corpus :</ins>

Description : we can start from the KB-Bert specialized in the Swedish language and then use a Domain-Adaptive Pretraining on the Swedish corpus in order to improve the model for the political settings

Using exBert/ VART  to extend the model with domain specific vocabulary :</ins>

Description : The particularity of exBert lies in its combination of a from-scratch training model and a fine-tuning approach. It integrates an additional extension vocabulary into the Bert model's structure by increasing the embedding layer, specifically dedicated to domain-specific vocabulary through WordPiece. Additionally, it incorporates a module extension into each transformer layer. The author initially applied this model in the biomedical field; there is potential for experimentation with the same architecture in the context of the Swedish Parliament. There is also a recent adaptation of exBERT called VART which boost the efficiency in both pre-training and fine-tuning phases while saving computational resources. 

Pre train a new Bert model based on the corpus with MosaicBert :

Description: if we go for from scratch training, opting for Mosaic Bert is optimal, as it facilitates the creation of an efficient Bert model with minimal training hours and memory requirements. Notably, Mosaic Bert incorporates features such as Flash Attention, optimizing read/write operations between GPU HBM and GPU SRAM, and ALiBi, which calculates embedding positions in sentences using attention scores and can be extended for longer sequences. Achieving a high return necessitates adjusting the global batch size to the number of nodes. 
Advantage: it allows the model to be tailored specifically for the Swedish Parliament, addressing the issue of missing vocabulary. 
Drawback : an increased time and cost investment with no guarantee of better results



**Evaluation of the model :**

Intro classification<br />
Speech/not Speech  classification<br />
Speaker party detection<br />
Speaker gender detection<br />
Header/ non Header prediction<br />
**perplexity/perplexity ish**



