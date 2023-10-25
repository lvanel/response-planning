# Transparency in Large Language Models: Planning Social and Emotional Responses Using Next Labels Sequence Prediction 

_Code related to a submission for COLING-LREC 2024_

---

The architecture we propose in this paper is composed of two modules: the first step is to predict the sequence of labels \ref{fig:visub}, both the dialogue strategies and the emotion labels, that represents the behaviour expected by the agent in the next utterance. Then, this sequence is used to condition the selection of a final response from a set of generated candidate answers.

In order to asses the efficiency of the proposed architecture, we formalise the following hypothesis: _Planning a response using socio-emotional labels improves the quality of the generated answer._

To investigate this hypothesis, we design two sets of experiments meant to test out and compare the performance of different approaches, such as generation, classification or prompt-based methods. This panel of open-source neural models will be used to evaluate each of the two steps of our pipeline: the multi-label sequence prediction, and the use of this sequence to condition the textual response generation. We obtain a model-agnostic benchmark that sheds light on the viability of our planning strategy and architecture. However, our architecture requires a large volume of conversations annotated in social and emotional labels, preferably multiple labels per speaker turn. This strict condition on the labels is rare when it comes to task-oriented corpora. Here, we use the public dataset DailyDialog which is annotated in both emotion and dialog acts.

## Experiments ##

### Experiment 1: Next Sequence of Labels Prediction ###
In this first experiment, we aim to evaluate the performance of various models on the task of predicting a sequence of labels that models the social and emotional behaviours that are expected to be displayed in a generated response to a conversational context. In other words, we want to test the first step of our approach and determine the most suitable model to use as the planning module. 

_> See folder "labels"_

**STEP 1**: Process data (script: daily_dialog.py)

**STEP 2**: Train various models on the considered task:
* BERT (script: bert.ipynb)
* BART (script: bart.ipynb)
* Beluga (script: beluga.ipynb)

**STEP 3**: Get metrics + Randiom Selector (script: evaluate_sequence_labels.ipynb)


### Experiment 2: Conditional Response Generation ###
Once we've confirmed the quality of the predicted labels, we want to test our main hypothesis: does planning the response by imposing a list of expected socio-emotional behaviours yield better result? We use generative models to compare responses generated without conditioning and those planned using socio-emotional labels. The conditioning method we choose is filter & reranking a set of candidate responses.

**STEP 1**: Generate response (_see folder: "generation"_) and outputs csv of the N candidate responses for each test sample.
* BART (script: bart.ipynb)
* GPT2 /DialoGPT: (script: gpt2_dialoGPT.ipynb)
* Beluga: Both as a filter and rerank & direct prompting (script: beluga.ipynb)

**STEP 2**: Filter & Rerank (_see folder: "filter_rerank"_)
* In this folder you should have the model Bert Current trained using the script labels/bert.ipynb -> task = current
* The script filter_rerank.ipynb takes care of loading the candidate responses generated previously, getting the set of "expected labels" (either from the dataset if CD1, or generating them using BART NO-CD if CD2). For each test sample, BERT Current is called to predict the sequence of labels present in each candidate. The sequences of labels of each candidate is matched with the expected sequence and ordered using Normalized Levenshtein Distance. The highest-ranking candidate is selected as the final response. Metrics are computed.
