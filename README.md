# Socio-Emotional Response Generation: A Human Evaluation Protocol for LLM-Based Conversational Systems 

_Code related to a submission for COLM 2024_


_Disclaimer: The code was cleaned up and presented under the notebook format to be more legible: the conversion and the anonymisation might have created some inconsistencies with the save / loading / output paths. It is recommended to check everything when running the code._

---

The architecture we propose in this paper is composed of two modules: a first model is dedicated to predicting the sequence of socio-emotional strategies that the agent is expected to follow in the next speaker turn. Then, in the second module, this sequence is fed to a generative LLM to condition the selection of a final response from a set of generated candidate answers.

In order to asses the efficiency of the proposed architecture, we formalise the following hypothesis: _Planning a response using socio-emotional labels improves the quality of the generated answer._

To investigate this hypothesis, we design two sets of experiments meant to test out and compare the performance of different approaches, such as generation, classification or prompt-based methods. This panel of open-source neural models will be used to evaluate each of the two steps of our pipeline: the multi-label sequence prediction, and the use of this sequence to condition the textual response generation. We obtain a model-agnostic benchmark that sheds light on the viability of our planning strategy and architecture. However, our architecture requires a large volume of conversations annotated in social and emotional labels, preferably multiple labels per speaker turn. This strict condition on the labels is rare when it comes to task-oriented corpora. Here, we use the public dataset DailyDialog which is annotated in both emotion and dialog acts.

### Annex: Next Sequence of Labels Prediction ###
In this first experiment, whose results are presented in Annex A of the paper, we aim to evaluate the performance of various models on the task of predicting a sequence of labels that models the social and emotional behaviours that are expected to be displayed in a generated response to a conversational context. In other words, we want to test the first step of our approach and determine the most suitable model to use as the planning module. 

_> See folder "Labels Prediction"_

**STEP 1**: Process data (script: daily_dialog.ipynb)

**STEP 2**: Train various models on the considered task:
* BERT (script: BERT_labels.ipynb)
* BART (script: BART_labels.ipynb)
* Beluga (script: BELUGA_labels.ipynb)

**STEP 3**: Get metrics + Random Selector (script: evaluate_sequence_labels.ipynb)


### Experiment: Conditional Response Generation ###
Once we've confirmed the quality of the predicted labels, we want to test our main hypothesis: does planning the response by imposing a list of expected socio-emotional behaviours yield better result? We use generative models to compare responses generated without conditioning and those planned using socio-emotional labels. The conditioning method we choose is filter & reranking a set of candidate responses.

**STEP 1**: Generate response (_see folder: "Response Generation"_) and outputs csv of the N candidate responses for each test sample.
* BART (script: Bart_dialogue.ipynb)
* GPT2 /DialoGPT: (script: GPT2_DialoGPT_dialogue.ipynb)
* Beluga: Both as a filter and rerank & direct prompting (script: Beluga_dialogue.ipynb)

**STEP 2**: Filter & Rerank (_see folder: "Filter Rerank"_)
* In this folder you should have the model Bert Current trained using the script labels/BERT_labels.ipynb -> task = current
* A script to parse the outputs of Beluga response generation: parse_Beluga.ipynb
* The script filter_rerank.ipynb takes care of loading the candidate responses generated previously, getting the set of "expected labels" (either from the dataset if CD1, or generating them using BART NO-CD if CD2). For each test sample, BERT Current is called to predict the sequence of labels present in each candidate. The sequences of labels of each candidate is matched with the expected sequence and ordered using Normalized Levenshtein Distance. The highest-ranking candidate is selected as the final response. Metrics are computed.


### Human Evaluation ###
To obtain dataset-independent results that reflect this fact, we proceed to a human evaluation on a randomly selected sample of 300 contexts extracted from the test set. After running each \{model, conditioning\} combination over our test dataset, a list of 23 generated responses per context is obtained, to which the human reference found in the dataset is added. Since the CD-GT and CD-pred conditioning methods rely on a filter \& rerank approach based on the same pool of 10 generated candidates, they can often select the same candidate. For each context, the duplicates are thus removed. 

The annotation process is divided into three steps, to reduce the workload for the annotators. First, the responses associated with the same context are divided into those that are consistent and those that are not. Second, the best responses among the consistent ones are selected by the annotators. Third, once only the "best" responses remain, they will then be annotated with more precise criteria.

(_see folder: "Human Evaluation"_)

**STEP 1 & 2**: Step 1: Filtering by Relevance and Step 2: Selecting a Top-3 best responses (_see folder: "Filtering"_)
* The code to run the evaluation platform (script: )
* The annotated data (_see folder: "data"_)

**STEP 3**: Socio-Emotional Criteria Annotation (_see folder: "SocEmo Annotation"_)
* The code to run the evaluation platform (script: )
* The annotated data (_see folder: "data"_)
