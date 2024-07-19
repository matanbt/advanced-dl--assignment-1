# Adv. DL Assignment 1: What is Best for _Long Range Arena_?

Experimenting performance of different architectures and different methods of training on the Long Range Arena (LRA) dataset ListOPS.

[Project code](https://github.com/matanbt/advanced-dl--assignment-1)

## Description
We chose to evaluate on the ListOps dataset from the Long Range Arena (LRA) benchmark. 
The dataset consists of sequences of integers and the task is to predict the result of a mathematical operation on the integers.
We have implemented the data processing, models (S4, LSTM, Transformer) and the training scheme under `./data_processing.py` , `./models` and `train.py` respectively.


## Running the experiments
The experiments can be run (and were run) on the notebook `ExperimentingListOps.ipynb`, where we also describe their setting and chosen configuration. 
We note that, due to hardships with timeouts and the limited compute provided in Google Colab, along with the desire to run experiments of higher quality - we ran the experiment on the university servers (using slurm). Evaluation we ran the aforementioned notebook
(on the university servers) with the resulted outputs available in `./ExperimentingListOps__outputs.html`, or [here](https://htmlpreview.github.io/?https://github.com/matanbt/advanced-dl--assignment-1/blob/main/ExperimentingListOps__outputs.html).




## Results
The results (as printed in the last cell of the notebook) are as follows:

| training                   | model        | test_acc |
|----------------------------|--------------|----------|
| ListOps_CLS                | lstm         | 0.4687   |
| ListOps_CLS                | transformer  | 0.5123   |
| ListOps_CLS                | s4           | 0.558    |
| Wikitext_AUT->ListOps_CLS  | lstm         | 0.1129   |
| Wikitext_AUT->ListOps_CLS  | transformer  | 0.2471   |
| Wikitext_AUT->ListOps_CLS  | s4           | 0.2435   |
| ListOps_AUT->ListOps_CLS   | lstm         | 0.2331   |
| ListOps_AUT->ListOps_CLS   | transformer  | 0.4644   |
| ListOps_AUT->ListOps_CLS   | s4           | 0.5245   |


## Conclusions
From our limited experiment we can conclude the following:
- The S4 model, trained on classification, outperforms the LSTM and Transformer models on the ListOps dataset (with 55% accuracy).
- Under wikitext auto-regressive pretraining (prior to classification), transformer model seem to _slightly_ outperforms S4,
indicating that the transformer model might benefit more from pretraining.
- Our experiment can probably benefit from increasing the training time, after which we might observe different trends 
