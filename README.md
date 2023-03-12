# Can Demographic Factors Improve Text Classification? Revisiting Demographic Adaptation in the Age of Transformers

Authors: Chia-Chien Hung, Anne Lauscher, Dirk Hovy, Simone Paolo Ponzetto, Goran Glavaš 

EACL 2023. Findings: coming soon ([arXiv](https://arxiv.org/pdf/2210.07362.pdf))

## Introduction
Demographic factors (e.g., gender or age) shape our language. Previous work showed that incorporating demographic factors can consistently improve performance for various NLP tasks with traditional NLP models. In this work, we investigate whether these previous findings still hold with state-of-the-art pretrained Transformer-based language models (PLMs). We use three common specialization methods proven effective for incorporating external knowledge into pretrained Transformers (e.g., domain-specific or geographic knowledge). We adapt the language representations for the demographic dimensions of gender and age, using continuous language modeling and dynamic multi-task learning for adaptation, where we couple language modeling objectives with the prediction of demographic classes. Our results, when employing a multilingual PLM, show substantial gains in task performance across four languages (English, German, French, and Danish), which is consistent with the results of previous work. However, controlling for confounding factors – primarily domain and language proficiency of Transformer-based PLMs – shows that downstream perfor- mance gains from our demographic adaptation do not actually stem from demographic knowledge. Our results indicate that demographic specialization of PLMs, while holding promise for positive societal impact, still represents an unsolved problem for (modern) NLP.


## Citation
If you use any source codes, or datasets included in this repo in your work, please cite the following paper:
<pre>
@article{hung2022can,
  title={Can Demographic Factors Improve Text Classification? Revisiting Demographic Adaptation in the Age of Transformers},
  author={Hung, Chia-Chien and Lauscher, Anne and Hovy, Dirk and Ponzetto, Simone Paolo and Glava{\v{s}}, Goran},
  journal={arXiv preprint arXiv:2210.07362},
  year={2022}
}
</pre>


## Datasets
The datasets contain two main parts: 
1. data used for intermediate training purpose, in order to encode knowledge via the sociodemographic-specific corpus. You can download the data from [here](https://drive.google.com/file/d/1gINzYBqO1ZZjkY8Q0FexWO-IyhZd1ycH/view?usp=sharing). 
2. data used for downstream tasks. You can download the data for [gender](https://drive.google.com/file/d/1kbskcGxd7Sh215FlknizOHTJ3cTdBI3F/view?usp=sharing) and [age](https://drive.google.com/file/d/1kyc0MS6z7nCDUe3kPetL2KDUldjftcKd/view?usp=sharing).


## Structure
This repository is currently under the following structure:
```
.
└── data
└── downstream
└── specialization
└── README.md
```
