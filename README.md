# On the Limitations of Sociodemographic Adaptation with Transformers

## Introduction
Sociodemographic factors (e.g., gender or age) shape our language. Previous work showed that incorporating specific sociodemographic factors can consistently improve performance for various NLP tasks in traditional NLP models. We investigate whether these previous findings still hold with state-of-the-art pretrained Transformers. We use three common specialization methods proven effective for incorporating external knowledge into pretrained Transformers (e.g., domain-specific or geographic knowledge). We adapt the language representations for the sociodemographic dimensions of gender and age, using continuous language modeling and dynamic multi-task learning for
adaptation, where we couple language modeling with the prediction of a sociodemographic class. Our results when employing a multilingual model show substantial performance gains across four languages (English, German,
French, and Danish). These findings are in line with the results of previous work and hold promise for successful sociodemographic specialization. However, controlling for confounding factors like domain and language shows that, while sociodemographic adaptation does improve downstream performance, the gains do not always solely stem from sociodemographic knowledge. Our results indicate that sociodemographic specialization, while very important, is still an unresolved problem in NLP


## Citation
If you use any source codes, or datasets included in this repo in your work, please cite the following paper:
<pre>
@article{hung2022limitations,
  title={On the Limitations of Sociodemographic Adaptation with Transformers},
  author={Hung, Chia-Chien and Lauscher, Anne and Hovy, Dirk and Ponzetto, Simone Paolo and Glava{\v{s}}, Goran},
  journal={arXiv preprint arXiv:2208.01029},
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
