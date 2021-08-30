# exampe_repo_from_scutbds
![License](https://img.shields.io/github/license/scut-bds/exampe_repo_from_scutbds)![Download](https://img.shields.io/github/downloads/scut-bds/exampe_repo_from_scutbds/total)![repo_size](https://img.shields.io/github/repo-size/scut-bds/exampe_repo_from_scutbds)
![logo](./figure/logo.png)


**Author**: [Yirong Chen](https://github.com/scutcyr)

README: [English](https://github.com/scut-bds/exampe_repo_from_scutbds/blob/main/README.md) | [中文](https://github.com/scut-bds/exampe_repo_from_scutbds/blob/main/README-zh.md)

## Introduction
This is an open source project specification for [Research Center of Body Data Science from South China University of Technology](https://github.com/scut-bds).

## Tools
* [Git](https://git-scm.com/): Git is an open source distributed version control system for agile and efficient processing of any small or large project.
* [Github](https://github.com): GitHub is a code hosting cloud service website that helps developers store and manage their project source code, and can track, record, and control users' modifications to their code.
* [shields](https://shields.io/): Shields is an icon generator, often used to add some icons to the project to improve readability.
* [Markdown](http://www.markdown.cn/): Markdown is a text-to-HTML conversion tool used on the Web. It can generate structured HTML documents in a simple, easy-to-read and easy-to-write text format. Currently github, Stackoverflow and other websites support this format.

## Structure of the Whole Repository
+ ```./config```: Store the model's hyperparameter configuration, vocabulary, etc.
+ ```./data```: Store the data set or data sample of the training model in .txt, .json or .csv format
+ ```./eval```: Store the code used to verify the model or evaluate the results of the model output
+ ```./figure```: Storage related pictures of the project
+ ```./model```: Store model code, the folder name can also be baselines or model name, etc.
+ ```./utils```: Store the code to load the data set
+ ```LICENSE```: The copyright permission selected when github created the repository, the automatically generated file
+ ```README.md```: English version description of the project
+ ```README-zh.md```: Chinese version description of the project
+ ```requirements.txt```: the requirement file for running the model
+ ```run_train_model.sh```： the bash command file for running the model

**Note**: The above is not the only structure. You can adjust the name of directory/file according to the actual situation, and you can also add or delete the unneeded directory/file.

## Structure of the README.md file
* Introduction: One sentence is usually used to indicate the purpose of the project. If there is a corresponding published paper or data set, the corresponding link needs to be given. For example, The text below is an introduction from the project [BERT](https://github.com/google-research/bert).

BERT, or Bidirectional Encoder Representations from Transformers, is a new method of pre-training language representations which obtains state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks.

Our academic paper which describes BERT in detail and provides full results on a number of tasks can be found here: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805).


* 
