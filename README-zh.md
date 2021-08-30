# exampe_repo_from_scutbds
![License](https://img.shields.io/github/license/scut-bds/exampe_repo_from_scutbds)![Download](https://img.shields.io/github/downloads/scut-bds/exampe_repo_from_scutbds/total)![repo_size](https://img.shields.io/github/repo-size/scut-bds/exampe_repo_from_scutbds)
![logo](./figure/logo.png)
**Author**: [陈艺荣](https://github.com/scutcyr)

README: [English](https://github.com/scut-bds/exampe_repo_from_scutbds/blob/main/README.md) | [中文](https://github.com/scut-bds/exampe_repo_from_scutbds/blob/main/README-zh.md)

## 简介
这是[华南理工大学人体数据科学实验室](https://github.com/scut-bds)的项目开源模板及其规范。 

## Tools
* [Git](https://git-scm.com/): Git 是一个开源的分布式版本控制系统，用于敏捷高效地处理任何或小或大的项目。
* [Github](https://github.com): GitHub是一个代码托管云服务网站，帮助开发者存储和管理其项目源代码，且能够追踪、记录并控制用户对其代码的修改。
* [shields](https://shields.io/): shields是一个图标生成器，常用于给项目增加一些图标以提高可读性。
* [Markdown](http://www.markdown.cn/): Markdown 是一个 Web 上使用的文本到HTML的转换工具，可以通过简单、易读易写的文本格式生成结构化的HTML文档。目前 github、Stackoverflow 等网站均支持这种格式。

## 整个仓库的结构
+ ```./config```: 存放模型的超参数配置、词表等
+ ```./data```： 存放训练模型的数据集或者数据样本，为.txt或者.json或者.csv等格式
+ ```./eval```： 存放用于验证模型或者评估模型输出的结果的好坏的代码
+ ```./figure```: 存放仓库相关的图片
+ ```./model```: 存放模型代码，该文件夹命名也可以为baselines或者模型名字等等
+ ```./utils```: 存放读取数据集的代码
+ ```LICENSE```: github创建仓库时选择的版权许可，自动生成的文件
+ ```README.md```: 仓库的英文版本说明
+ ```README-zh.md```: 仓库的中文版本说明
+ ```requirements.txt```: 仓库的依赖说明
+ ```run_train_model.sh```: 运行模型的bash命令文件

**说明**: 上面的并非唯一的结构，大家可以结合实际调整目录/文件命名，也可增删相应的目录/文件。
