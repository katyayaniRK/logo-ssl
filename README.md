# LoGo-HTR: Self-Supervised Handwritten Text Recognition — Partial Replication
![Static Badge](https://img.shields.io/badge/Field-Deep%20Learning-blue) ![Static Badge](https://img.shields.io/badge/Domain-ComputerVision-green) ![Static Badge](https://img.shields.io/badge/Method-Self--SupervisedLearning-purple) ![Static Badge](https://img.shields.io/badge/Task-HandwrittenTextRecognition-orange) ![Static Badge](https://img.shields.io/badge/Platform-KaggleColab-yellow)

**What is LoGo-HTR?**

**LoGo-HTR (Local-Global HTR)** is a self-supervised learning framework for Handwritten Text Recognition (HTR) that reduces dependence on large labeled datasets by learning visual representations from unlabeled handwritten images.

It combines two complementary objectives:


Local Loss — patch-wise contrastive loss that aligns spatial features across augmented views of the same image
Global Loss — Barlow Twins decorrelation loss that reduces feature redundancy across embedding dimensions



