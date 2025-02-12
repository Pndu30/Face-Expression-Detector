


# Face-Expression Detector





## Environment and Installation



### 1. Python




I am using Python 3.12.4 to run this



### 2. Dependencies Setup



You can use Conda to make an environment or just download the requirements locally by doing
```bash
pip install -r requirement.txt
```



## Training the model



Although I have not been able to replicate the accuracy shown in the paper, I was able to reach an accuracy of 54.87% and F1-Score of 0.5139 with a seed of 42 with the FER2013 dataset. 


To train the model you can run the training.ipynb notebook



## Running the cam





## Sources and Citations





```text
@ARTICLE{10812829,
  author={Roy, Arnab Kumar and Kathania, Hemant Kumar and Sharma, Adhitiya and Dey, Abhishek and Ansari, Md. Sarfaraj Alam},
  journal={IEEE Signal Processing Letters}, 
  title={ResEmoteNet: Bridging Accuracy and Loss Reduction in Facial Emotion Recognition}, 
  year={2024},
  pages={1-5},
  keywords={Emotion recognition;Feature extraction;Convolutional neural networks;Accuracy;Training;Computer architecture;Residual neural networks;Facial features;Face recognition;Facial Emotion Recognition;Convolutional Neural Network;Squeeze and Excitation Network;Residual Network},
  doi={10.1109/LSP.2024.3521321}
}
```

```text
@MISC{Goodfeli-et-al-2013,
       author = {Goodfellow, Ian and Erhan, Dumitru and Carrier, Pierre-Luc and Courville, Aaron and Mirza, Mehdi and Hamner, Ben and Cukierski, Will and Tang, Yichuan and Thaler, David and Lee, Dong-Hyun and Zhou, Yingbo and Ramaiah, Chetan and Feng, Fangxiang and Li, Ruifan and Wang, Xiaojie and Athanasakis, Dimitris and Shawe-Taylor, John and Milakov, Maxim and Park, John and Ionescu, Radu and Popescu, Marius and Grozea, Cristian and Bergstra, James and Xie, Jingjing and Romaszko, Lukasz and Xu, Bing and Chuang, Zhang and Bengio, Yoshua},
     keywords = {competition, dataset, representation learning},
        title = {Challenges in Representation Learning: A report on three machine learning contests},
         year = {2013},
  institution = {Unicer},
          url = {http://arxiv.org/abs/1307.0414},
     abstract = {The ICML 2013 Workshop on Challenges in Representation
Learning focused on three challenges: the black box learning challenge,
the facial expression recognition challenge, and the multimodal learn-
ing challenge. We describe the datasets created for these challenges and
summarize the results of the competitions. We provide suggestions for or-
ganizers of future challenges and some comments on what kind of knowl-
edge can be gained from machine learning competitions.

http://deeplearning.net/icml2013-workshop-competition}
}
```