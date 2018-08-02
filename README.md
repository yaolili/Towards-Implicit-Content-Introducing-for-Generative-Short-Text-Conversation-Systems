# Towards-Implicit-Content-Introducing-for-Generative-Short-Text-Conversation-Systems
This repository is the implementation of the EMNLP '17 paper [Towards Implicit Content-Introducing for Generative Short-Text Conversation Systems](http://aclweb.org/anthology/D17-1233). If you use this code (based on [dl4mt](https://github.com/nyu-dl/dl4mt-tutorial)) or our results in your research, please cite as appropriate:
```
@inproceedings{yao2017towards,
    title={Towards implicit content-introducing for generative short-text conversation systems},
    author={Yao, Lili and Zhang, Yaoyuan and Feng, Yansong and Zhao, Dongyan and Yan, Rui},
    booktitle={Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
    pages={2190--2199},
    year={2017}
}
```

# Usage
- Step1: Prepare your query, reply, cue words file; one instance per line.
- Step2: Build dictionary using build_dictionary.py
- Step3: Update the absolute path of data in training and testing script.
- Step4: You can train and inference after that.
```sh
$ sh train.sh
$ sh test.sh
```

It's a preliminary version. Please feel free to contact with me if there are any bugs. My email: yaolili12235 AT gmail DOT com

