# NPFL128 - Stance Detection in Tweets

This project focuses on the subtask of Sentiment Analysis that is Stance
Detection, as described in [1]. 

The dataset used is the SemEval-2016 Stance Dataset [2], which is a dataset of
tweets annotated for stance towards a given target, target of opinion, and
sentiment. An example is given below.

```
ID  Target	Tweet															Stance  Opinion towards Sentiment
113 Atheism	RT @prayerbullets: Let the righteousness, peace, and joy of the kingdom be established in my life -Rom. 14:17 #SemST	AGAINST OTHER		POSITIVE
```

Following Mohammad et al. [3], I experiment with an SVM system that uses
features drawn from word (1- to 3-gram) and character (2- to 5-gram) ngrams. I
also experiment with training 100-dimensional word embeddings with word2vec, and
using additional features from the dataset such as whether there is an opinion
expressed towards the entity of interest.

## Results

The evaluation metric used is an average micro-F1 score only on the labels
"FAVOR" and "AGAINST", which is defined as
$$F_{average} = \frac{F_{favor} + F_{against}}{2}.$$

|Method| Average F1 | Target Mentioned F1 | Target Not Mentioned F1 |
| --- | --- | --- | --- |
| Linear Kernel SVM | 0.657 | 0.701 | 0.443 |
| Linear Kernel + Target | 0.677 | 0.717 | 0.244 |
| Linear Kernel + Target + WE | 0.674 | 0.715 | 0.242 |
| WE | 0.661 | 0.698 | 0.379 |
| Mohammad et al. | 0.691 | 0.750 | 0.430 |

Target mentioned is whether an opinion towards the target is expressed or not.

The results achieved in this implementation do not quite reach the performance
achieved in the original paper. This may be due to slightly different
tokenisation methods, or the original paper using more carefully tuned
regularisation parameters for the linear kernel. I find that the pattern of
results matches the results from the paper, however, with linear kernel + target
features achieving the best average F1 results, and that results are generally
much better when there is an opinion expressed towards the target in the tweet
than when there is not.

Trained word embeddings works marginally better than just using ngram features,
with an average F1 of 0.661 vs 0.657. I did not experiment with using pretrained
word embeddings or contextual word embeddings from a larger dataset, which may
help to improve performance even more.

## Usage
Requirements are in `requirements.txt`.
Install with `pip install -r requirements.txt`

```bash
usage: stance_detection.py [-h] [--classifier {svm,ft}] [--target] [--we]
                           [--epochs EPOCHS] [--wordNgrams minn maxn]
                           [--charNgrams minn maxn]
                           train test

positional arguments:
  train                 path to training data
  test                  path to test data

optional arguments:
  -h, --help            show this help message and exit
  --classifier {svm,ft}
                        choose classifier to use, either Linear Kernel SVM or
                        FastText
  --target              include target of interest presence/absence as a
                        feature. Default is False
  --we                  include FastText word embeddings as additional SVM
                        features. Default is False
  --epochs EPOCHS       number of epochs for FastText training
  --wordNgrams minn maxn
                        word n-gram range. Default is [1,3]
  --charNgrams minn maxn
                        character n-gram range. Default is [2,5]
```

For example,

```python 
$ stance_detection.py data/trainingdata-all-annotations.txt data/testdata-all-annotations.txt
Average F1: 0.6565412715893524
```

## References

[1] Mohammad, S.M. (2017). Challenges in Sentiment Analysis. In: Cambria, E., Das, D., Bandyopadhyay, S., Feraco, A. (eds) A Practical Guide to Sentiment Analysis. Socio-Affective Computing, vol 5. Springer, Cham. https://doi.org/10.1007/978-3-319-55394-8_4 \
[2] http://www.saifmohammad.com/WebPages/StanceDataset.htm \
[3] Saif M. Mohammad, Parinaz Sobhani, and Svetlana Kiritchenko. 2017. Stance and Sentiment in Tweets. ACM Trans. Internet Technol. 17, 3, Article 26 (August 2017), 23 pages. https://doi.org/10.1145/3003433
