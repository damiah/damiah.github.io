### Motivation

To understand at a high level how 'embeddings' work, how to apply deep learning methods from the ground up, and stumble to the finish line with something useful and interesting.

### Inspiration

After following Jeremy Howard's fantastic fast.ai collaborative filtering lessons, I wanted to produce a recommender using PyTorch (what the fast.ai library is built atop of). I thought it would be cool to see if I could build a low-fi music recommender.

### Final output(s)
Our model learns weights for each album in vector space in such a way that it optimises our loss (error) during classification. Below is how each album looks after dimensionality reduction using PCA.

![Embeddings](https://i.imgur.com/ugxb7Yt.jpg)


We can see that albums that are close in 'sound' are closer. Rap albums The Eminem Show and Get Rich or Die Tryin are bunched together and shifted to the left of the visual, and albums Lateralus, St. Anger, and Aenima are bunched in a similar way. This is all without knowing anything about the content of the album - just who purchased them.


![Similar albums](https://i.imgur.com/XPDxbMJ.png)



```Python
python code here
```
