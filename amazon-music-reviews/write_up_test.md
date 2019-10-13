### Motivation

To understand at a high level how 'embeddings' work, how to apply deep learning methods from the ground up, and stumble to the finish line with something useful and interesting.

### Inspiration

After following Jeremy Howard's fantastic fast.ai collaborative filtering lessons, I wanted to produce a recommender using PyTorch (what the fast.ai library is built atop of). I thought it would be cool to see if I could build a low-fi music recommender.

## Final output(s)
Our model learns weights for each album in vector space in such a way that it optimises our loss (error) during classification. Below is how each album looks after dimensionality reduction using PCA.

![Embeddings](https://i.imgur.com/ugxb7Yt.jpg)


We can see that albums that are close in 'sound' are closer in space. Rap albums The Eminem Show and Get Rich or Die Tryin are bunched together and shifted to the left of the visual, and albums Lateralus, St. Anger, and Aenima are bunched in a similar way. This is all without knowing anything about the content of the album - just who purchased them.


<p align="center"><img src="https://i.imgur.com/XPDxbMJ.png" /></p>

The albums identified as closest in space can be deemed as similar, and can then be used as recommendations.

## Technical details

### The dataset and approach

The [dataset](http://jmcauley.ucsd.edu/data/amazon/) used is a snapshot of music purchases from amazon.com.  
 
 
> This dataset contains product reviews and metadata from Amazon, including 142.8 million reviews spanning May 1996 - July 2014.
  

![dataset](https://i.imgur.com/QUiSDzg.png)

  
An important thing to note is that the vast majority of reviewers of albums rate them very highly, as expected they are likely already familar with the artist, thus the review rating itself is not terribly useful for our problem. A naive attempt early on to disregard this quirk led to the model tossing up silly album recommendations.
  
We instead treat this as a classification problem (binary); whether the customer purchased the album or not. In order to do this we have to generate negative samples (albums the customer didn't purchase) for each customer, in order for our model to learn important things about our albums and customers.

### The final model

The beauty of this model is it's simplicity. We separate users (customers) and items (albums) into two matrices of their cardinality in length (seen below as n_users and n_items). We then multiply the two matrices together and run each output from the resulting matrix through the [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) which squeezes the output between 0 and 1.
The 'things that we are learning' are each customer and albums weight's in n_dim space.
 
 ![model diagram](https://i.imgur.com/xK67cXB.png) 
 
The matrix multiplication between these two just essentially links the customers and albums together. The alternative to this is to have one very sparse matrix, but that is far more memory intensive, as we would have to hold information for every combination of customer and album.  
 
n_dim is the dimensionality that we force the model to learn in. the could be any number equal to or larger than 1. In this case we force the model to learn 40 dimensional vectors for every customer and album. This idea of 'learned embeddings/representations' can be difficult to grasp (see [Dmitriy Genzel's quora answer](https://www.quora.com/What-is-the-difference-between-an-embedding-and-the-hidden-layer-of-an-autoencoder)).
  
We haven't added any additional layers so we can't call this deep learning. But it's still cool.
  
  

```Python
class EmbeddingModel(Module):
    #initiate the weights and biases of user and product.
    #these need to be leared through backward pass
    def __init__(self, n_dims, n_users, n_items):
        super(EmbeddingModel, self).__init__()
        (self.u_weight, self.i_weight, self.u_bias, self.i_bias) = [get_embs(*o) for o in [
            (n_users, n_dims), #user weights
            (n_items, n_dims), #product weights
            (n_users,1), #user bias
            (n_items,1)]] #product bias
    def forward(self, users, items):
        matmul = self.u_weight(users)* self.i_weight(items)
        out = matmul.sum(1) + self.u_bias(users).squeeze() + self.i_bias(items).squeeze()
        #run output through a sigmoid
        return torch.sigmoid(out)
```
### Training the model

We train this model using mini-batch gradient descent of size 64 and we schedule [cyclical learning rates](https://arxiv.org/pdf/1506.01186.pdf).


```Python
df_new = music_dataset()
ds = DataLoader(df_new, batch_size=64, shuffle=True)

model = EmbeddingModel(n_dims=40, n_users=len(np.unique(df_new.u)),
           n_items=len(np.unique(df_new.p)))
lr=5e-3
loss_func = BCELoss()
max_epochs = 2
optimizer = Adam(model.parameters(),lr=lr,weight_decay=1e-5)
iterations_per_epoch = len(ds)
scheduler = CyclicLR(optimizer, cosine(t_max=iterations_per_epoch * 2, eta_min=lr/10))
optimizer.zero_grad()
loss_values = []
for epoch in range(max_epochs):
    # Training
    loss= 0.
    for local_index, local_batch in enumerate(ds, 0):
        loss= 0.
        
        #pass in the indices of the batch user and prod 
        output = model.forward(local_batch[0][:,0:1].squeeze(1).long(), local_batch[0][:,1:2].squeeze(1).long())
        
        #compare outputs of batch with n=64 to label and compute the loss
        labels = local_batch[1].float()

        #calculate the loss
        loss = loss_func(output, labels.squeeze(1))
        scheduler.step()
        
        #update the parameters using backpropogation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_values.append(loss.data.item())
        if (len(loss_values) % 100 == 0) | (len(loss_values) == 1):
            print(loss.data.item())  
```

## Lessons learned

- Initialising the weights at the beginning of training is more important than I had originally thought; giving the model a helping hand to find a 'good' weight space heps immensely. 

- Scheduling learning rates is a great way to train more efficiently. It almost has to be done.

- Learn when to quit. If a model doesn't significantly drop in loss over the first few mini-batches, it's a good sign that something's wrong.

- Efficient batch loading is very important - you can create a massive bottleneck at the start of the training if your data is taking too long to fetch. Sometimes it's not immediately clear that this could be a problem, at least at small scale, as the time saving might not seem significant.

For complete code on final model notebook [here](https://github.com/damiah/damiah.github.io/blob/master/amazon-music-reviews/final-model.ipynb), and rough workings notebook [here](https://github.com/damiah/damiah.github.io/blob/master/amazon-music-reviews/recommender-pytorch.ipynb)

