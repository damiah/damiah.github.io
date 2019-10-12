### Motivation

To understand at a high level how 'embeddings' work, how to apply deep learning methods from the ground up, and stumble to the finish line with something useful and interesting.

### Inspiration

After following Jeremy Howard's fantastic fast.ai collaborative filtering lessons, I wanted to produce a recommender using PyTorch (what the fast.ai library is built atop of). I thought it would be cool to see if I could build a low-fi music recommender.

### Final output(s)
Our model learns weights for each album in vector space in such a way that it optimises our loss (error) during classification. Below is how each album looks after dimensionality reduction using PCA.

![Embeddings](https://i.imgur.com/ugxb7Yt.jpg)


We can see that albums that are close in 'sound' are closer in space. Rap albums The Eminem Show and Get Rich or Die Tryin are bunched together and shifted to the left of the visual, and albums Lateralus, St. Anger, and Aenima are bunched in a similar way. This is all without knowing anything about the content of the album - just who purchased them.


![Similar albums](https://i.imgur.com/XPDxbMJ.png)

The albums identified as closest in space can be deemed as similar, and can then be used as recommendations.

### Technical details

#### The dataset


#### The final model

The beauty of this model is it's simplicity. We separate users (customers) and items (albums) into two matrices of their cardinality in length (seen below as n_users and n_items).
Our output (1 or 0) is produced through the matrix multiplication of these two matrices.
The 'things that we are learning' is each customer and albums weight's in n_dim space.
n_dim is the dimensionality that we force the model to learn. the could be any number equal to or larger than 1.


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
#### Training the model

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

#### Lessons learned


