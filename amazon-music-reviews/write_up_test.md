### Motivation

To understand at a high level how 'embeddings' work, how to apply deep learning methods from the ground up, and stumble to the finish line with something useful and interesting.

### Inspiration

After following Jeremy Howard's fantastic fast.ai collaborative filtering lessons, I wanted to produce a recommender using PyTorch (what the fast.ai library is built atop of). I thought it would be cool to see if I could build a low-fi music recommender.

### Final output(s)
Our model learns weights for each album in vector space in such a way that it optimises our loss (error) during classification. Below is how each album looks after dimensionality reduction using PCA.



```Python
nodes = {player: idx for idx, player in enumerate(all_players)}

edges = []

for game in games:
    for t1_player in game['team_1']:
        for t2_player in game['team_2']:
            if t1_player['kd'] > t2_player['kd']:
                new_edge = {
                    "sender": nodes[t2_player['name']], 
                    "receiver": nodes[t1_player['name']]
                }
                edges.append(new_edge)
            elif t1_player['kd'] < t2_player['kd']:
                new_edge = {
                    "sender": nodes[t1_player['name']], 
                    "receiver": nodes[t2_player['name']]
                }
                edges.append(new_edge)
```
