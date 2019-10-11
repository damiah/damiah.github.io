## Learning Good Representations of Data 

## Building the Graph

### CS:GO Graphs


### Our Graph

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
