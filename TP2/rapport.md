- Nom : Bilal OUEDERNI

#### Commandes d'installation/activation d'environnement :
```bash
python -m venv venv
source venv/bin/activate
pip install -r TP2/requirements.txt
```

#### Versions (présentes dans requirements.txt)
```bash
torch==2.9.1
tiktoken==0.12.0
tqdm==4.67.1
pandas==2.3.3
matplotlib==3.10.8
tensorflow==2.20.0
jupyterlab==4.5.1
```
Python 3.12.3

# Question 2
L'objet `setting` est un dictionnaire avec des clés à l'intérieur qui sont des entiers décrivant l'architecture : 
- `n_vocab` : taille du vocabulaire, 
- `n_ctx` : longueur de contexte maximale, 
- `n_embd` : dimension d'embedding, 
- `n_head` : nombre de têtes d'attention, 
- `n_layer` : nombre de couches

# Question 3
L'objet `params` est aussi un dictionnaire avec des clés à l'intérieur : 
- `wte` : matrice d'embeddings de tokens, 
- `wpe` : matrice d'embeddings de positions, 
- `blocks` : liste contenant les paramètres par couche, 
- `g` et `b` : paramètres de normalisation (gains et biais), 

# Question 4
Selon `gpt_utils.py` : 
```
cfg is a dictionary that requires the following keys: 
- vocab_size
- emb_dim
- context_length
- drop_rate
- n_layers
```
Il manquait ici un autre paramètre requis par la librairie GPT2, celui ci étant `n_heads`.
J'ai eu un mapping à réaliser sur les variables suivantes : 
```python
"vocab_size": settings["n_vocab"],
"emb_dim": settings["n_embd"],
"context_length": settings["n_ctx"],
"n_layers": settings["n_layer"],
"n_heads": settings["n_head"]
```
Après mapping et avoir ajouté n_heads, le modèle a été chargé et configuré avec succès : 
![Q4.png](img/Q4.png)

# Question 5.1

On utilise ici `df = df.sample(frac=1, random_state=123)` pour pouvoir reproduire les résultats en utilisant le même seed (`123`).

# Question 5.2.

On constate que la distribution n'est pas équilibrée entre les deux labels `ham` et `spam`.
```python
label_counts = df['Label'].value_counts()
print(label_counts)
```
rend : 
```
ham     4825
spam     747
```
Ça pourrait en effet causer des problèmes :
- problèmes de biais pour la classe majoritaire (ham)
- mauvaise généralisation
- métriques d'évaluation peu fiables
- overfitting

# Question 7

On peut obtenir la réponse en faisant une simple division : 
```python
print(len(train_dataset) // 16)
```
J'ai obtenu 278 batches.

La taille du dataset d'entraînement est de 4457.

# Question 8.3

On gèle les couches internes avec `param.requires_grad = False` pour les empêcher d'être mises à jour durant l'entraînement.

# Question 10
Globalement, je constate que le loss débute à 7 mais descend petit à petit pour atteindre en-dessous de 1.0 entre les batch 120 et 130 dans l'epoch 1. Ensuite, la loss semble conserver une valeur globalement entre 0.1 et 0.5. Ça signifie que l'entraînement du model se déroule correctement, car on ne remarque pas une hausse latente de la loss, ou bien une stabilité atteinte à une valeur non autant proche de 0.

(J'avais redémarré le kernel pour refaire une exécution propre du notebook, mais cette fois-ci, le loss a commencé à 2.88. C'est la seule différence avec l'exécution précédente)