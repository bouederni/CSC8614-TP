- Nom : Bilal OUEDERNI

#### Commandes d'installation/activation d'environnement :
```bash
python -m venv venv
source venv/bin/activate
pip install -r TP3/requirements.txt
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

Seed utilisé : 3333

# Question 1

Oui. Dans le "Model Structure after LoRA", les couches originelles `nn.Linear` (`W_query`, `W_key`, `W_value`, `out_proj`, et celles du feed-forward) ont été remplacées par des `LinearWithLoRA`. Ça confirme que le wrapping LoRA a bien été appliqué.

# Question 2

- Paramètres entraînables : 1,735,304
- Paramètres totaux : 164,772,488
- Fraction entraînable : 1,05 %

Ça nous montre que seule une petite partie du modèle peut être entraînée (comme attendu avec LoRA).

# Question 3

On constate en effet une différence. Ici, on a 1.33 million de paramètres entraînables, et une fraction entraînable de 1.06.

- Le nombre total de paramètres a diminué (la tête de sortie ayant été remplacée par une `nn.Linear` plus petite avec 2 classes)
- Le nombre de paramètres entraînables a aussi diminué (les anciennes matrices LoRA de la head ont disparu)
- La fraction reste presque identique, car le coeur du modèle LoRA (attention et FFN) n'a pas changé.

# Question 4

On constate une forte baisse au début, puis des valeurs très faibles, hormis quelques pics isolés (ex. batch 110). Ces pics sont dûs à des batchs plus difficiles.

En accuracy finale, on obtient 91.44% dès la première époque. C'est raisonnable pour une tâche de classification binaire avec un modèle pré-entraîné + LoRA. Le modèle converge vite et apprend de manière efficace, et les fluctuations locales sont normales.

Voici un schéma du loss observé : 

![Q4.png](img/Q4.png)

# Question 5

L'accuracy sur le test set est plus élevée que l'accuracy train. Ça indique que la génération est bonne et pas de sur-apprentissage, et que le modèle n'a probablement pas totalement convergé sur le train (1 seule époque).

C'est raisonnable avec un test set plus petit/simple, un dataset plus difficile/bruité, et un modèle pré-entraîné avec LoRA qui généralise bien rapidement.