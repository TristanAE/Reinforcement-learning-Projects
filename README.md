# Reinforcement-learning-Projects
Jeux et simulations apprentissage par renforcement

Nous utiliserons la bibliothèque Stable-baselines3 pour effectuer un apprentissage par renforcement des différentes simulations.
Pour simplifier et gagner du temps, nous n’utiliserons pas la bibliothèque Optuna, bien qu’optimiser les hyperparamètres de nos algorithmes serait beaucoup plus efficace.

# I-Gym classic control CartPole :

 __Objectif__: Un poteau doit tenir en équilibre sur un chariot mobile

__Action space__ : discrete(2) pour aller à gauche ou aller à droite avec le chariot

__Observation space__ : tableau contenant des informations comme la position du chariot, sa vélocité…

__Reward__ : +1 à chaque frame passé sans perdre 

__Mise en place entrainement:__

-	Ouverture de la simulation : `env = gym.make("CartPole-v1")`

-	Vectorisation factice de l’environnement :  `env = DummyVecEnv([lambda: env])`

-	Choix de l’algorithme : Action space discrete(2) donc on peut utiliser PPO. Observation space tableau donc policy model 'MlpPolicy'.

`model = PPO('MlpPolicy', env, verbose = 1,tensorboard_log=log_path)`

-	Apprentissage sur certain nombre de frames : `model.learn(total_timesteps=20000)`

-	Sauvegarde du modèle : `model.save(PPO_path)`

__Prédiction des mouvements en boucle :__

-	Chargement du modèle entrainé : `model = PPO.load(PPO_path, env=env)`

-	Prédiction action à faire en fonction du tableau d’observation : `action, states = model.predict(obs)`

-	Réalisation de l’action : `obs, rewards, done, info = env.step(action)`

-	Affichage : `env.render()`


    ![ezgif com-gif-maker (8)](https://user-images.githubusercontent.com/92324336/185174735-725cb864-4a97-4905-99cc-9c7bcfd0e44f.gif)

# II- Gym Box2D CarRacing :

__Objectif__ : Une voiture doit rester sur le circuit en roulant

__Action space__ : box() pour contrôler vitesse, degré des tournants, freinage

__Observation space__ : Image de dimension (96,96,3) pour la longueur, largeur et couleur RGB

__Reward__ : voir sur le site 

__Mise en place entrainement :__

-	Ouverture de la simulation : `env = gym.make("CarRacing-v0")`

-	Transformation noir et blanc de l’environnement pour supprimer 3 channels RGB et n’en conserver qu’1 pour optimiser apprentissage : 

 `env = GrayScaleObservation(env, keep_dim=True)`

-	Vectorisation de l’environnement : Stockage de 4 frames à conserver dans notre environnement maintenant vectorisé pour optimiser l’apprentissage.

`env = DummyVecEnv([lambda: env])`
 `env = VecFrameStack(env, 4, channels_order='last')`

-	Choix de l’algorithme : Action space box() donc on peut utiliser PPO. Observation space image donc policy model « CnnPolicy ».

`model = PPO('CnnPolicy', env, verbose = 1,tensorboard_log=log_path)`

-	Apprentissage sur un certain nombre de frames : `model.learn(total_timesteps=20000)`

__Prédiction des mouvements en boucle :__

-	Prédiction action à faire en fonction de l’image d’observation : `action, _states = model.predict(obs)`

-	Réalisation de l’action : `obs, rewards, done, info = env.step(action)`

![ezgif com-gif-maker (9)](https://user-images.githubusercontent.com/92324336/185174773-7e757eb5-2bef-45d0-8e84-d864eaa1b787.gif)


# III- Gym Atari Breakout :

__Objectif__ : Casser des briques avec un slider

__Action space__ : discrete(4) pour gauche, droite, tirer, ne rien faire

__Observation space__ : Image de dimension dépendant du mode 

__Reward__ : points des briques cassées

__Mise en place entrainement:__

-	Télécharger la rom souhaitée et l’installer

-	Ouverture de la simulation dont l’environnement est déjà optimisé avec un channel, une image plus petite et en indiquant nombre de frames stockées: 

`env = make_atari_env("Breakout-v0", n_envs=4, seed=0)`

-	Vectorisation de l’environnement : Stockage de 4 frames à conserver dans notre environnement maintenant vectorisé pour optimiser l’apprentissage.

`env = VecFrameStack(env, n_stack=4)`

-	Choix de l’algorithme : Action space discrete(4) donc on peut utiliser PPO ou A2C. Observation space image donc policy model « CnnPolicy ».

`model = A2C('CnnPolicy', env, verbose = 1,tensorboard_log=log_path)`

-	Apprentissage sur un certain nombre de frames : `model.learn(total_timesteps=100000)`

__Prédiction des mouvements en boucle :__

-	Prédiction action à faire en fonction de l’image d’observation : `action, _states = model.predict(obs)`

-	Réalisation de l’action : `obs, rewards, done, info = env.step(action)`
 
![ezgif com-gif-maker (10)](https://user-images.githubusercontent.com/92324336/185174812-4bec0bc8-10f5-46eb-919d-249a505c7478.gif)


# IV- Gym Retro MortalKombat :

__Objectif__ : Battre son adversaire

__Action space__ : multibinary(12)

__Observation space__ : Image de dimension que l’on choisira soit (100,145,1)

__Reward__ : Récompense que l’on choisira soit +15 si touche adversaire et -5 si touché

La bibliothèque Gym retro fournit un ensemble de jeux en trouvant la bonne ROM associée. Les jeux disposent de différentes informations telles que le score, la position du personnage, les pièces récoltées…ma is il nous faut parfois ajuster les récompenses.

Il nous faut donc recréer un environnement en utilisant la dépendance Env de Gym. Il nous faudra ainsi instancier chaque méthode dans une nouvelle classe, méthodes que nous utilisions de manière automatique avant. Le but est de mettre en place un système de récompenses en utilisant les informations fournis par l’émulation.

__Mise en place de l’environnement :__

-	Création d’un ficher .py pour la création de notre environnement

- Initialisation de la classe MortalKombat(), on indique :
  - Observation_space image donc box avec pixel plus bas et plus haut, et sa shape: `self.observation_space = Box(low=0, high=255, shape=(150, 214, 1), dtype=np.uint8)`
  - Action_space : `self.action_space = MultiBinary(12)`
  - Démarrage du jeu avec actions filtrées : `self.game = retro.make (game='MortalKombatII-Genesis',   use_restricted_actions = retro.Actions.FILTERED)`

- 1ère méthode, reset(self) :
  - On obtient la 1ère frame :  `obs = self.game.reset()`
  - On transforme l’image selon notre envie et on la retourne : `obs =  self.preprocess(obs)`

- 2ème méthode, preprocess(self, observation):
  - On récupère l’image et on la met en noir et blanc : `gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)`
  - On change la taille de l’image : `resize = cv2.resize(gray, (215, 150))`
  - On indique à nouveau que notre image a 3 dimensions avec 1 channel et on retourne : `channels = np.reshape(resize, (150, 215, 1))`

- 3ème méthode, step(self, action):
  - On effectue une action et on récupère informations : `obs, reward, done, info = self.game.step(action)`
  - On traite l’image :  `obs = self.preprocess(obs)`
  - On traite la récompense :    
      
        		`if healthEnemypreced>info["enemy_health"]:
          
            			h=15
               
        		if health>info["health"]:
          
            			h=-5
               
          health=info["health"]
          
        		healthEnemypreced=info["enemy_health"]`


- 4ème  méthode, render(self, , *args, **kwargs) :
  - On affiche : `self.game.render()`
- 5ème méthode, close(self) : 
   - On ferme tout : `self.game.close()`


  __Mise en place entrainement :__
  
-	Télécharger et installer la rom souhaitée

-	Vectorisation de l’environnement : 
`env = DummyVecEnv([lambda: env])`
`env = VecFrameStack(env, 4, channels_order='last')`

-	Choix de l’algorithme : Action space multibinary() donc on peut utiliser PPO. Observation space image donc policy model « CnnPolicy ».
`model = PPO('CnnPolicy', env, verbose = 1,tensorboard_log=log_path)`

-	Apprentissage sur un certain nombre de frames : `model.learn(total_timesteps=500000)`

__Prédiction des mouvements en boucle__

![ezgif com-gif-maker (11)](https://user-images.githubusercontent.com/92324336/185745961-65dbb946-8049-4168-bb84-0c6fd7bb2a39.gif)

# V- Création environnement gym : Piano Tiles

Le jeu n’étant pas dans un environnement gym déjà conçu, nous allons reconstruire une classe Env en mettant en place nous même les méthodes.

Nous utiliserons les bibliotèques mss pour les captures de jeu, pytesseract pour la lecture du score et pynput pour les actions souris.

__Objectif__ : Cliquer sur touches noires

__Action space__ : Discrete(5)

__Observation space__ : Image de dimension que l’on choisira soit (80,960,1)

__Reward__ : Récompense que l’on choisira soit +5 si touche noire ou +1 à chaque frame

__Mise en place de l’environnement :__

- Initialisation de la classe PianoTiles(), on indique :
  - Observation_space : `self.observation_space = Box(low=0, high=255, shape=(80, 960, 1), dtype=np.uint8)`
  - Action_space : `self.action_space = Discrete(5)`
  - Reward_range : `self.reward_range=(0,5)`
  - Les dictionnaires des differents endroits importants du jeu pour mss :
  
 `self.ScreenGamePos = {"top": 880, "left": 50, "width": 1800, "height": 150}`
  
 `self.ScreenScore={"top": 80, "left": 780, "width": 350, "height": 150}`

- 1ère méthode, reset(self) :
  - On attend que le Game Over s'affiche entierement :  `time.sleep(2)`
  - On clique sur les boutons pour relancer une partie : `mouse.click(pynput.mouse.Button.left, 1)`
  - On retoune la 1ère frame du jeu issu de preprocess()
  
- 2ème méthode, preprocess(self):
  - On récupère l’image par des captures d'écran : `raw = np.array(self.sct.grab(self.ScreenGamePos))`
  - On change en noir et blanc pour 1 channel :   `gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)`
  - On change la taille de l’image : `resize = cv2.resize(gray,  (960,80))`
  - On indique à nouveau que notre image a 3 dimensions avec 1 channel et on retourne : `channels = np.reshape(resize,  (80,960,1))`    
    
- 3ème méthode, step(self, action):
  - On effectue une action avec pynput en fonction de la valeur de l'action : 4 actions de clics selon la case et ne rien faire
  - On regarde si la partie est terminée :  `done, done_cap = self.get_done()`
  - On fournit la récompense : `reward= 1 + self.get_score()`
  - On récupère image de l'environnement : `observation = self.preprocess()`

- 4ème  méthode, get_done(self) :
  - On filtre la couleur du jeu pour ne voir que le rouge lorsqu'une case est incorrecte. On évite d'utiliser pytesseract 2 fois pour ne pas ralentir le programme : 
 
 `frame = cv2.inRange(frame, lo, hi)`
 
 `color = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]`
 
  `if len(color) > 0 :`
  
         c = max(color, key=cv2.contourArea)
            
         ((x1, y1), radius1) = cv2.minEnclosingCircle(c)
            
          if radius1 > 10:
            
              done=True
                
- 5ème méthode,  get_score(self)) : 
   - Le reward sera le score affiché. On filtre capture score pour ne voir que le chiffre rouge et on réduit image.
   - On utilise pytesseract : `string = pytesseract.image_to_string(resized, config='--psm 6')`
   - On tranforme si possible la chaine en entier et on le compare avec le chiffre lu précédemment pour savoir si le score a changé
   
  `if number!=self.numberpreced:`
   
            self.numberpreced = number
            
            return 4
        else:
        
            return 0

![ezgif com-gif-maker (13)](https://user-images.githubusercontent.com/92324336/187224102-776e5e74-7192-44ab-97f7-9fecf4b00e59.gif)

 

    
  
       
