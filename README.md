# Reinforcement-learning-Projects
Jeux et simulations apprentissage par renforcement

Nous utiliserons la bibliothèque Stable-baselines3 pour effectuer un apprentissage par renforcement des différentes simulations.
Pour simplifier et gagner du temps, nous n’utiliserons pas la bibliothèque Optuna, bien qu’optimiser les hyperparamètres de nos algorithmes serait beaucoup plus efficace.

# I-OpenAi classic control CartPole :

 __Objectif__: Un poteau doit tenir en équilibre sur un chariot mobile

__Action space__ : discrete(2) pour aller à gauche ou aller à droite avec le chariot

__Observation space__ : tableau contenant des informations comme la position du chariot, sa vélocité…

__Reward__ : +1 à chaque frame passé sans perdre 

__Mise en place entrainement:__

•	Ouverture de la simulation : 

      env = gym.make("CartPole-v1")

•	Vectorisation factice de l’environnement :  

     env = DummyVecEnv([lambda: env])

•	Choix de l’algorithme : Action space discrete(2) donc on peut utiliser PPO. Observation space tableau donc policy model 'MlpPolicy'.

    model = PPO('MlpPolicy', env, verbose = 1,tensorboard_log=log_path)

•	Apprentissage sur certain nombre de frames : 

    model.learn(total_timesteps=20000)

•	Sauvegarde du modèle : 
    model.save(PPO_path)

__Prédiction des mouvements en boucle :__

•	Chargement du modèle entrainé : 

    model = PPO.load(PPO_path, env=env)

•	Prédiction action à faire en fonction du tableau d’observation : 

    action, _states = model.predict(obs)

•	Réalisation de l’action : 

    obs, rewards, done, info = env.step(action)

•	Affichage : 

    env.render()

 
