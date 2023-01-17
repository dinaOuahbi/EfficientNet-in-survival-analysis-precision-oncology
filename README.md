# EfficientNet-in-survival-analysis-precision-oncology


famille EfficientNet [b0-b7] : la résolution du model augmente de 0 à 7     (Tan et le, 2019)

Dépendance : Version Tensorflow : tf2.3

Efficientnet à été former sur imagenet, on peut donc inclure les poids de image net 

Valable pour un apprentissage par transfert 

Contrôle aussi la force de régularisation (0,2 par défaut)

Les EfficientNet surpassent de manière significative d'autres ConvNets



Un Rappel sur l’apprentissage par transfert
    Instanciation d’un modèle de base et chargement des poids préformés. 
    Gélation de tous les calques du modèle de base. 
    Création d’un nouveau modèle au-dessus de la sortie d'une (ou plusieurs) couches du modèle de base. 
    Entraînement du nouveau modèle sur un nouvel ensemble de données. 

    Compilation du modèle après n’importe quel changement de celui-ci 

Le Batch Size est le nombre d’images utilisé dans chaque cycle d’ entraînement (forward + back propagation )

Si on divise la taille du Train set par le Batch Size, nous obtenons le nombre de cycles d’entraînement pour une Epoch.

