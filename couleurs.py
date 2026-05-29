from PIL import Image
import numpy as np
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Patch


# chemin vers le dossier de mes images, c'est ici que tu modifies le chemin vers ton dossier où se trouvent tes images
dossier_path = "C:/Users/louartani/Documents/PROJETS\ATELIERS/foret_urbaine/traitement/data/ECOBOX_photos"

os.makedirs("resultats", exist_ok=True)

# ouvrir les images dans le dossier
images  = [
    f for f in os.listdir(dossier_path)
]

# lecture des images en RGB
for image in images:
    image_path = os.path.join(dossier_path, image)
    #conversion en rgb
    img_rgb = Image.open(image_path).convert("RGB")
    # dimensionnement
    img_rgb.thumbnail((400,400))
    arr = np.array(img_rgb)

    #reshape pour avoir un nuage de pixels
    pixels = arr.reshape(-1,3)

    # print("image :", image_path)
    # print('forme image :', arr.shape)
    # print('forme pixels : ', pixels.shape)
    # print("premier pixel : ",  pixels[0])
    # print('fin')

    # KMEANS

    kmeans = KMeans(
        n_clusters=5,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(pixels)

    colors = kmeans.cluster_centers_.astype(int)

    # print("couleurs rgb :")
    # print(colors)

    # calculer le pourcentage de chaque couleur dominante

    counts = np.bincount(labels)
    pourcentage = (counts/counts.sum()) *100

    # for color, pct in zip(colors, pourcentage):
    #   print(color, "→", round(pct, 2), "%")

    # ordre du plus présent (plus gros pourcentage) au moins (plus petit pourcentage)

    ordre = np.argsort(pourcentage)[::-1]

    colors = colors[ordre]
    pourcentage=pourcentage[ordre] # remettre les pourecntages dans l'ordre décroissant

    # for color, pct in zip(colors, pourcentage):
    #     print(color, "→", round(pct, 2), "%")

    ## faire le graphique
    
    fig, ax = plt.subplots(figsize=(15, 6))

    x = range(len(pourcentage))

    ax.bar(
        x,
        pourcentage,
        color=colors / 255
    )

    imagebox = OffsetImage(np.array(img_rgb), zoom=0.75)

    ab = AnnotationBbox(
        imagebox,
        (0.75, 0.75),  # position dans le graphique
        xycoords='axes fraction',
        frameon=True
    )

    labels = [
            f"{pct:.1f}%"
            for pct in pourcentage
        ]
    
    legend_elements = [
        Patch(
            facecolor= color/255,
            edgecolor='none',
            label=label
        )
        for color, label in zip(colors, labels)
    ]

    ax.legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5)
    )

    ax.add_artist(ab)
    ax.set_xticks([])
    ax.set_ylim(0, max(pourcentage) + 5)
    plt.tight_layout()

    nom = os.path.splitext(image)[0]
    plt.savefig(
        f"resultats/{nom}_palette.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()