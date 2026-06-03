# **Gaze Pre-Processing**

This repository provides Python for **preprocessing raw gaze data and detecting fixations** and and R scripts for subsequent analysis of the processed data. The preprocessing pipeline includes various steps to clean and process the gaze data, while the fixation detection algorithm identifies periods of stable gaze points, known as fixations. Most of the constants and window sizes are chosen for the context of my study and might need tuning depending on the context. R Scripts and Raw Gaze Data of participants (along with logs of their click, or viewing events) have been provided too, to encourage further analysis and transparency. The **Python Notebook** file **details the steps** one might use to also process their raw gaze data from normalized coordinates to list of list of fixations.

Code used in the ETRA'24 paper: 

## Shifting Focus with HCEye: Exploring the Dynamics of Visual Highlighting and Cognitive Load on User Attention and Saliency Prediction



If you use the scripts, **please cite**: 

> Anwesha Das, Zekun Wu, Iza Skrjanec, and Anna Maria Feit. 2024. Shifting Focus with HCEye: Exploring the Dynamics of Visual Highlighting and Cognitive Load on User Attention and Saliency Prediction. Proc. ACM Hum.-Comput. Interact. 8, ETRA, Article 236 (May 2024), 18 pages. https://doi.org/10.1145/3655610

> @article{10.1145/3655610,
author = {Das, Anwesha and Wu, Zekun and Skrjanec, Iza and Feit, Anna Maria},
title = {Shifting Focus with HCEye: Exploring the Dynamics of Visual Highlighting and Cognitive Load on User Attention and Saliency Prediction},
year = {2024},
issue_date = {May 2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {8},
number = {ETRA},
url = {https://doi.org/10.1145/3655610},
doi = {10.1145/3655610},
abstract = {Visual highlighting can guide user attention in complex interfaces. However, its effectiveness under limited attentional capacities is underexplored. This paper examines the joint impact of visual highlighting (permanent and dynamic) and dual-task-induced cognitive load on gaze behaviour. Our analysis, using eye-movement data from 27 participants viewing 150 unique webpages reveals that while participants' ability to attend to UI elements decreases with increasing cognitive load, dynamic adaptations (i.e., highlighting) remain attention-grabbing. The presence of these factors significantly alters what people attend to and thus what is salient. Accordingly, we show that state-of-the-art saliency models increase their performance when accounting for different cognitive loads. Our empirical insights, along with our openly available dataset, enhance our understanding of attentional processes in UIs under varying cognitive (and perceptual) loads and open the door for new models that can predict user attention while multitasking.},
journal = {Proc. ACM Hum.-Comput. Interact.},
month = {may},
articleno = {236},
numpages = {18},
keywords = {cognitive load, computer vision, eye tracking, saliency prediction, visual attention}
}

