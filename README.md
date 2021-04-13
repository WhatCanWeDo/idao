# International data analysis olimpiad (IDAO)
# Track 1 statement
![statement image](https://github.com/WhatCanWeDo/idao/blob/master/images/statement-image.png)
The presence of Dark Matter (DM) in the Universe is nowadays an established, yet still mysterious, paradigm: deciphering its essence is one of the most compelling tasks for fundamental physics today. The nature of DM represents one of the key issues to understanding our Universe. Diﬀerent models predict the existence of neutral particles with a mass of few GeV or higher that would ﬁll our Galaxy. They could interact with the nuclei present in ordinary matter producing highly ionizing nuclear recoils (NR) with a kinetic energy as small as few keV. These NR would travel for hundreds to thousands of microns in gas leaving a trail of ionized atoms and free electrons.
Several experiments have been studied to detect those trails and possibly to reconstruct their main properties as position, direction, total amount of produced ionization. CYGNO collaboration is developing a technology able to collect those electrons on a structure where they produce photons, so that to have the possibility of creating images representing the 2D projection of the 3D NR trajectories.
![statement image 2](https://github.com/WhatCanWeDo/idao/blob/master/images/statement-image2.png)
Low energy photons produced by natural radioactivity can ionize electrons from atoms and molecules in the detector, producing recoils that would represent an important and dangerous background to Dark Matter signals. Electron recoils (ER) will produce distinctive and diverse patterns of ionization charge and therefore diﬀerent images, given the diﬀerent way they deposit energy and interact with matter.
Thus, NR and ER can be eﬃciently identiﬁed and separated down to low kinetic energies by studying the properties of the particle tracks.
In all cases, while traveling in gas, ionised electrons will be subject to diﬀusion eﬀects that blur ﬁnal images. Moreover, optical sensors used to collect the photons have a small, yet not negligible, electronic noise that adds up to physical signals.
In this competition we propose you images taken from 12 classes (6 classes of ER and 6 of NR) obtained by simulating detector response to ER and NR events with diﬀerent energies.
The test asks to properly assign each image to the correct particle type (ER/NR) and reconstruct the particle angle.
![statement image 3](https://github.com/WhatCanWeDo/idao/blob/master/images/statement-image3.png)
![statement image 4](https://github.com/WhatCanWeDo/idao/blob/master/images/statement-image4.png)

For the Track 2 we had to submitted zip file with pretrained model, which would predict in runtime. \\
**Objective**
Your objective is to create a model that, given an event, outputs the energy of the particle.
In the data there are electrons, He nucleus and energies in the range 1, 3, 6, 10, 20, 30 keV - 12 classes in total. We split those classes into training classes and test classes, with 3, 10, 30 keV electrons and 1, 6, 20 keV He being training classes and the remaining six being the test classes.
![statement image 5](https://github.com/WhatCanWeDo/idao/blob/master/images/statement-image5.png)
**Quality metric: (AUC - MAE) * 1000**

Let's take a closer look at the data
[](https://github.com/WhatCanWeDo/idao/blob/master/images/data_pic.jpg)
This picture for simple CNN embeddings and KNN to the calibration data. 
Our solution you can find at track1 and track2 folders.
For track 1 we used EfficientNet B1, for track 2 MobileNet v3 due to the time limitation.

