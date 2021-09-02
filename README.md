###################################    Técnicas de búsqueda de arquitecturas neuronales para el   #####################################
###################################             diseño automático de redes convolucionales:       #####################################
################################### Aplicación a la clasificación de lesiones gastrointestinales  #####################################
                                 
---------- Alejandro Pinel Martínez        
---------- Supervisado por Pablo Mesejo Santiago

En este trabajo se comparan tres aproximaciones de NAS en un problema de clasificación binaria de pólipos gastrointestinales.

En el fichero main hay llamadas comentadas a cada experimento. Descomentar para ejecutar cada uno de ellos.

Requisitos:
- Tensorflow 2.4.0
- Keras 2.4.3
- Pytorch 1.7.1
Compilados con CUDA 10.1

Referencias:

ENAS:

H. Pham, M. Guan, B. Zoph, Q. Le and J. Dean, "Efficient Neural Architecture Search via Parameter Sharing", ICML, 2018.
Se ha utilizado la librería NNI: https://nni.readthedocs.io/en/stable/NAS/ENAS.html

Autokeras:

H. Jin, Q. Song and X. Hu, "Auto-Keras: An Efficient Neural Architecture Search System", the 25th ACM SIGKDD International Conference, 2019.
Se ha utilizado la librería Auto-Keras de los mismos autores: https://autokeras.com/

Auto CNN:

Y. Sun, B. Xue, M. Zhang, G. Yen and J. Lv, "Automatically Designing CNN Architectures Using the Genetic Algorithm for Image Classification", IEEE Transactions on Cybernetics, vol. 50, no. 9, pp. 3840-3854, 2020.
Se ha utilizado el repositorio proporcionado por los autores: https://github.com/Marius-Juston/AutoCNN






