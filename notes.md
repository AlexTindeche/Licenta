# To do:

- [ ] Automatic identification of crosswalks and automatically detect crossing lines for pedestrians and vehicles  
- [ ] Automatically detect the traffic light corresponding to a crosswalk  
- [ ] Posture identification of pedestrians  
- [ ] Graph generation of pedestrian crossings over time  


# Issues:
1. [X] Line counter nu functioneaza foarte bine pentru ca trebuie sa vada cum trec peste linie toate cele 4 colturi ale bounding boxului. 
    Trebuie gasit un algoritm mai eficient



1. Line counter functioneaza asa: pentru fiecare detectie, la fiecare frame, verifica daca detectia a trecut cu toate cele 4 puncte peste linie. Detectia are un state (stanga/dreapta vectorului selectat) iar acest state isi modifica valoarea abia dupa ce cele 4 puncte ale bboxului au trecut de vector. Problema apare cand un vehicul este prea mare si detectia dispare inainte sa fi trecut cu toate punctele peste vector.