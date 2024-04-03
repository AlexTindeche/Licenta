# To do:

- [ ] Automatic identification of crosswalks and automatically detect crossing lines for pedestrians and vehicles  
- [ ] Automatically detect the traffic light corresponding to a crosswalk  
- [ ] Posture identification of pedestrians  
- [X] Graph generation of pedestrian crossings over time  
- [ ] De rulat pe video-urile date de Bogdan Alexe si vazut cum merge
    - [ ] Daca merge bine de facut o clusterizare a traiectoriilor masinilor si a pietonilor si vizualizarea lor
    - [ ] De identificat anomanlii care ies din aceste clustere
- [X] Definirea unor zone ale traficului in intersectie si identificarea si numararea vehiculelor care trec dintr-o zona in alta

# Issues:
1. [ ] Identificarea culorii stopului nu functioneaza foarte bine
2. [ ] Atunci cand un om trece pe sub un stalp si yolo pierde complet predictia, cand trece de stalp si se vede din nou, bytetrack ii da un nou id
3. [ ] Atunci cand am rulat pe Coldwater videos am observat ca are probleme destul de mari pentru masinile care trec prin spatele unui stalp, partial sau complet obturate


# Tried, not working

- Am incercat pair-uiesc ByteTracker cu FairMOT pentru a implementa si un Re-ID dar nu am reusit sa fac asta. Momentan ByteTrack functioneaza doar pe baza unui Kalman Filter si miscarea subiectilor, facand un IoU intre predictia data de Kalman Filter si pozitia bboxului din frame

# Notes

1. Line counter functioneaza asa: pentru fiecare detectie, la fiecare frame, verifica daca detectia a trecut cu toate cele 4 puncte peste linie. Detectia are un state (stanga/dreapta vectorului selectat) iar acest state isi modifica valoarea abia dupa ce cele 4 puncte ale bboxului au trecut de vector. Problema apare cand un vehicul este prea mare si detectia dispare inainte sa fi trecut cu toate punctele peste vector.

2. ByteTracker functioneaza asa:
    - Imparte detectiile in 2 categorii in functie de un treshold dat de noi: high and low
    - Prezice o noua locatie pentru track-urile din frame-ul anterior folosind un filtru Kalman (motion based)
    - Pentru detectiile high asigneaza tracklets in functie de cele din frame-ul anterior cu o functie de similaritate 1 (IoU sau Re-ID)
    - Pentru cele low acelasi lucru cu o functie de similaritate 2 (doar IoU)
    - Detectiile care nu au reusit sa fie asignate catre un tracklet sunt sterse dupa un nr de frame-uri dat de noi

# Next up

1. Identificare mai buna a claselor
2. Identificarea mai buna pentru obiecte obturate
3. Tracking mai bun (maybe Re-ID)
4. Clusterizarea directiilor