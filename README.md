# 🚇 Hybrid Graph State-Space Models in Reinforcement Learning

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Framework: Gymnasium](https://img.shields.io/badge/Framework-Gymnasium-blue.svg)](https://gymnasium.farama.org/)

Oficjalne repozytorium dla projektu badawczego: **"Hybrid Graph State-Space Models in Reinforcement Learning: Attention as a Stabilizer for Dynamic Metro Network Control"**.

Projekt bada zachowanie różnych architektur encdoderów sieci grafowych (GNN, Graph Transformers, Graph Mamba, Graph Jamba) w środowisku uczenia ze wzmocnieniem (RL), gdzie topologia grafu dynamicznie rośnie w czasie.

##  Środowisko Symulacyjne: Dynamiczna Sieć Metra

Stworzyliśmy dedykowane środowisko zgodne ze standardem **Gymnasium**. Agent RL musi autonomicznie zarządzać rosnącą siecią metra, łącząc nowe stacje (reprezentowane przez figury geometryczne) i zarządzając flotą pociągów, aby zoptymalizować przepływ pasażerów.

![Symulator Metra](./SS_gra.png)
*Wizualizacja dynamicznego środowiska symulacji sieci metra. W miarę upływu czasu pojawiają się nowe stacje, wymagające od agenta adaptacji topologii w czasie rzeczywistym.*

##  Główne Wnioski z Badań

Autonomiczne sterowanie w środowisku, gdzie wymiary stanu stale się zmieniają, to duże wyzwanie. Przetestowaliśmy cztery podejścia:


##  Architektura Graph Jamba

Nasz model wykorzystuje trójścieżkowe przetwarzanie równoległe:
* **Gałąź identyczności (GNN):** Ekstrakcja cech lokalnych.
* **Gałąź Mamba:** Liniowe modelowanie sekwencji długoterminowych.
* **Gałąź Attention:** Stabilizacja dryfu stanu ukrytego (wykorzystujemy warianty *Standard Self-Attention* oraz *Linear Attention*).
