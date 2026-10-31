# Kontekst: jak działa debiasing oparty na B2T (pseudo-etykiety CLIP + GroupDRO)

## Problem
Klasyfikator uczy się skrótu (fałszywej korelacji): w Waterbirds 95% ptaków wodnych jest na tle wody,
więc model rozpoznaje TŁO zamiast ptaka. Średnia poprawność ~97%, ale na nietypowych kombinacjach
(ptak lądowy przy wodzie) spada do ~60%. Miarą jest worst-group accuracy, nie średnia.

## Pojęcie grup
Każdy obraz ma klasę y (to, co przewidujemy) i atrybut biasu a (to, czego model używać nie powinien).
Grupa = 2*y + a → 4 grupy: dwie liczne (typowe) i dwie małe (nietypowe).
Problem praktyczny: atrybutu a zwykle nikt nie opisał. Tę lukę zamyka B2T.

## Etap 1 — pseudo-etykiety (b2t_debias/infer_group_label.py)
- CLIP zamienia obraz i tekst na wektory (embeddingi) w jednej przestrzeni; podobne znaczenie = podobny kierunek.
- Ze słów kluczowych (class_keywords_all) i ~80 szablonów zdań x class_templates powstają tysiące zdań
  na wartość atrybutu; ich embeddingi są normalizowane i uśredniane → 2 "wzorce tekstowe".
- Każde zdjęcie treningowe: embedding → iloczyn skalarny z oboma wzorcami → argmax → 0/1.
  To klasyfikacja zero-shot: nic się nie trenuje.
- Wynik: tensor 0/1, po jednej wartości na zdjęcie, w kolejności zbioru → pseudo_bias/<dataset>.pt.
- Skrypt wypisuje classification_report (precision/recall wykrywania grupy mniejszościowej) —
  tylko diagnostyka, możliwa wyłącznie tam, gdzie istnieją prawdziwe etykiety.

## Etap 2 — trening (b2t_debias/gdro/group_dro.py)
- Dane treningowe dostają 'a' z pliku .pt zamiast prawdziwego atrybutu.
- WeightedRandomSampler: waga 1/liczność grupy → każda grupa równie często w batchu.
- GroupDRO (klasa GroupEMA): loss liczony osobno per grupa; wagi grup w_g ← w_g * exp(0.01 * loss_g),
  normalizowane; optymalizowana suma ważona → nacisk na najgorszą grupę. Skrót przestaje się opłacać.
- Model: ResNet-50 pretrenowany na ImageNet, SGD, lr 1e-5; Waterbirds 300 epok / wd 1,
  CelebA 50 epok / wd 1e-1.
- Ewaluacja co epokę: poprawność per grupa, worst-group i średnia (TensorBoard w results/).
  Checkpoint zapisywany, gdy poprawi się worst-group accuracy na val.
- W repo val/test używają PRAWDZIWYCH etykiet grup. Przy własnym zbiorze trzeba walidować
  na pseudo-grupach, a ocenę oprzeć na małym, ręcznie opisanym zbiorze testowym.

## Warianty skryptów
run_dro_<dataset>.sh — prawdziwe etykiety grup (górna granica).
run_dro_<dataset>_b2t.sh — pseudo-etykiety z B2T (metoda bez ręcznego etykietowania).
Im bliżej siebie wyniki, tym lepiej działa metoda.

## Alternatywy wobec GroupDRO (te same pseudo-grupy)
- Reweighting/upsampling (WeightedRandomSampler bez DRO).
- DFR: zamrożony backbone + douczenie ostatniej warstwy na zbalansowanym podzbiorze (minuty zamiast epok).
- Wybór modelu / early stopping po worst-group accuracy zamiast średniej.
- Uzupełnianie danych: generowanie brakujących kombinacji (b2t_diffusion/), podmiana tła.
- Bez etykiet: JTT (upweighting błędów ERM).

## Czego oczekiwać
Średnia poprawność zwykle lekko SPADA, worst-group rośnie (np. 60% → 85%).
Brak poprawy zwykle oznacza nieczyste pseudo-etykiety, bias nieopisywalny słowami
albo zbyt małą grupę mniejszościową.

## Zadanie
[wpisz, co chcesz zrobić]
