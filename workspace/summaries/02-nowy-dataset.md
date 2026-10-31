# Kontekst: dodanie nowego zbioru danych do pipeline'u B2T + debiasing

## Projekt
- Repo: C:\Users\mateu\PycharmProjects\Bias-2-Text (Windows 11, uv, Python 3.11, torch 2.14+cu130, CUDA OK).
- B2T (bias-to-text): b2t.py wykrywa bias klasyfikatora jako słowa kluczowe;
  b2t_debias/ trenuje model odporny na ten bias (pseudo-etykiety CLIP + GroupDRO).
- Oba obsługują dziś tylko 'waterbird'/'waterbirds' i 'celeba'.
- Komendy uruchamia użytkownik samodzielnie.

## Co trzeba przygotować
1. Zdjęcia RGB (dla CLIP/ResNet skalowane do 224x224).
2. metadata.csv w formacie Waterbirds: kolumny img_filename, y (0/1), split (0=train, 1=val, 2=test).
   Kolumny 'place' (prawdziwy atrybut biasu) z założenia BRAK.
3. Wytrenowany zwykły klasyfikator (ERM) na tym zbiorze — B2T analizuje jego błędy.
4. Zbiór testowy bez biasu (np. losowe kolory w Colored MNIST) jako miernik.

## Pliki do zmiany
| plik | zmiana |
|---|---|
| b2t.py (load_dataset, all_datasets) | gałąź dla nowego zbioru: klasa danych, nazwy klas, katalog zdjęć |
| b2t_debias/<nazwa>_templates.py | kopia waterbirds_templates.py; templates zostawić, podmienić class_templates i class_keywords_all (słowa z B2T) |
| b2t_debias/data/<nazwa>.py | klasa zwracająca x, (y, y_group, y_spurious), idx; bez prawdziwego biasu → atrapy (zera) |
| b2t_debias/infer_group_label.py | gałąź dla zbioru + wpis w DEFAULT_DATA_DIRS; WYŁĄCZYĆ classification_report (brak prawdziwych etykiet); etykietować też split='val' (osobny plik .pt) |
| b2t_debias/gdro/<nazwa>_dataset.py | kopia cub_dataset.py; zwraca dict {'x','y','a',...}, 'a' z pliku .pt |
| b2t_debias/gdro/data_loader.py | prepare_<nazwa> + gałąź w prepare_data; val ma dostać pseudo-etykiety (dziś na sztywno pseudo_bias=None) |
| b2t_debias/gdro/arguments.py | nazwa zbioru + wpis w DEFAULT_DATA_ROOTS |
| b2t_debias/gdro/scripts/ | kopia *_b2t.sh z hiperparametrami |
| b2t_debias/gdro/group_dro.py | tylko jeśli grup jest więcej niż 4: num_groups i wzór grupa = 2*y + a |

## Przebieg
1. Trening zwykłego modelu (ERM) → wysoka średnia, słaby wynik na zbiorze bez biasu.
2. b2t.py → słowa kluczowe per klasa (Score, Acc., Bias S/M).
3. Interpretacja (jedyny krok dla człowieka): czy słowa opisują obiekt, czy coś innego;
   jaki mają wspólny mianownik; czy jest symetria (błędy klasy A mają cechy typowe dla klasy B).
4. Szybka weryfikacja hipotezy przez interwencję (np. dla koloru: test na obrazach w skali szarości).
5. Słowa → class_keywords_all → infer_group_label.py → pseudo-etykiety dla train i val.
6. Kontrola jakości ręcznie: liczności 4 grup + podgląd 30–50 zdjęć z grup mniejszościowych.
7. GroupDRO; wybór epoki po worst-group accuracy na pseudo-grupach walidacyjnych.
8. Ocena: mały, ręcznie opisany zbiór testowy (50–100 zdjęć na grupę) + ponowne B2T na nowym modelu.

## Uwagi dla Colored MNIST
- Bias = kolor skorelowany z klasą. Największe ryzyko: domyślny captioner (ClipCap, Conceptual Captions)
  generuje bełkot dla cyfr — użyć ścieżki GPT (function/gpt_captioning.py) lub trybu --score vqa.
- CLIP dobrze rozpoznaje kolory, więc etap etykietowania powinien działać.
- 10 klas x 10 kolorów = 100 grup; prościej użyć wariantu binarnego (cyfry 0-4 vs 5-9, 2 kolory)
  albo atrybutu "kolor typowy dla klasy / nietypowy".
- Oczekiwane wyniki: ERM ~99% na typowych kolorach i 20-40% (czasem <10%) na losowych;
  po debiasingu zwykle 80-95% w najgorszej grupie.

## Tańsze alternatywy
- Tylko WeightedRandomSampler z wagą 1/liczność grupy (kilkanaście linii, duża część zysku).
- DFR: zamrożony backbone + douczenie ostatniej warstwy na zbalansowanym podzbiorze (minuty).
- JTT jako punkt odniesienia bez słów kluczowych: trenujesz ERM, a błędnie sklasyfikowane przykłady
  dostają większą wagę w drugim treningu.

## Zadanie
[wpisz, co chcesz zrobić]
