# Komendy — ściągawka

Wszystkie punkty wejścia w repozytorium, ich argumenty i przykłady uruchomienia.
Komendy uruchamiamy z katalogu głównego repo (chyba że zaznaczono inaczej) przez `uv run`.

---

## 0. Przygotowanie środowiska

```bash
uv sync                      # tworzy .venv (Python 3.11) i instaluje zależności, w tym PyTorch CUDA
```

Wymagane pliki (nie są w repo):

| Co | Gdzie |
|---|---|
| Klucz OpenAI (tylko dla opcji `gpt-*` i `--score vqa`) | `.env` → `OPENAI_API_KEY=sk-...` |
| Model ClipCap (Conceptual Captions) | `function/clipcap.pt` |
| Checkpointy klasyfikatorów | `model/best_model_CelebA_erm.pth`, `model/best_model_CelebA_dro.pth`, `model/best_model_Waterbirds_erm.pth`, `model/best_model_Waterbirds_dro.pth` |
| Waterbirds | `data/cub/data/waterbird_complete95_forest2water2/` |
| CelebA (wyrównane) | `data/celeba/img_align_celeba/data/` |
| CelebA (surowe) | `data/celeba/img_celeba/data/` |
| Adnotacje CelebA | `data/celeba/list_attr_celeba.csv`, `data/celeba/list_eval_partition.csv` |

---

## 1. `data/convert_celeba_annotations.py` — konwersja adnotacji CelebA

Zamienia oficjalne pliki `.txt` na `.csv`, których oczekuje `data/celeba.py`. Nie ma argumentów.

Wejście: `data/celeba/list_attr_celeba.txt`, `data/celeba/list_eval_partition.txt`
Wyjście: `data/celeba/list_attr_celeba.csv`, `data/celeba/list_eval_partition.csv`

```bash
uv run python data/convert_celeba_annotations.py
```

Uruchom raz, przed pierwszym `b2t.py --dataset celeba`.

---

## 2. `b2t.py` — główny pipeline B2T

Kroki: generowanie opisów obrazów → klasyfikacja zbioru walidacyjnego → wyciąganie słów kluczowych z opisów źle sklasyfikowanych obrazów → liczenie score (CLIP albo VQA).

### Argumenty

| Argument | Wartości | Domyślnie | Opis |
|---|---|---|---|
| `--dataset` | `waterbird`, `celeba` | `waterbird` | Zbiór danych |
| `--model` | nazwa pliku z `model/` | `best_model_Waterbirds_erm.pth` | Checkpoint klasyfikatora |
| `--captioning_model` | `clipcap`, `multicap`, `gpt-4o`, `gpt-4o-mini` | `clipcap` | Model do generowania opisów. `multicap` = ClipCap, 10 opisów sklejonych w jeden |
| `--keyword_extraction_model` | `yake`, `gpt-4o`, `gpt-4o-mini` | `yake` | Ekstrakcja słów kluczowych |
| `--score` | `clip`, `vqa` | `clip` | `clip` = score z artykułu, `vqa` = GPT odpowiada, czy słowo kluczowe widać na obrazie |
| `--number_val_images` | liczba całkowita | brak (wszystkie) | Ogranicza zbiór walidacyjny do pierwszych N obrazów (szybciej / taniej) |
| `--no_extract_caption` | flaga | wyłączona | Pomija generowanie opisów i używa już zapisanych |
| `--celeba_variant` | `align`, `raw` | `align` | Zestaw obrazów CelebA. `raw` nie pasuje do checkpointów, więc wyniki nie są porównywalne z artykułem. Dla waterbird ignorowane |
| `--save_result` | dowolna wartość | `True` | Zapis wyników do CSV (patrz uwagi) |

### Pliki wyjściowe

| Plik | Zawartość |
|---|---|
| `data/cub/caption/*.txt`, `data/celeba/caption_<variant>/*.txt` | Opisy obrazów |
| `result/<dataset>_<model>.csv` | Predykcje klasyfikatora i opisy (cache) |
| `diff/<dataset>_<model>_<klasa>.csv` | Słowa kluczowe i CLIP score (`--score clip`) |
| `result/vqa_scores.csv` | Surowe statystyki VQA (`--score vqa`) |

### Przykłady

Waterbirds, ustawienia z artykułu (ClipCap + YAKE + CLIP score):
```bash
uv run python b2t.py --dataset waterbird --model best_model_Waterbirds_erm.pth
```

Waterbirds, model GroupDRO:
```bash
uv run python b2t.py --dataset waterbird --model best_model_Waterbirds_dro.pth
```

CelebA (obrazy wyrównane), ERM:
```bash
uv run python b2t.py --dataset celeba --model best_model_CelebA_erm.pth
```

CelebA, model GroupDRO:
```bash
uv run python b2t.py --dataset celeba --model best_model_CelebA_dro.pth
```

CelebA na surowych obrazach:
```bash
uv run python b2t.py --dataset celeba --model best_model_CelebA_erm.pth --celeba_variant raw
```

Ponowne uruchomienie bez generowania opisów od nowa (opisy już są na dysku):
```bash
uv run python b2t.py --dataset waterbird --model best_model_Waterbirds_erm.pth --no_extract_caption
```

Szybki test na 100 obrazach:
```bash
uv run python b2t.py --dataset waterbird --model best_model_Waterbirds_erm.pth --number_val_images 100
```

Wiele opisów ClipCap na obraz (multicap):
```bash
uv run python b2t.py --dataset waterbird --model best_model_Waterbirds_erm.pth --captioning_model multicap
```

Opisy i słowa kluczowe z GPT (wymaga `OPENAI_API_KEY`, kosztuje):
```bash
uv run python b2t.py --dataset waterbird --model best_model_Waterbirds_erm.pth --captioning_model gpt-4o-mini --keyword_extraction_model gpt-4o-mini --number_val_images 50
```

Score VQA zamiast CLIP (wymaga `OPENAI_API_KEY`, jedno zapytanie na obraz):
```bash
uv run python b2t.py --dataset waterbird --model best_model_Waterbirds_erm.pth --score vqa --number_val_images 50
```

Pełny wariant GPT (opisy + słowa kluczowe + VQA):
```bash
uv run python b2t.py --dataset waterbird --model best_model_Waterbirds_erm.pth --captioning_model gpt-4o-mini --keyword_extraction_model gpt-4o-mini --score vqa
```

Lista argumentów:
```bash
uv run python b2t.py --help
```

### Uwagi

- **Cache predykcji:** jeśli `result/<dataset>_<model>.csv` już istnieje, klasyfikacja jest pomijana i opisy są brane z tego pliku, a nie z nowo wygenerowanych `.txt`. Po zmianie `--captioning_model`, `--celeba_variant` albo `--number_val_images` usuń ten plik, inaczej zostaną użyte stare opisy.
- **`--save_result`** jest parsowany jako tekst, więc `--save_result False` nadal zapisuje wyniki (każdy niepusty napis jest prawdziwy). Nie da się tego wyłączyć z linii poleceń.
- **`gpt-4o` vs `gpt-4o-mini`:** obecnie w obu przypadkach używany jest `gpt-4o-mini` (to domyślna wartość w `function/gpt_captioning.py` i `function/gpt_keywords.py`, a nazwa modelu nie jest tam przekazywana).
- **`--score vqa`** zapisuje zawsze do `result/vqa_scores.csv`, nadpisując poprzedni plik. Wyniki w `result_vqa_*` były przenoszone ręcznie.
- **`--number_val_images` + `--score clip`:** jeśli w którejś klasie wszystkie obrazy zostaną sklasyfikowane poprawnie, liczenie score rzuci błąd.

---

## 3. `analysis.py` — tabele VQA score

Czyta `result_vqa_<run>/vqa_scores.csv`, liczy znormalizowany VQA score dla każdego słowa kluczowego, wypisuje tabele i zapisuje `vqa_scores_tab_<klasa>.csv`. Nie ma argumentów — konfiguracja jest na górze pliku:

```python
classname = ["landbird", "waterbird"]
run = "full_gpt"          # "full_gpt", "baseline", "multicap"
save_results = True
```

```bash
uv run python analysis.py
```

Jeśli pliki wyjściowe już istnieją, skrypt pyta `Do you want to overwrite the file [Y,N]:` — dlatego uruchamiaj go interaktywnie.

Typowy przebieg: `b2t.py --score vqa` → przenieś `result/vqa_scores.csv` do `result_vqa_<run>/` → ustaw `run` → `analysis.py`.

---

## 4. `debug.py` — test VQA na kilku obrazach

Skrypt do debugowania VQA score, tylko dla Waterbirds. Bierze po `n = 2` obrazy z każdej kombinacji klasa × poprawne/błędne i odpytuje GPT. Nie ma argumentów — ustaw zmienne na górze pliku:

```python
model = 'best_model_Waterbirds_erm.pth'
keyword_extraction_model = "gpt-4o-mini"     # albo "yake"
caption_dir = 'data/cub/caption_gpt-4o-mini/'  # katalog z gotowymi opisami
```

```bash
uv run python debug.py
```

Nie generuje opisów — muszą już istnieć w `caption_dir`. Zapisuje do `result_test/` i tworzy `diff_test/`. Wymaga `OPENAI_API_KEY`.

---

## 5. `b2t_debias/` — trenowanie klasyfikatorów bez biasu

Komendy uruchamiamy **z katalogu `b2t_debias/`**.

### 5a. `infer_group_label.py` — pseudo-etykiety grup przez CLIP

| Argument | Wartości | Domyślnie |
|---|---|---|
| `--dataset` | `celeba`, `waterbirds` | `celeba` |
| `--data_dir` | ścieżka | `/data` |
| `--save_path` | ścieżka `.pt` | `./pseudo_bias/celeba.pt` |

Oczekuje `<data_dir>/celeba/` albo `<data_dir>/waterbird_complete95_forest2water2/`.

```bash
cd b2t_debias
uv run python infer_group_label.py --dataset celeba --data_dir ../data --save_path pseudo_bias/celeba.pt
uv run python infer_group_label.py --dataset waterbirds --data_dir ../data/cub/data --save_path pseudo_bias/waterbirds.pt
```

Uwaga: `b2t_debias/README.md` podaje `--data-dir`, ale poprawny argument to `--data_dir`.

### 5b. `gdro/group_dro.py` — trening GroupDRO

| Argument | Domyślnie | Opis |
|---|---|---|
| `--name` | `temp` | Nazwa przebiegu, logi w `results/<dataset>/<name>/` |
| `--dataset` | `celeba` | `celeba` albo `cub` |
| `--data_root` | `/data` | Katalog ze zbiorem |
| `--image_size` | `224` | |
| `--target_attr` | `9` | Atrybut docelowy CelebA (9 = Blond_Hair) |
| `--bias_attr` | `20` | Atrybut biasu CelebA (20 = Male) |
| `--pseudo_bias` | brak | Plik `.pt` z 5a; bez niego używane są prawdziwe etykiety grup |
| `--num_classes` | `2` | |
| `--model` | `resnet50` | |
| `--pretrained` | `imagenet` | |
| `--optimizer` | `sgd` | |
| `--epochs` | `50` | |
| `--batch_size` / `--batch-size` | `64` | |
| `--num_workers` | `4` | |
| `--lr` / `--learning-rate` | `0.1` | |
| `--momentum` | `0.9` | |
| `--weight-decay` / `--wd` | `5e-4` | |
| `--print-freq` / `-p` | `10` | |
| `--seed` | `1` | |

Bezpośrednio:
```bash
cd b2t_debias
uv run python gdro/group_dro.py --name gdro_celeba_b2t_seed_0 --dataset celeba --data_root ../data --epochs 50 --lr 1e-5 --weight-decay 1e-1 --seed 0 --pseudo_bias pseudo_bias/celeba.pt
```

Przez gotowe skrypty (argumenty: `<ścieżka do zbioru> <seed>`; wymagają basha, np. Git Bash, i aktywnego `.venv`, bo wołają `python`, a nie `uv run`):
```bash
cd b2t_debias
bash gdro/scripts/run_dro_celeba.sh ../data 0          # prawdziwe etykiety grup
bash gdro/scripts/run_dro_celeba_b2t.sh ../data 0      # pseudo-etykiety B2T
bash gdro/scripts/run_dro_waterbirds.sh ../data/cub/data 0
bash gdro/scripts/run_dro_waterbirds_b2t.sh ../data/cub/data 0
```

Uwaga: w obu skryptach `run_dro_waterbirds*.sh` jest `--data_root DATA_ROOT` (bez `$`), więc podana ścieżka jest ignorowana — trzeba to poprawić na `$DATA_ROOT` albo wołać `group_dro.py` bezpośrednio.

---

## 6. `b2t_diffusion/fair_diffusion.py` — Stable Diffusion + SD score

Generuje obrazy Stable Diffusion 1.5 dla promptu, liczy SD score i debiasuje. Nie ma argumentów — prompt (`"a photo of a nurse."`), liczba obrazów (`100`), seed i urządzenie (`cuda:0`) ustawia się w pliku.

```bash
cd b2t_diffusion
uv run python fair_diffusion.py
```

Uwaga: skrypt zapisuje do `images/` i `fair_images/` (oba katalogi muszą istnieć) i przy wczytywaniu używa niezdefiniowanej zmiennej `save_directory` — w obecnej postaci wymaga poprawek przed uruchomieniem.
