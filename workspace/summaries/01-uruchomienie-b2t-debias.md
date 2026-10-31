# Kontekst: uruchomienie b2t_debias w projekcie Bias-2-Text

## Projekt
- Repo: C:\Users\mateu\PycharmProjects\Bias-2-Text (Windows 11, PowerShell + Git Bash, git branch master)
- Implementacja artykułu Bias-to-Text (B2T). Katalog b2t_debias/ zawiera debiasing klasyfikatorów.
- Środowisko przez uv: Python 3.11.15, torch 2.14.0+cu130, torchvision 0.29, pandas 3.0.5,
  tensorboard, scikit-learn. CUDA działa (torch.cuda.is_available() == True).
- Komendy uruchamia użytkownik samodzielnie (wkleja z `!` w Claude Code). Asystent ich nie odpala.

## Układ danych (lokalny, inny niż w oryginalnym repo)
- Waterbirds: data/cub/data/waterbird_complete95_forest2water2/ (+ metadata.csv)
- CelebA: data/celeba/{list_attr_celeba.csv, list_eval_partition.csv, img_align_celeba/data/*.jpg}
  Pliki CSV powstały z .txt skryptem data/convert_celeba_annotations.py (separator: przecinek).
  Brakuje identity_CelebA.txt, list_bbox_celeba.txt, list_landmarks_align_celeba.txt.
- Checkpointy ERM/DRO: model/*.pth

## Jak działa pipeline (2 etapy)
1. b2t_debias/infer_group_label.py — CLIP RN50 zero-shot przypisuje każdemu zdjęciu treningowemu
   atrybut biasu (0/1) na podstawie list słów w waterbirds_templates.py / celeba_templates.py
   (class_keywords_all). Zapisuje tensor 0/1 do pseudo_bias/<dataset>.pt. Wypisuje classification_report
   względem prawdziwych etykiet (tylko diagnostyka).
2. b2t_debias/gdro/group_dro.py — trening ResNet-50 (pretrained ImageNet) z GroupDRO.
   Grupy = 2*y + a (4 grupy). WeightedRandomSampler wyrównuje grupy, GroupEMA zwiększa wagę
   najgorszej grupy. Model wybierany po worst-group accuracy na val. Logi: results/<dataset>/<name>/.
   UWAGA: val i test zawsze używają PRAWDZIWYCH etykiet grup (pseudo_bias=None).
   Skrypty gdro/scripts/run_dro_*.sh; wersja _b2t.sh używa pseudo-etykiet, wersja bez — prawdziwych
   (porównanie: górna granica vs metoda bez ręcznych etykiet).

## Naprawione błędy (zmiany NIEZACOMMITOWANE, kod nie był uruchamiany — tylko py_compile)
- b2t_debias/gdro/data_loader.py: ścieżka '/data/waterbird_complete95_forest2water2' → os.path.join(args.data_root, ...)
- gdro/scripts/run_dro_waterbirds*.sh: '--data_root DATA_ROOT' → '$DATA_ROOT'
- b2t_debias/data/celeba.py: usunięty delim_whitespace (nie istnieje w pandas 3), czytanie CSV po przecinku,
  .values.copy() (pandas 3 zwraca widok read-only), ścieżka zdjęć img_align_celeba/data/
- gdro/arguments.py: dodany brakujący --val_frac (default 1.0); --data_root default=None + DEFAULT_DATA_ROOTS
  (cub → data/cub/data, celeba → data) liczone względem pliku
- gdro/celeba_dataset.py: przepisany bez torchvision.datasets.CelebA (wymagało brakujących .txt, MD5 i gdown);
  czyta lokalne CSV; zachowane numery atrybutów (9 = Blond_Hair, 20 = Male)
- gdro/group_dro.py: kod z poziomu modułu przeniesiony do main() (Windows spawn re-importuje moduł
  w workerach DataLoadera); train() dostaje group_weight_ema jako parametr
- infer_group_label.py: --data_dir default=None + DEFAULT_DATA_DIRS; --save_path default=None
  → pseudo_bias/<dataset>.pt obok skryptu
- .gitattributes (nowy): '*.sh text eol=lf' (core.autocrlf=true psułoby skrypty basha)

## Znane, nienaprawione ryzyka
- Pierwsze uruchomienie wymaga internetu: wagi CLIP RN50 (~/.cache/clip, jest tam tylko ViT-B-32.pt)
  i ResNet-50 ImageNet.
- Nie zweryfikowano, czy torch.jit.load wczyta CLIP RN50 pod torch 2.14.
- Skrypty .sh nadal wymagają podania ścieżki jako $1 (domyślna ścieżka działa tylko przy
  bezpośrednim wywołaniu group_dro.py).
- W pliku na dysku opcja nazywa się --data_dir (podkreślnik); wcześniejsza zmiana na --data-dir
  została cofnięta.

## Komendy (z katalogu repo)
Waterbirds:
  ! cd b2t_debias && uv run python infer_group_label.py --dataset waterbirds
  ! cd b2t_debias && uv run bash gdro/scripts/run_dro_waterbirds_b2t.sh ../data/cub/data 0
CelebA:
  ! cd b2t_debias && uv run python infer_group_label.py --dataset celeba
  ! cd b2t_debias && uv run bash gdro/scripts/run_dro_celeba_b2t.sh ../data 0
Trening Waterbirds: 300 epok (długo). CelebA: 50 epok.
W razie problemów z workerami na Windows: --num_workers 0.

## Zadanie
[wpisz, co chcesz zrobić]
