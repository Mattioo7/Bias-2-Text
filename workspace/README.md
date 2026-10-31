# workspace/

Notatki, podsumowania i skrypty pomocnicze powstające we współpracy z Claude Code.
Nie jest to część pipeline'u B2T — kod projektu żyje w b2t.py, function/, data/,
b2t_debias/ i b2t_diffusion/.

## Układ

- `summaries/` — samodzielne podsumowania do wklejenia na starcie nowej rozmowy.
  Każdy plik zawiera kontekst repo, środowiska i dotychczasowych ustaleń, więc nie
  trzeba dołączać niczego więcej. Na końcu jest sekcja `## Zadanie` do wypełnienia.
- `scripts/` — skrypty pomocnicze (analizy, konwersje, jednorazowe eksperymenty).

## summaries/

| plik | temat |
|---|---|
| `01-uruchomienie-b2t-debias.md` | stan kodu b2t_debias, naprawione błędy, komendy dla Waterbirds i CelebA |
| `02-nowy-dataset.md` | jak dołożyć własny zbiór (np. Colored MNIST): pliki do zmiany, przebieg, oczekiwane wyniki |
| `03-interpretacja-i-automatyzacja.md` | jak czytać słowa kluczowe z B2T, jak zautomatyzować i zweryfikować wnioski |
| `04-teoria-debiasing.md` | jak działa metoda: pseudo-etykiety CLIP + GroupDRO, alternatywy, czego oczekiwać |

Podsumowania opisują stan na 2026-10-31. Jeśli kod się zmieni, warto je zaktualizować.
