# Kontekst: interpretacja słów kluczowych z B2T i automatyzacja tego kroku

## Projekt
- Repo: C:\Users\mateu\PycharmProjects\Bias-2-Text. B2T zamienia bias klasyfikatora na słowa.
- b2t.py: klasyfikuje zbiór walidacyjny → generuje opisy → wyciąga słowa z opisów BŁĘDNIE
  sklasyfikowanych zdjęć (per klasa) → punktuje je. Wyniki w outputs/ oraz old_results_backup/diff/*.csv.
- Kolumny wyniku (function/print_similarity.py):
  - Score: podobieństwo CLIP słowa do zdjęć błędnych MINUS do poprawnych (wysokie = słowo towarzyszy błędom)
  - Acc.: poprawność modelu na zdjęciach tej klasy, których opis zawiera to słowo (porównaj z ogólną)
  - Bias: 'S' = dodatni Score tylko w tej klasie (kandydat na fałszywą korelację),
          'M' = dodatni w obu klasach (raczej trudne zdjęcia), '' = Score ujemny
- Przykład (Waterbirds, klasa landbird): ocean 3.20 / 0.76 / S; beach 2.58; boat 2.03 / 0.57;
  klasa waterbird: bamboo forest 4.05; dense foliage 2.61; grassland 0.88.

## Jak człowiek interpretuje słowa (3 pytania)
1. Czy słowa opisują klasyfikowany obiekt, czy coś innego? (ocean/boat nie opisują ptaka)
2. Jaki mają wspólny mianownik? (miejsce → tło; blurry/dark → jakość zdjęcia; actress → płeć)
3. Czy jest symetria: błędy klasy A mają cechy typowe dla klasy B? To najsilniejszy sygnał
   fałszywej korelacji i JEDYNY, który da się policzyć bez wiedzy o dziedzinie.
Odrzucać: słowa opisujące klasę (bird, wading) i artefakty captionera (superimposed).

## Automatyzacja — 3 poziomy
A. Bez nazywania pojęcia (w pełni automatyczny): do debiasingu wystarczą DWIE LISTY SŁÓW, nie nazwa.
   Filtr: Score > 0 i Bias = S → odrzuć słowa podobne do nazwy klasy (CLIP) → zostaw te, które
   pasują bardziej do POPRAWNYCH zdjęć drugiej klasy. Wynik wpisujesz w class_keywords_all.
B. LLM nazywa i porządkuje: wyślij tabele słów obu klas, poproś o JSON
   {"attribute": "...", "values": {"<wartosc>": [słowa]}, "discarded": {...}}.
   Repo ma już integrację z GPT (function/gpt_keywords.py, gpt_models.py).
   Tańsza wersja bez LLM: klastrowanie embeddingów tekstowych CLIP (k-means), bez nazw.
C. Automatyczna weryfikacja (zamiast oglądania zdjęć):
   C1. CLIP etykietuje grupy → licz poprawność w 4 grupach; luka >15 p.p. potwierdza hipotezę.
   C2. Test kontrfaktyczny: edycja obrazu wg tekstu (np. InstructPix2Pix, "change background to forest")
       i pomiar, jak często zmienia się decyzja modelu. To test przyczynowości, nie korelacji.
Inne testy ręczne: przegląd 30-50 błędnych zdjęć; predykcja z samego tła (obiekt zamalowany);
podmiana tła; Grad-CAM.

## Ograniczenia
- B2T widzi tylko to, co opisze captioner i rozpozna CLIP. Nie wykryje znaku wodnego, szumu JPEG,
  modelu aparatu, subtelnych tekstur; słabo działa w medycynie/obrazach satelitarnych.
- Score to korelacja, nie przyczyna — stąd testy C2.
- Bez interpretacji: JTT (upweighting błędów ERM) — brak wyjaśnienia, ale działa też dla biasów
  nieopisywalnych słowami. Dobry punkt odniesienia.

## Zadanie
[wpisz, co chcesz zrobić]
