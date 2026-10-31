Dobra, muszę spisać ogólny plan działania. Co wgl chcę, po kolei, w punktach, wykonać.

# Plan ogólny:
1. Ablation studies nad samym B2T:
   - Porównanie różnych technik i modeli:
     - Captioning
     - Ekstrakcja słów kluczowych
     - Scoring
   - Co z tego będę miał?
     - Będę miał na koniec słówa kluczowe, które mogę subiektywnie ocenić
     - Subiektywna ocena nie jest najlepsza, bo może być stronnicza. Dlatego trzeba będzie zrobić też ocenę obiektywną.
       - Ale jaką?
     - Mają różne słowa kluczowe, powiedzmy baseline z pracy oraz moje różne warianty to do czego mogę to wykorzystać? 
       - Mogę to wykorzystać do oceny, które słowa kluczowe są lepsze w kontekście debiasingu. Mogę też porównać wyniki debiasingu z różnymi zestawami słów kluczowych i zobaczyć, które dają lepsze rezultaty.

2. Debiasing:
   - Wykorzystanie słów kluczowych do debiasingu modeli.
   - 


# Plan szczegółowy:
1. Ablation studies nad samym B2T:
   - Porównanie różnych technik i modeli:
     - Captioning
       - Wykorzystanie różnych modeli captioningu
         - Baseline z pracy - ClipCap
         - MultiCaption
         - LLM płatne - np. GPT-4, Claude
         - LLM open source - np. LLaMA?
         - Inne?
       - Porównanie jakości generowanych captionów
         - Subiektywna ocena jakości captionów
         - Obiektywna ocena jakości captionów (np. BLEU, ROUGE, METEOR)???
         - Czy da się to zrobić obiektywnie?
     - Ekstrakcja słów kluczowych
       - Baseline z pracy - YAKE
       - Inne metody ekstrakcji
         - LLM - np. GPT
         - Inne metody NLP - np. KeyBERT, RAKE
           - NLP???
       - Porównanie jakości i trafności wyekstrahowanych słów kluczowych
         - Subiektywna ocena jakości słów kluczowych??
         - Obiektywna ocena jakości słów kluczowych (np. porównanie z ground truth, jeśli jest dostępne)???
     - Scoring
       - Różne metody scoringu
       - Baseline z pracy - ClipScore
       - VQA score
         - jak to jest wgl liczone?
       - Inne metody scoringu
       - LLM?
2. Debiasing:
   - ...