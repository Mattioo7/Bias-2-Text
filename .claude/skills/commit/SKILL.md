---
name: commit
description: Analizuje niezacommitowane zmiany, proponuje commit message wg wytycznych projektu i commituje po akceptacji.
disable-model-invocation: true
allowed-tools: Bash(git status:*), Bash(git diff:*), Bash(git add:*), Bash(git commit:*), Bash(git log:*)
---

## Kontekst
- Status: !`git status --short`
- Zmiany staged: !`git diff --cached`
- Zmiany unstaged: !`git diff`
- Ostatnie commity: !`git log --oneline -5`

## Wytyczne commit message
- Jedna linia, po angielsku, krótko (do ~70 znaków).
- Tryb rozkazujący, np. "Add config confirmation", "Fix typo in .gitignore".
- Bez kropki na końcu, bez body, bez trailerów (żadnego Co-Authored-By).

## Zadanie
1. Przeanalizuj zmiany z kontekstu powyżej.
2. Zaproponuj commit message zgodny z wytycznymi.
3. Pokaż propozycję i listę plików, które wejdą do commita (staged + unstaged + untracked). Zapytaj o akceptację.
4. Dopiero po akceptacji wykonaj `git add -A` i `git commit -m "<message>"`.
