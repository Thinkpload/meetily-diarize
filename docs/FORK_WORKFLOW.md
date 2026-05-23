# Fork Workflow — синхронизация с апстримом

Этот форк (`Thinkpload/meetily-diarize`) живёт параллельно с оригиналом
(`Zackriya-Solutions/meetily`). Здесь описан рабочий процесс: как держать
`main` зеркалом апстрима и подтягивать обновы в свою фичу-ветку.

## Remotes

```
origin    → https://github.com/Thinkpload/meetily-diarize.git   (свой форк)
upstream  → https://github.com/Zackriya-Solutions/meetily.git   (оригинал)
```

Проверить: `git remote -v`

Если `upstream` потерялся:
```powershell
git remote add upstream https://github.com/Zackriya-Solutions/meetily.git
```

## Ветки

- **`main`** — зеркало `upstream/main`. **Не коммитить сюда напрямую.**
- **`feature/diarization`** — основная рабочая ветка для диаризации.
- Новые подзадачи — отдельными ветками от `feature/diarization` при необходимости.

## Синхронизация с апстримом

### Шаг 1. Обновить локальный `main`

Алиас уже настроен глобально:
```powershell
git sync-main
```

Что он делает (эквивалент):
```powershell
git fetch upstream
git checkout main
git merge upstream/main --ff-only
git push origin main
```

`--ff-only` страхует: если в `main` появились локальные коммиты — команда
упадёт. Это сигнал перенести их в фичу-ветку и сбросить `main`.

### Шаг 2. Подтянуть обновы в фичу-ветку

```powershell
git checkout feature/diarization
git merge main
git push origin feature/diarization
```

**Merge, не rebase.** Не переписывает историю, безопасно для запушенной ветки,
конфликты решаются один раз.

### Конфликты

```powershell
git status                  # список конфликтных файлов
# вручную разрешить маркеры <<<<<<< ======= >>>>>>>
git add <файлы>
git commit                  # завершить merge
```

## Когда синкаться

- Перед началом нового куска работы.
- Раз в неделю как минимум.
- Перед тем как открывать PR (если планируется контрибьют в апстрим).

## Восстановление алиаса

Если `git sync-main` не работает:
```powershell
git config --global alias.sync-main '!git fetch upstream && git checkout main && git merge upstream/main --ff-only && git push origin main'
```
