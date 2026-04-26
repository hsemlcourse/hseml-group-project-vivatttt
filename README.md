[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project - Классификация экзопланет по физическому типу

**Студент:** Пустовой Георгий Юрьевич

**Группа:** БИВ231


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуск](#запуск)
4. [Данные](#данные)
5. [Результаты](#результаты)
6. [Отчёт](#отчёт)


## Описание задачи

**Задача:** многоклассовая классификация экзопланет по физическому типу - `terrestrial` (землеподобные), `neptunian` (нептуноподобные), `gas_giant` (газовые гиганты).

**Датасет:** [Open Exoplanet Catalogue (Kaggle)](https://www.kaggle.com/datasets/mrisdal/open-exoplanet-catalogue), `data/raw/oec.csv`, 3584 строки x 25 признаков.

**Целевая переменная:** выводится из массы (`PlanetaryMassJpt`) и радиуса (`RadiusJpt`); сами эти признаки удаляются из X, чтобы избежать тривиальной утечки таргета.

**Целевая метрика:** `F1-macro` (классы примерно сбалансированы, важен качественный отлов всех типов).

## Структура репозитория
Опишите структуру проекта, сохранив при этом верхнеуровневые папки. Можно добавить новые при необходимости.
```
.
├── data
│   ├── processed               # Очищенные и обработанные данные
│   └── raw                     # Исходные файлы
├── models                      # Сохранённые модели 
├── notebooks
│   ├── 01_eda.ipynb            # EDA
│   ├── 02_baseline.ipynb       # Baseline-модель
│   └── 03_experiments.ipynb    # Эксперименты и ablation study
├── presentation                # Презентация для защиты
├── report
│   ├── images                  # Изображения для отчёта
│   └── report.md               # Финальный отчёт
├── src
│   ├── preprocessing.py        # Предобработка данных
│   └── modeling.py             # Обучение и оценка моделей
├── tests
│   └── test.py                 # Тесты пайплайна
├── requirements.txt
└── README.md
```

## Запуск
```bash
git clone <url>
cd hseml-group-project-vivatttt

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_baseline.ipynb

pytest -q
ruff check src/ --line-length 120
```

## Данные
- `data/raw/oec.csv` - исходный Open Exoplanet Catalogue.
- `data/processed/oec_clean.csv` - после очистки, генерируется `notebooks/01_eda.ipynb`.

## Результаты

Сплит: stratified 70/15/15 (train/val/test), `random_state=42`.

| Модель           | Val Accuracy | Val F1-macro | Комментарий                              |
|------------------|--------------|--------------|------------------------------------------|
| Baseline LogReg  | 0.68        | 0.69        | LogisticRegression «из коробки»          |
| KNN (k=15)       | 0.72        | 0.72        | сильно лучше baseline                    |
| RandomForest     | 0.71        | 0.70        | переобучается, нужен тюнинг (этап 2)     |

Финальная модель и тюнинг - этап 2.

## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
