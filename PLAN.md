# Plan de trabajo: Home Credit Default Risk (GCI Global 2026)

**Objetivo:** predecir la probabilidad de impago (`TARGET = 1`) de cada cliente de `test.csv`.
**Métrica:** ROC-AUC (se entrega la probabilidad, nunca 0/1).
**Fecha límite:** 20 de noviembre, 1:00 UTC (2:00 hora de Italia). Cuenta solo la **última** entrega.

> Analogía general: `train.csv` son exámenes antiguos ya corregidos, `test.csv` es el examen nuevo sin nota. El modelo estudia los primeros para adivinar la nota del segundo.

---

## Datos en resumen

| Archivo | Filas | Columnas |
|---|---|---|
| `train.csv` | 171.202 | 34 (incluye `TARGET`) |
| `test.csv` | 61.500 | 33 |

- Solo el **8,1%** de los clientes tiene `TARGET = 1`: datos desbalanceados.
- Valores faltantes altos: `EXT_SOURCE_1` (69,5%), `OWN_CAR_AGE` (66%), `EXT_SOURCE_3` (31,9%), `AMT_REQ_CREDIT_BUREAU_*` (13,5%), `OCCUPATION_TYPE`.
- Variables más correlacionadas con el target: `EXT_SOURCE_3`, `EXT_SOURCE_2`, `EXT_SOURCE_1`.

## Reglas a respetar

- [ ] Sin datos externos: solo `train.csv` y `test.csv`.
- [ ] Sin hand-labeling: todas las predicciones salen del modelo, nada de reglas manuales sobre el test.
- [ ] Reproducibilidad: fijar `SEED = 42` en todo (`numpy`, `random_state`, modelos).
- [ ] Respetar el formato de `sample_submission.csv` (`SK_ID_CURR`, `TARGET`).
- [ ] Para Honors: subir también el notebook en ZIP.

---

## Fase 0. Preparación

- [ ] Ejecutar `tutorial.ipynb` tal cual y subir la primera submission (para comprobar que el flujo funciona).
- [ ] Anotar el AUC del tutorial en la tabla de experimentos (abajo).
- [ ] Crear un notebook propio, por ejemplo `main.ipynb`, sin tocar el tutorial.
- [ ] Revisar `HomeCredit_columns_description.xlsx` para entender cada columna.

## Fase 1. EDA (entender los datos)

> Analogía: antes de cocinar, mirar qué hay en la nevera, qué está caducado y qué falta.

- [x] Tipos de variables: numéricas, categóricas, binarias (flags), fechas en días.
- [x] Valores faltantes por columna en train y test.
- [x] Distribución de cada variable y su relación con `TARGET` (tasa de impago por categoría o por tramo).
- [x] Anomalías ya detectadas, a confirmar y decidir qué hacer:
  - `DAYS_EMPLOYED = 365243` en el **18%** de los clientes (valor imposible, ~1000 años). Probablemente jubilados o desempleados.
  - `ORGANIZATION_TYPE = "XNA"` en 30.898 filas (es un faltante disfrazado).
  - `CODE_GENDER = "XNA"` en 4 filas, `NAME_FAMILY_STATUS = "Unknown"` en 2.
  - `FLAG_MOBIL` es casi constante (solo 1 cliente con 0): no aporta información.
  - `OWN_CAR_AGE` con valores raros por encima de 60.
  - `AMT_INCOME_TOTAL` con outliers muy grandes (mirar en escala logarítmica).
- [x] Comprobar que train y test tienen distribuciones parecidas.
- [x] Hecho en `eda.ipynb` (9 nov). Hallazgos nuevos:
  - `DAYS_EMPLOYED = 365243` son exactamente los pensionistas (+10 desempleados) y `FLAG_EMP_PHONE = 0` repite esa misma información.
  - `OWN_CAR_AGE` = 64 y 65 son códigos (1.825 filas), no edades reales: el corte es en 64, no en 60.
  - `OCCUPATION_TYPE` no tiene NaN: los desconocidos vienen como `Other` (31%).
  - Un ingreso de 117 millones en test (máximo de train: 13,5 M).
  - Que falte un dato sube el impago (hasta +2,7 puntos en `AMT_REQ_CREDIT_BUREAU_*`).

## Fase 2. Preprocesamiento

- [x] `DAYS_EMPLOYED == 365243`: convertir a `NaN` y crear una flag `DAYS_EMPLOYED_ANOM`.
- [x] `"XNA"` y `"Unknown"`: convertir a `NaN`.
- [x] Eliminar `FLAG_MOBIL`.
- [x] `OWN_CAR_AGE >= 64`: convertir a `NaN` y crear una flag `OWN_CAR_AGE_ANOM`.
- [x] Probado quitar `FLAG_EMP_PHONE`: empeora (-0,0006), se mantiene.
- [ ] (Pasa a Fase 3) Pasar los `DAYS_*` a valores positivos o a años (por ejemplo, edad = `-DAYS_BIRTH / 365`).
- [x] Codificar categóricas con el tipo `category` de LightGBM. Pendiente en Fase 3: compararlo con Count Encoding para `ORGANIZATION_TYPE` y `OCCUPATION_TYPE`.
- [x] Faltantes: LightGBM los gestiona solo; solo imputar si se usa Regresión Logística.
- [x] Regla de oro: todo lo que se "aprende" (medias, conteos) se calcula en train y se aplica igual a test.

## Fase 3. Feature engineering (crear variables nuevas)

> Analogía: un banquero no mira solo cuánto ganas o cuánto pides, sino la relación entre las dos cosas.

- [x] Ratios financieros:
  - `CREDIT_INCOME_RATIO = AMT_CREDIT / AMT_INCOME_TOTAL`
  - `ANNUITY_INCOME_RATIO = AMT_ANNUITY / AMT_INCOME_TOTAL`
  - `CREDIT_TERM = AMT_ANNUITY / AMT_CREDIT` (aprox. duración del préstamo)
  - `GOODS_CREDIT_RATIO = AMT_GOODS_PRICE / AMT_CREDIT`
  - `INCOME_PER_PERSON = AMT_INCOME_TOTAL / CNT_FAM_MEMBERS`
- [x] Ratios de tiempo:
  - `EMPLOYED_BIRTH_RATIO = DAYS_EMPLOYED / DAYS_BIRTH`
  - `AGE_YEARS`
- [x] Combinaciones de `EXT_SOURCE_1/2/3`: media, mínimo, máximo, desviación, producto, número de faltantes.
- [x] Número de faltantes por fila.
- [x] Añadir de una en una o en pequeños grupos y comprobar si el AUC de validación sube.
- [x] Hecho en `main.ipynb` (9 nov). Resultado: **solo los ratios financieros (grupo B) mejoran: CV AUC 0,7577 (+0,0078)**.
  - `CREDIT_TERM` (cuota / crédito ≈ 1 / número de cuotas) aporta casi el 90% de la mejora.
  - Combinaciones de `EXT_SOURCE`, ratios de tiempo, `N_MISSING` y Count Encoding: dentro del ruido, no se usan.
  - Submission: `output/submission_v2_lgbm_ratios.csv`.

- [x] Segunda vuelta (9 nov), sección 9 de `main.ipynb`: **tipo de interés implícito** deducido de crédito y cuota para plazos estándar (12-60 meses) + señales de cuotas → **CV AUC 0,7652** (+0,0034 sobre el modelo ajustado). Submission: `output/submission_v4_lgbm_rates.csv` → LB **0,76912**, puesto 446.
  - Sin mejora: comparar al cliente con la mediana de su grupo (sector, ocupación...) y tasa de impago de los 200 clientes más parecidos (KNN).

- [x] Tercera vuelta (10 nov), sin mejora: interés calculado sobre el precio del bien, parte extra del crédito, edad al empezar a trabajar, edad al terminar el préstamo, total de consultas al buró (todas dentro del ruido).

## Fase 4. Modelo base sólido

> Analogía de la validación cruzada: en vez de un solo examen de prueba, hacer 5 simulacros distintos y promediar la nota. Es más fiable.

- [x] Validación: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- [x] Modelo principal: **LightGBM** con early stopping.
- [x] Guardar el AUC out-of-fold (OOF) como nota de referencia.
- [x] Predicción en test = media de los 5 modelos de los folds.
- [x] Revisar la importancia de variables (feature importance) para saber qué funciona.
- [x] Hecho en `main.ipynb` (9 nov): **CV AUC 0,7499** (tutorial: 0,669). Submission: `output/submission_v1_lgbm_phase2.csv`.
- [x] Comparar con el AUC del Public Leaderboard: v2 dio **0,76405** en el LB (CV 0,7577), puesto 1849. CV y LB van en la misma dirección.

## Fase 5. Mejora del modelo

- [x] Ajustar hiperparámetros (`learning_rate`, `num_leaves`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`) a mano o con Optuna.
- [ ] Probar `scale_pos_weight` o `is_unbalance` por el desbalance (a veces no ayuda al AUC, comprobar).
- [x] Probar otros modelos: **CatBoost** con las mismas variables → CV 0,7644 (LightGBM 0,7652). Correlación entre ambos: 0,965.
- [x] Hecho en `main.ipynb` (9 nov): `num_leaves=15`, `colsample_bytree=0.2`, `learning_rate=0.02` → **CV AUC 0,7618** (+0,0041). Submission: `output/submission_v3_lgbm_tuned.csv` → LB **0,76557**, puesto 1040.
  - Sin mejora: `cat_smooth`, más regularización L1/L2, `min_child_samples` 100-200, `num_leaves` 7 o 63.

## Fase 6. Ensemble

> Analogía: preguntar a varios expertos y promediar su opinión suele equivocarse menos que uno solo.

- [x] Promediar LightGBM con varias semillas (seed averaging): 3 semillas → LightGBM 0,7657, blend **0,7669**. Submission de prueba: `output/submission_v6_blend_lgbm3seeds_catboost.csv` (pendiente de pasar al notebook final).
- [x] Mezclar modelos distintos (por ejemplo, LightGBM + CatBoost) con pesos elegidos según el AUC OOF.
  - Hecho (10 nov), sección 10 de `main.ipynb`: media 50/50 → **CV AUC 0,7665**. Pesos entre 0,4 y 0,6 dan lo mismo, por eso 50/50. Submission: `output/submission_v5_blend_lgbm_catboost.csv` → LB **0,76957**, puesto 413.
- [ ] Probar la media de rankings en vez de la media de probabilidades.

## Fase 7. Entrega final

- [ ] Comprobar el formato: mismas filas y mismo orden que `sample_submission.csv`, sin `NaN`, valores entre 0 y 1.
- [ ] Elegir la submission final según el CV, no solo según el Public LB (el ranking final usa el Private LB).
- [ ] Subir la última versión **antes del 19 de noviembre** para tener margen.
- [ ] Limpiar el notebook, ejecutarlo de arriba a abajo y comprobar que se reproduce.
- [ ] Generar el ZIP del notebook (requisito para Honors).
