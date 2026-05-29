"|Andres Tejocote Millan Mi primer modelo de inteligencia artificial"
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print(" Paso 1 completado — librerías cargadas correctamente")

# ── PASO 2: DATASET ───────────────────────────────
data = load_breast_cancer()

X = data.data
y = data.target

print("=" * 60)
print("DATASET: Breast Cancer Wisconsin")
print("=" * 60)

print(f"Total de muestras: {X.shape[0]}")
print(f"Características: {X.shape[1]}")
print(f"Clases: {list(data.target_names)}")

print(f"Malignos: {sum(y==0)}")
print(f"Benignos: {sum(y==1)}")

print("\nPrimera muestra:")
print(X[0, :5].round(2))
print(f"Etiqueta: {data.target_names[y[0]]}")

print("\n Paso 2 completado")

# ── PASO 3: DIVISIÓN ──────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("=" * 60)
print("DIVISIÓN DEL DATASET")
print("=" * 60)

print(f"Total: {len(X)}")
print(f"Entrenamiento: {len(X_train)}")
print(f"Prueba: {len(X_test)}")

print("\nDistribución:")
print(f"Malignos train: {sum(y_train==0)}")
print(f"Benignos train: {sum(y_train==1)}")
print(f"Malignos test: {sum(y_test==0)}")
print(f"Benignos test: {sum(y_test==1)}")

print("\n Paso 3 completado")

# ── PASO 4: MODELO ───────────────────────────────

from sklearn.ensemble import RandomForestClassifier

N_ESTIMATORS = 10
MAX_DEPTH = 3
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1

modelo = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_split=MIN_SAMPLES_SPLIT,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    random_state=42
)

print("=" * 60)
print("MODELO CREADO — Random Forest Classifier")
print("=" * 60)

print(f"n_estimators : {N_ESTIMATORS} (árboles)")
print(f"max_depth : {MAX_DEPTH}")
print(f"min_samples_split : {MIN_SAMPLES_SPLIT}")
print(f"min_samples_leaf : {MIN_SAMPLES_LEAF}")

print("\n⚙ Modelo creado pero NO entrenado aún.")
print("➡ El entrenamiento será en el Paso 5")

print("\n Paso 4 completado")

# ── PASO 5: ENTRENAR EL MODELO ─────────────────────

print("Entrenando el modelo... ", end="", flush=True)

modelo.fit(X_train, y_train)

print("listo ✓")
print()

# ── INFORMACIÓN DEL MODELO ─────────────────────────

n_arboles_reales = len(modelo.estimators_)

print("=" * 60)
print("RESULTADOS DEL ENTRENAMIENTO")
print("=" * 60)

print(f"Árboles creados : {n_arboles_reales}")
print(f"Clases aprendidas : {list(modelo.classes_)} → 0=maligno, 1=benigno")
print(f"Características usadas : {modelo.n_features_in_}")

print(f"Profundidad árbol #1 : {modelo.estimators_[0].get_depth()}")

print("\n✅ Paso 5 completado — modelo entrenado")
print("El modelo ya aprendió de los datos de entrenamiento.")

# ── PASO 6: PREDECIR Y MÉTRICAS ───────────────────

y_pred = modelo.predict(X_test)

# ── MÉTRICAS ───────────────────────────────────────
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# ── RESULTADOS ─────────────────────────────────────
print("=" * 60)
print(f"RESULTADOS — n_est={N_ESTIMATORS}, depth={MAX_DEPTH}")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f} ← importante (malignos)")
print(f"F1-Score  : {f1:.4f}")

print("\nReporte detallado:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# ── MATRIZ DE CONFUSIÓN ───────────────────────────
cm_vals = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm_vals.ravel()

print("\nDesglose de predicciones:")
print(f"TP = {TP} → benignos correctamente detectados")
print(f"TN = {TN} → malignos correctamente detectados")
print(f"FP = {FP} → malignos clasificados como benignos (RIESGO)")
print(f"FN = {FN} → benignos clasificados como malignos")

print(f"\n⚠ {FP} tumores malignos NO detectados")

print("\n✅ Paso 6 completado — métricas calculadas")

# ── PASO 7: MATRIZ DE CONFUSIÓN ───────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

fig.suptitle(
    f"Matriz de Confusión — n_est={N_ESTIMATORS}, depth={MAX_DEPTH}",
    fontsize=13,
    fontweight="bold"
)

# ── PANEL IZQUIERDO ───────────────────────────────
valores = np.array([[TN, FP], [FN, TP]], dtype=float)

etiquetas_rc = ["Maligno (real)", "Benigno (real)"]
etiquetas_cc = ["Pred: Maligno", "Pred: Benigno"]
nombres_celdas = [["TN", "FP"], ["FN ⚠", "TP"]]

ax1 = axes[0]
img1 = ax1.imshow(valores, cmap="Blues")

for i in range(2):
    for j in range(2):
        val = int(valores[i, j])
        color = "white" if valores[i, j] > valores.max() * 0.5 else "black"

        ax1.text(
            j, i,
            f"{nombres_celdas[i][j]}\n{val}",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color=color
        )

ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])

ax1.set_xticklabels(etiquetas_cc, fontsize=10)
ax1.set_yticklabels(etiquetas_rc, fontsize=10)

ax1.set_xlabel("Predicción del modelo", fontsize=11)
ax1.set_ylabel("Valor real", fontsize=11)
ax1.set_title("Valores absolutos", fontsize=11)

plt.colorbar(img1, ax=ax1)

# ── PANEL DERECHO ─────────────────────────────────
ax2 = axes[1]

metr_nombres = ["Accuracy", "Precision", "Recall", "F1-Score"]
metr_valores = [accuracy, precision, recall, f1]
colores_barras = ["#378ADD", "#f97316", "#ef4444", "#7c3aed"]

barras = ax2.bar(
    metr_nombres,
    metr_valores,
    color=colores_barras,
    alpha=0.85,
    edgecolor="white"
)

for barra, val in zip(barras, metr_valores):
    ax2.text(
        barra.get_x() + barra.get_width() / 2,
        barra.get_height() + 0.005,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold"
    )

ax2.set_ylim(0.0, 1.08)
ax2.set_title("Métricas de rendimiento", fontsize=11)
ax2.set_ylabel("Valor", fontsize=11)

ax2.axhline(0.9, color="gray", linestyle="--", linewidth=1, label="umbral 90%")
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()

# ── GUARDAR IMAGEN ───────────────────────────────
nombre_img = f"resultado_n{N_ESTIMATORS}_d{MAX_DEPTH}.png"

plt.savefig(nombre_img, dpi=120, bbox_inches="tight")

print(f"\nImagen guardada → {nombre_img}")

plt.show()

print("\n✅ Paso 7 completado — gráfica generada")


#################
configuraciones = [
    {"nombre": "Config A — base", "n_estimators": 10, "max_depth": 3, "min_samples_leaf": 1},
    {"nombre": "Config B — +árboles", "n_estimators": 50, "max_depth": 3, "min_samples_leaf": 1},
    {"nombre": "Config C — +profund", "n_estimators": 50, "max_depth": 10, "min_samples_leaf": 1},
    {"nombre": "Config D — +regular", "n_estimators": 100, "max_depth": 10, "min_samples_leaf": 3},
    {"nombre": "Config E — sin límite", "n_estimators": 100, "max_depth": None, "min_samples_leaf": 1},
]
########

# ── PASO 8: COMPARAR CONFIGURACIONES ───────────────

tabla = []

print("Entrenando 5 configuraciones... ", end="", flush=True)

for cfg in configuraciones:

    m = RandomForestClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        min_samples_leaf=cfg["min_samples_leaf"],
        random_state=42
    )

    m.fit(X_train, y_train)
    yp = m.predict(X_test)

    c = confusion_matrix(y_test, yp)
    tn2, fp2, fn2, tp2 = c.ravel()

    tabla.append({
        "nombre": cfg["nombre"],
        "n_est": cfg["n_estimators"],
        "depth": str(cfg["max_depth"]),
        "leaf": cfg["min_samples_leaf"],
        "TP": int(tp2),
        "FP": int(fp2),
        "FN": int(fn2),
        "Accuracy": round(accuracy_score(y_test, yp), 4),
        "Precision": round(precision_score(y_test, yp), 4),
        "Recall": round(recall_score(y_test, yp), 4),
        "F1": round(f1_score(y_test, yp), 4),
    })

    print(".", end="", flush=True)

print(" listo ✓\n")

# ── TABLA ──────────────────────────────────────────

SEP = "=" * 105

print(SEP)
print("TABLA COMPARATIVA DE HIPERPARÁMETROS")
print(SEP)

print(f"{'Configuración':<22} {'n_est':>6} {'depth':>7} {'leaf':>5}"
      f"{'TP':>4} {'FP':>4} {'FN':>4}"
      f"{'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")

print("-" * 105)

mejor_f1 = max(r["F1"] for r in tabla)
menor_fn = min(r["FN"] for r in tabla)

for r in tabla:

    marca = ""
    if r["F1"] == mejor_f1:
        marca = " ← mejor F1"
    elif r["FN"] == menor_fn:
        marca = " ← menos FN"

    print(f"{r['nombre']:<22} {r['n_est']:>6} {r['depth']:>7} {r['leaf']:>5}"
          f"{r['TP']:>4} {r['FP']:>4} {r['FN']:>4}"
          f"{r['Accuracy']:>9.4f} {r['Precision']:>10.4f}"
          f"{r['Recall']:>8.4f} {r['F1']:>8.4f}{marca}")

print(SEP)

# ── GRÁFICAS ───────────────────────────────────────

fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))

fig2.suptitle(
    "Comparación de 5 configuraciones de hiperparámetros",
    fontsize=13,
    fontweight="bold"
)

nombres_conf = [r["nombre"].strip() for r in tabla]
metricas_comp = ["Accuracy", "Precision", "Recall", "F1"]

col_comp = ["#378ADD", "#f97316", "#ef4444", "#7c3aed"]
xpos = np.arange(len(tabla))
ancho = 0.18

for i, (met, col) in enumerate(zip(metricas_comp, col_comp)):
    vals = [r[met] for r in tabla]

    ax3.bar(xpos + i * ancho, vals, ancho,
            label=met, color=col, alpha=0.85, edgecolor="white")

ax3.set_xticks(xpos + 1.5 * ancho)
ax3.set_xticklabels(nombres_conf, rotation=20, ha="right", fontsize=9)
ax3.set_ylim(0.80, 1.03)
ax3.set_title("Métricas por configuración")
ax3.legend(fontsize=9)
ax3.grid(axis="y", alpha=0.3)

# ── FN COMPARACIÓN ────────────────────────────────

fns_comp = [r["FN"] for r in tabla]

col_fn = [
    "#22c55e" if v == min(fns_comp)
    else "#ef4444" if v == max(fns_comp)
    else "#f97316"
    for v in fns_comp
]

bars_fn = ax4.bar(nombres_conf, fns_comp,
                  color=col_fn, alpha=0.88, edgecolor="white")

ax4.set_xticks(range(len(nombres_conf)))
ax4.set_xticklabels(nombres_conf, rotation=20, ha="right", fontsize=9)

for b, v in zip(bars_fn, fns_comp):
    ax4.text(
        b.get_x() + b.get_width() / 2,
        b.get_height() + 0.1,
        str(v),
        ha="center",
        fontsize=12,
        fontweight="bold"
    )

ax4.set_title("Falsos Negativos por configuración")
ax4.set_ylabel("Cantidad de FN")
ax4.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("comparacion_5configs.png", dpi=120, bbox_inches="tight")

print("Gráfica guardada → comparacion_5configs.png")
plt.show()

print("\n✅ Paso 8 completado — comparación lista")

input("\nPrograma completado. Presiona Enter para salir.")

Publicación
