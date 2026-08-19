# ⚡ QuantOdds Pro 360 — Multi-Market Stochastic Engine

Plataforma cuantitativa de grado institucional para la *deconvolución de cuotas de apuestas deportivos, estimación de valor esperado ($+EV$) multimercado y detección automática de **trampas de liquidez (*Value Traps)**.

## 🚀 Características Principales

* *Motor Bivariado Dixon-Coles:* Genera matrices $10 \times 10$ de marcadores exactos con corrección de dependencia de goles ($\tau$).
* *Desmarcado por Modelo de Shin:* Aísla el parámetro $z$ (Informed Money Index) para extraer probabilidades verdaderas libres de margen.
* *Ingeniería Inversa (SciPy L-BFGS-B):* Reconstruye los goles esperados implícitos ($\lambda_{\text{imp}}, \mu_{\text{imp}}$) asumidos por las casas de apuestas a partir de sus cuotas públicas.
* *Auditor Multimercado Anti-Trampas:* Realiza validación cruzada entre $1\text{X}2$, Hándicap Asiático, Totales y BTTS.
* *Kelly Fraccionado Dinámico:* Dimensiona el tamaño de la postura ($1/5$ Kelly) minimizando el riesgo de drawdown.

## 🛠️ Instalación y Ejecución

1. Clonar el repositorio:
   ```bash
   git clone [https://github.com/TU_USUARIO/quant-odds-pro.git](https://github.com/TU_USUARIO/quant-odds-pro.git)
   cd quant-odds-pro
