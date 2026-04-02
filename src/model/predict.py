import pickle
import numpy as np
import os

def cargar_modelo():
    ruta = os.path.join("src", "model", "modelo.pkl")
    with open(ruta, "rb") as f:
        modelo = pickle.load(f)
    return modelo

def predecir(alc, conduct, sdt, fluoruros, dur):
    modelo = cargar_modelo()
    muestra = np.array([[alc, conduct, sdt, fluoruros, dur]])
    prediccion = modelo.predict(muestra)[0]
    probabilidades = modelo.predict_proba(muestra)[0]
    clases = modelo.classes_

    print(f"\nPredicción: {prediccion}")
    print("\nProbabilidades por clase:")
    for clase, prob in sorted(zip(clases, probabilidades), key=lambda x: -x[1]):
        print(f"  {clase}: {prob*100:.1f}%")

    return prediccion

if __name__ == "__main__":
    # Ejemplo de predicción con valores escalados
    predecir(
        alc=-0.15,
        conduct=-0.28,
        sdt=-0.25,
        fluoruros=-0.28,
        dur=-0.28
    )