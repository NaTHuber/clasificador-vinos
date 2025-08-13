# Notas del proyecto 
El siguiente proyecto tiene como objetivo prácticar los conocimientos adquiridos en el [video del curso de deep learning del Qubit de Newton](https://youtu.be/EZJOxvMOZas?si=IO8x9ebSbQo0gpE2)

Estas notas son apuntes personales del proyecto 

# Flujo de trabajo 

## Importaciones y carga de datos

###  Librerias  

- torch → Librería principal de PyTorch.

- torch.nn → Módulo que contiene clases para construir redes neuronales (capas, funciones de activación, etc.).

- torch.optim → Algoritmos de optimización (SGD, Adam, etc.).

- load_wine → Dataset de vinos incluido en scikit-learn (atributos químicos de vinos y su clasificación en 3 tipos).

- train_test_split → Divide el dataset en entrenamiento y prueba.

- StandardScaler → Escala los datos para que tengan media 0 y desviación estándar 1

- matplotlib.pyplot → Para graficar.

- numpy → Para operaciones numéricas.

### Carga del dataset 

- data.data (X) → Matriz de características (atributos de cada vino).

- data.target (y) → Vector con la clase de cada vino (0, 1 o 2).


---


## Exploración, limpieza, preprocesamiento y estructuración del dataset

- Número de muestras: 178

- Número de características: 13

**Características del dataset**
| Campo                             | Descripción                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|
| **alcohol**                       | Contenido de alcohol en el vino (% vol.)                                   |
| **malic_acid**                    | Ácido málico (contribuye al sabor ácido)                                   |
| **ash**                           | Cenizas (minerales inorgánicos)                                            |
| **alcalinity_of_ash**             | Alcalinidad de las cenizas (medida de pH)                                  |
| **magnesium**                     | Contenido de magnesio (mg)                                                 |
| **total_phenols**                 | Fenoles totales (contribuyen al sabor y aroma)                             |
| **flavanoids**                    | Flavonoides (antioxidantes que afectan el color y sabor)                   |
| **nonflavanoid_phenols**          | Fenoles no flavonoides (otros compuestos fenólicos)                        |
| **proanthocyanins**               | Proantocianidinas (taninos que influyen en la astringencia)                |
| **color_intensity**               | Intensidad del color (relacionado con la concentración de antocianinas)    |
| **hue**                           | Tonalidad del color (matiz, del amarillo al azul/púrpura)                  |
| **od280/od315_of_diluted_wines**  | Absorbancia a 280 nm / 315 nm (relacionado con proteínas y polifenoles)    |
| **proline**                       | Prolina (aminoácido que influye en el sabor y estructura del vino)         |

### Histograma 

![alt text](img/histograma.png)


- Hay características con escalas muy diferentes (proline llega a 1600, mientras nonflavanoid_phenols < 1) → necesitaremos escalado.

- Algunas variables como color_intensity o malic_acid son muy asimétricas, lo que podría afectar modelos que asumen normalidad.

- Algunas distribuciones parecen bimodales (flavanoids), lo que podría indicar que ayudan a diferenciar clases.


### Matriz de correlación

![alt text](img/matriz_correlaciones.png)

- Las variables como flavanoids, color_intensity y proline parecen muy relevantes para diferenciar clases.

- Algunas variables están muy correlacionadas, lo que podría implicar redundancia, pero en redes neuronales eso no es un problema grave porque el modelo aprende ponderaciones.

- Esta matriz nos da pistas de qué variables podrían ser más “predictoras” y cuáles son casi ruido.

### Gráfico de pares
![alt text](img/grafico_pares.png)

Esto nos da una intuición visual: hay variables muy buenas para separar clases (flavanoids, proline, od280/od315) y otras menos útiles.

Aunque se entrenará con todas, este análisis podría inspirar una versión reducida del modelo usando solo las más relevantes y comparar el rendimiento.

### División de dataset en entrenamiento y prueba y escalamiento de datos 

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,  
    random_state=42,  
    stratify=y        
)
```
- `stratify=y` → asegura que en train y test haya la misma proporción de cada clase, bastante importante en datasets pequeños.
```python
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test) 
```
- `fit_transform` vs `transform` → primero se ajusta el escalador con train y después se aplica tanto a train como a test, así evitamos fuga de datos.

--- 

## Crear la clase de la red neuronal 
MLP (Multilayer Perceptron) clásico de 3 capas, dos ocultas con ReLU y una de salida.
```python
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x) 
```
- `nn.Linear(input_size, hidden_size)` → capa densa.

- `nn.ReLU()` → no linealidad que permite aprender funciones complejas.

- Se repite capa oculta del mismo tamaño.

- `nn.Linear(hidden_size, output_size)` → capa de salida con 3 neuronas (una por clase).

<!-- - Sin softmax explícito: en clasificación multi-clase con CrossEntropyLoss, PyTorch espera logits y aplica log_softmax internamente. -->

---

## Seleccionar hiperparámetros e instanciar la red 

- `input_size = 13`: Lo ponemos igual al número de características del dataset para que la capa de entrada encaje perfectamente.

- `hidden_size = 64`: Indica cuántas neuronas tendrá cada capa oculta. Más neuronas → más capacidad de aprender patrones complejos, pero también más riesgo de sobreajuste.

- `output_size = 3`: Una neurona por cada clase.

- `learning_rate = 0.01`: Qué tan rápido se ajustan los pesos en cada paso.

- `num_epochs = 100`: Cuántas veces pasa el modelo por todos los datos de entrenamiento.

```python
model = MLP(input_size, hidden_size, output_size)
```
Crea la red con 13 entradas, 64 neuronas por capa oculta, y 3 neuronas de salida.

## Definir la función de pérdida y el optimizador

- `nn.CrossEntropyLoss()`: CrossEntropyLoss es la más común para clasificación multiclase

- `optim.Adam(model.parameters(), lr=learning_rate)`: optimizador adaptativo, ajusta la tasa de aprendizaje por parámetro. 

## Entrenar a la red 
```mermaid
flowchart TD
    A[Inicio] --> B[Inicializar listas de pérdidas]
    B --> C{¿Época <= num_epochs?}
    C -->|Sí| D[Entrenamiento: Forward + Backward]
    D --> E[Validación: Forward sin gradientes]
    E --> F[Guardar pérdidas]
    F --> G{¿Época % 10 == 0?}
    G -->|Sí| H[Imprimir métricas]
    G -->|No| C
    C -->|No| I[Fin]
```
Inicialización:

- Listas para trackear el historial de pérdidas.

Bucle de Épocas:

- Entrenamiento:

    - model.train(): Habilita dropout/batch norm.

    - zero_grad(): Evita acumulación de gradientes.

    - loss.backward(): Retropropagación automática.

Validación:

- model.eval(): Desactiva dropout/batch norm.

- no_grad(): Optimiza memoria y velocidad.

Logging:

- Muestra progreso cada 10 épocas.

## Evaluar con set de test y guardar el modelo