# Actividad 3. Análisis de un conjunto de datos de origen biológico mediante técnicas de machine learning supervisadas y no supervisadas

Autores: Ana Amador, Carla González, Valeria Pesa y Judit Puntí

## Introducción

---
El presente análisis se centra en un dataset que recopila información sobre la expresión de cientos de genes en muestras asociadas a cinco tipos distintos de cáncer:

- AGH
- CHC
- HPB
- CGC
- CFB
El objetivo principal es implementar de forma razonada técnicas de aprendizaje supervisado y no supervisados para detectar patrones intrínsecos en el conjunto de datos y evaluar la capacidad de estos métodos para distinguir e identificar los distintos tipos de cáncer en base a los distintos perfiles de expresión génica.

1) ENTORNO
En primer lugar, se cargaron las siguientes librerías para poder llevar a cabo el análisis:
```{r cars}
# Manipulación, importación, exploración y visualización de datos
library(tidyverse) 
library(skimr) 
library(readr)
library(readxl)
library(dplyr)
# Graficos
library(ggplot2)
library(pheatmap)
library(patchwork)

# Métodos de aprendizaje no supervisados
library(RDRToolbox)
library(cluster)
library(stats)
library(Rtsne)
library(stats)

# Métodos de aprendizaje supervisados
library(caret)
library(pROC)
library(randomForest)
library(MASS)
```
2) PROCESAMIENTO DE LOS DATOS
Se importaron los tres archivos provistos y se generó un único dataframe.
```{r}
setwd("C:/Users/Usuario/Documents/Master en Bioinformática/AÑO 2 (2025-2026)/Algoritmos e Inteligencia Artificial/Actividad 3")
clases <- read_excel("classes.xlsx")
nombres_genes <- read.csv("nombres.csv", header = FALSE)
datos_expresion<- read_excel("gene_expresion.xlsx")
View(datos_expresion)
#empezar a unir base de expresion y de nombres
nombres_genes <- nombres_genes[[1]] #extraemos los nombres de los genes
ncol(datos_expresion) #vemos so me coinciden las dimensiones
length(datos_expresion)
colnames(datos_expresion) <- nombres_genes #le añadimos los nombres a las columnas de los datos de expresion
#unir bbdd de expresion y de clases con la funcion cbind
df_final <- cbind(clases, datos_expresion)
View(df_final)
```
Tras una inspección inicial del conjunto de datos, se confirmó la ausencia de valores perdidos (NA) en todas las variables:
```{r}
which(is.na(df_final))
```
Por lo tanto, no fue necesario aplicar ningún método de imputación.
Luego, se utilizó la librería skim para realizar una rápida exploración de los datos, incluyendo el calculo de estadísticos descriptivos:
```{r}
skim_df <- skim(df_final)
skim_sd0 <- skim_df %>%
  filter(skim_type == "numeric") %>%
  arrange(numeric.sd) %>%
  head()

```
Se identificaron 3 genes con desviación estándar igual a cero (sd = 0), es decir, genes cuya expresión es constante en todas las muestras. Estos genes fueron eliminados del análisis, ya que no aportan información discriminativa y pueden causar problemas numéricos en métodos como PCA, LDA o clustering.
```{r}
varianza_genes <- apply(df_final, 2, var)  # calculamos la varianza por columna
genes_sinvarianza <- names(varianza_genes)[varianza_genes == 0]

df_filter <- df_final[, !(names(df_final) %in% genes_sinvarianza)]
df_numeric <- df_filter[, -c(1,2)] #eliminamos la columna CHC y la de sample del data_final para que sea solamente numerica antes de escalarla 
#A la columna de las clases, voy a llamarle Class en la base de datos df_filter
colnames(df_filter)[2] <- "Class"
```
Finalmente, los datos de expresión génica fueron escalados a media cero y varianza unitaria:
```{r}
#Hacemos el escalado del dataframe
df_scaled <- scale(df_numeric)
df_scaled <- data.frame(df_numeric)

#vemos si hay algun NA
anyNA(df_scaled)
```
El escalado elimina diferencias de magnitud entre genes, reduce el impacto de valores altos, y es esencial para alguno de los métodos utilizados a continuación.
Por lo tanto, tenemos varias bases de datos con las que trabajaremos:
df_final: conjunto de todos los genes + columna de sample + columna de Class
df_numeric: solamente genes
df_filter: conjunto de todos los genes con varianza >0 + columna de sample + columna de Class
df_scaled: base de datos df_filter escalada 

Vamos a hacer un heatmap para tener una visión de la expresión de los genes. En este caso, probamos primero con todos los genes, luego con 100 y finalmente con 50 genes con una mayor varianza. 
```{r}
varianza <- apply(df_numeric, 2, var, na.rm = TRUE) #nos quedamos con los genes que mayor varianza tengan
top_genes50 <- names(sort(varianza, decreasing = TRUE))[1:50]
matriz_genes <- df_numeric[, top_genes50] #matriz de 50 genes
 #creamos la matriz de 50 genes 
View(matriz_genes)
#Hacemos el heatmap con estos 50 genes
heatmap50 <- pheatmap(
  matriz_genes,
  show_rownames = TRUE,
  show_colnames = TRUE,
  clustering_distance_rows = "euclidean",
  clustering_distance_cols = "euclidean",
  clustering_method = "complete"
)
```
3) MÉTODOS DE APRENDIZAJE NO SUPERVISADO
3.1 Técnicas de reducción de dimensionalidad 
Se implementaron cuatro métodos no supervisados de reducción de dimensionalidad (PCA, Isomap, LLE y t-SNE) con el objetivo de explorar la estructura interna del conjunto de datos y evaluar su capacidad para revelar patrones y agrupamientos entre los distintos tipos de cáncer. Estos métodos fueron seleccionados por ofrecer enfoques complementarios, tanto lineales como no lineales, adecuados para datos de expresión génica con relaciones complejas.

PCA se utilizó como método lineal de referencia para capturar la varianza global de los datos. Isomap se seleccionó por su capacidad para preservar la estructura global en datos no lineales, LLE por mantener las relaciones locales entre muestras, y t-SNE por su elevada capacidad para visualizar agrupamientos locales y separar clases en espacios de baja dimensión.

3.1.1 Análisis de Componentes Principales (PCA): 
l objetivo principal del PCA (Análisis de Componentes Principales) es maximizar la varianza y reducir la dimensionalidad del conjunto de variables. Se caracteriza por combinar linealmente las variables originales y transformarlas en un nuevo conjunto de variables no correlacionadas, conocidas como componentes principales (PC).
```{r}
#vamos a utilizar la base de datos eescaladas, pero tengo que añadirle la columna de Class de la base de datos ya filtrada
df_scaled_class <- cbind(df_scaled, df_filter[, 2, drop = FALSE])

pca_resultados <- prcomp(df_scaled, center = TRUE, scale. = FALSE) #hago con df_scaled, en vez de df_filter porque es la que contiene solamente datos numéricos ya escalados previamente, que es uno de los requerimientos de esta técnica. 

#para los calculos de los componentes principales
pca_dataframe <- data.frame(pca_resultados$x)
varianzas_pca <- pca_resultados$sdev^2 #calculo de las varianzas
total_varianzas_pca <- sum(varianzas_pca) #total varianza de los datos
varianza_explicada <- varianzas_pca/total_varianzas_pca #varianza de cada componente principal 
varianza_acumulada <- cumsum(varianza_explicada) #calculo de la varianza acumulada
num_componentespca <- min(which(varianza_acumulada > 0.90)) #ver el numero de componentes principales que me calculan el 90% de la varianza de los datos 
#graficamos
x_label <- paste0(paste('PC1', round(varianza_explicada[1]*100,2)), '%')
y_label <- paste0(paste('PC2', round(varianza_explicada[2]*100,2)), '%')

#para ver los datos por clase, tengo que añadir esta variable al data.frame de PCA:
pca_dataframe$Class <- df_filter[, 2]

pca_grafica <- ggplot(pca_dataframe, aes(x = PC1, y = PC2, color = Class)) +
  geom_point(size = 3) +
  labs(title = 'PCA - Types of Cancer', x = x_label, y = y_label, color = 'Grupo') +
  theme_classic() +
  theme(
    panel.grid.major = element_line(color = "gray90"),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "gray95"),
    plot.title = element_text(hjust = 0.5)
  )
pca_grafica
```
En la técnica de analisis de componentes principales (PCA) podemos ver como la clase AGH se diferencia notablemente de las demás con el primer componente. Sin embargo, el resto de clases no se diferencian teniendo en cuenta estos dos componentes, por lo que esta técnica quedaría limitada para dar unos resultados concluyentes, posiblemente porque las variables no estén muy correlacionadas y se necesite un mayor numero de componentes que maximicen la varianza entre las diferentes variables.

3.1.2 Isomap 
ISOMAP es un algoritmo eficaz para descubrir estructuras no lineales en conjuntos de datos de alta dimensionalidad, ya que busca preservar las distancias geodésicas entre los puntos, es decir, las distancias a lo largo de la variedad subyacente. Este método extiende el Análisis de Componentes Principales (PCA) a contextos no lineales mediante la construcción de un grafo de vecinos y el cálculo de las distancias más cortas entre ellos. No obstante, ISOMAP presenta ciertas limitaciones, como su sensibilidad a la elección del número de vecinos y a la presencia de ruido, lo que puede provocar distorsiones en la estimación de las distancias y afectar a la calidad de la proyección final.

El número de vecinos está representado por el número k. En este caso, se escogió el valor de k=10 ya que nos permitía visualizar mejor graficamente las relaciones locales entre las diferentes clases.
```{r}
set.seed(123)
X <- as.matrix(df_scaled)
y   <- df_filter$Class
isomap.results <- Isomap(data = X, dims = 1:10, k = 10, plotResiduals = TRUE)

#sacamos el dataframe de isomap con 2 dimensiones, pero sabemos que la gráfica se empieza a aplanar con 4. 
isomap.df <- data.frame(
  X1 = isomap.results$dim2[, 1],
  X2 = isomap.results$dim2[, 2],
  class = y
)
```
Para graficar, se utilizó el siguiente código:
```{r}
ggplot(isomap.df, aes(x = X1, y = X2, color = y)) +
  geom_point(size = 2.5, alpha = 0.85) +
  labs(
    title = "Isomap (k = 10)",
    x = "Dimensión 1",
    y = "Dimensión 2",
    color = "Clase"
  ) +
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5))
```
La graficicación obtenida mediante Isomap (k = 10) muestra una separación clara entre la mayoría de las clases analizadas. La clase AGH aparece claramente aislada del resto, mientras que CFB forma un clúster amplio y bien definido. Por el contrario, las clases CGC y HPB presentan cierta proximidad y solapamiento, lo que sugiere similitudes en sus patrones de expresión génica. En conjunto, Isomap captura de forma eficaz la estructura no lineal de los datos, proporcionando una representación más informativa que métodos lineales como PCA.

3.1.3 Locally Lineal embedding (LLE)
El LLE es un algoritmo altamente eficiente en descubrir las estructuras no lineales en los datos y preservar las distancias dentro del vecindario local. El LLE presenta ciertas limitaciones, pues es muy sensible al ruido y a los valores atípicos. Además, es posible que dos puntos de datos, que en realidad no se encuentren en el mismo parche localmente lineal, sean agrupados como si lo estuvieran.

El número de vecinos, k, es su único parámetro libre, lo que simplifica el ajuste el algoritmo. En esta ocación fueron evaluados varios valores de k entre 10 y 80, seleccionando 50 como el óptimo. El algoritmo se implementó de la siguiente manera:
```{r}
#convertir df_scaled a matriz numerica
df_matriz_scaled <- as.matrix(df_scaled)
lle.results.50 <- LLE(df_matriz_scaled, dim = 2, k = 50) # La dimensión final es 2, ya que son las dos que se van a representar gráficamente

lle.df.50 <- data.frame(lle.results.50) # Creación de un dataframe a partir de los resultados

lle.df.50$Class <- df_filter$Class # Incorporación de la columna especificando los tipos de cáncer
```

Para la representación gráfica, se utilizó el siguiente código:

```{r}
plot_LLE_50 <- ggplot(lle.df.50, aes(x = X1, y = X2, color = Class)) +
  geom_point(size = 3) +
  scale_color_manual(values = rainbow(5)) +
  labs(title = "Método LLE", x = "X1", y = "X2") +
  theme_classic()
plot_LLE_50
```
Este metodo permite una clara distinción de los tipos de cáncer AGH y CHC, aunque se superponen aún CGC y CHCC y una parte de los componentes del grupo CFB. Es evidente la ventaja de utilizar, para este dataframe en particular, un método no lineal ya que separa las clases de AGH y CHC como no se habian separado antes.

3.1.4 t-SNE
El método t-distribuited Stochastic Neighbor Embedding (t-SNE) es una técnica de reducción de dimensionalidad no lineal que preserva las relaciones locales entre las muestras. Este método de análisis és especialmente útil cuando los datos no se distribuyan de forma lineal, como ocurre en la expresión génica.
```{r}
set.seed(123)
# Matriz numérica de estructura local con 30 vecinos (perplexity)
tsne <- Rtsne(as.matrix(df_scaled), perplexity = 30) 

# Dataframe con las dimensiones y las clases (tipos de cáncer)
tsne_df <- data.frame(Dim1 = tsne$Y[,1], Dim2 = tsne$Y[,2], Class = df_filter$Class) 

# Representación gráfica
tsne_grafica <- ggplot(tsne_df, aes(x = Dim1, y = Dim2, color = Class)) +
  geom_point(size = 3, alpha = 0.8) +
  labs(
    title = "Método t-SNE",
    x = "Dimensión 1",
    y = "Dimensión 2",
    color = "Tipo de cáncer"
  ) +
  theme_classic() +
  theme(
    panel.grid.major = element_line(color = "gray90"),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "gray95"),
    plot.title = element_text(hjust = 0.5)
  )

tsne_grafica
```

Entre las ventajas de t-SNE se incluyen:
- Visualización clara de agrupamientos locales incluso en datos con relaciones no lineales.
- Facilita la detección de subgrupos y patrones en un espacio reducido de dimensión.

Limitaciones a considerar:
- La interpretación de distancias globales entre grupos lejanos puede no reflejar similitudes reales.
- Los resultados dependen de parámetros como perplexity y número de iteraciones, requiriendo ajuste cuidadoso.
- No genera un modelo aplicable a nuevas muestras; se utiliza únicamente para exploración y visualización.

Como se observó en el análisis previo, la mejor separación entre los distintos tipos de cáncer se obtuvo con t-SNE, seguida de LLE. Ambos métodos permiten capturar eficazmente la estructura no lineal de los datos, si bien t-SNE destaca por su mayor capacidad para revelar patrones locales y preservar la organización global del conjunto de datos.


3.2 TÉCNICAS DE CLUSTERIZACIÓN 
Se aplicaron distintos métodos de clusterización para identificar agrupamientos naturales en los datos de expresión génica. K-means se utilizó sobre la proyección Isomap 2D por su simplicidad y eficiencia, fijando el número de clústeres según las clases reales.

Además, se emplearon métodos jerárquicos aglomerativos (single, complete, average y Ward) y el método divisivo DIANA sobre los genes de mayor varianza para reducir ruido y explorar la estructura de los datos desde enfoques complementarios.

3.2.1 Clusterización no jerárquica
```{r}
#se lleva a cabo sobre la tecnica de Isomap realizada previamente
set.seed(123)                                    # Fija la semilla para reproducibilidad

# Número de clusters = número de clases reales que se calcula sobre el numero de clases que tenemos  (Y)
k <- length(unique(y))                    

# Aplicación de K-means sobre las dos dimensiones de Isomap
km <- kmeans(
  isomap.df[, c("X1", "X2")],               
  centers = k,                                
  nstart = 25                                   
)

# Añadir el clúster asignado a cada muestra
isomap.df$cluster_km <- factor(km$cluster)

# Cálculo de la envolvente convexa para cada clúster
hulls <- isomap.df %>%
  group_by(cluster_km) %>%                       
  slice(chull(X1, X2)) %>%               
  ungroup()                                      

# Gráfico de los resultados de K-means sobre Isomap
ggplot(isomap.df, aes(X1, X2, color = cluster_km)) +
  
  geom_polygon(                                  # Dibuja los polígonos de cada clúster
    data = hulls,                                # Usa solo los puntos frontera
    aes(fill = cluster_km, group = cluster_km),  # Relleno y agrupación por clúster
    alpha = 0.2,                                 # Transparencia del polígono
    color = NA                                   # Sin borde para el polígono
  ) +
  
  geom_path(                                     # Dibuja el contorno del polígono
    data = hulls,
    aes(group = cluster_km),
    linewidth = 0.8                              # Grosor del borde
  ) +
  
  geom_point(                                    # Dibuja las muestras individuales
    size = 2,                                    # Tamaño de los puntos
    alpha = 0.85                                 # Transparencia ligera
  ) +
  
  labs(
    title = paste0("K-Means plot, centros = ", k), # Título del gráfico
    x = "Isomap - Dimensión 1",                     # Etiqueta eje X
    y = "Isomap - Dimensión 2",                     # Etiqueta eje Y
    color = "Cluster",                              # Leyenda del color
    fill  = "Cluster"                               # Leyenda del relleno
  ) +
  
  theme_minimal()                                # Tema visual limpio y académico
```
El algoritmo K-means aplicado sobre la proyección Isomap 2D identifica cinco clústeres bien definidos, con escaso solapamiento y una clara diferenciación espacial entre los grupos.Destacando la clara diferenciación del cluster numero 3. 
No podemos asegurar con certeza que los clústeres obtenidos sean los mejores, ya que este mínimo puede ser local y no global, ya que el resultado depende de la inicializacion de los centroides y del parámetro subjetivo que es k. 

3.2.2 Clusterización jerárquica aglomerativa 
Como 800 genes no nos da resultado de nada, vamos a hacer las mismas técnicas con los 50 genes selecionados anteriormente en la construcción del Heatmap con mas varianza
```{r}
distancia_matriz50 <- dist(matriz_genes50)
distancia_matriz50

hclust_single50 <- hclust(distancia_matriz50, method = "single")
hclust_complete50 <- hclust(distancia_matriz50, method = "complete")
hclust_average50 <- hclust(distancia_matriz50, method = "average")
hclust_ward50 <- hclust(distancia_matriz50, method = "ward.D2")

#calculamos la variable de indices para el orden en el dendograma

hclust_single50$order
hclust_complete50$order
hclust_average50$order
hclust_ward50$order

#graficamos los resiltados por colores y con la funcion fviz_den para hacer el dendograma

colores <- rainbow(10)
clust_single <- fviz_dend(hclust_single50, 
                          cex = 0.5, 
                          k = 10, 
                          palette = colores, 
                          main = "Single Linkage Dendogram", 
                          xlab = "Índice de Observaciones", 
                          ylab = "Distancia") + theme_classic()
clust_single

clust_complete <- fviz_dend(hclust_complete50, 
                          cex = 0.5, 
                          k = 10, 
                          palette = colores, 
                          main = "Complete Linkage Dendogram", 
                          xlab = "Índice de Observaciones", 
                          ylab = "Distancia") + theme_classic()
clust_complete

clust_average <- fviz_dend(hclust_average50, 
                          cex = 0.5, 
                          k = 10, 
                          palette = colores, 
                          main = "Average Linkage Dendogram", 
                          xlab = "Índice de Observaciones", 
                          ylab = "Distancia") + theme_classic()
clust_average

clust_ward <- fviz_dend(hclust_ward50, 
                          cex = 0.5, 
                          k = 10, 
                          palette = colores, 
                          main = "Ward Linkage Dendogram", 
                          xlab = "Índice de Observaciones", 
                          ylab = "Distancia") + theme_classic()
clust_ward
#Utilizamos la funcion de grid.arrange para juntar en una sola gráfica los dendogramas obtenidos
library(gridExtra)
dendogramas_todos <- grid.arrange(clust_single, clust_complete, clust_average, clust_ward, nrow = 2)
```
Hemos decidido ver los 4 métodos de clusterización jerarquica aglomerativa para ver cual es el más optimo. Consideramos que el más óptimo es el Ward Linkage Dendogram porque, además de ser el más comunmente usado, calcula varianzas además de las distancias, pudiendo observar como se van clasificando el primer lugar haciendo dos clusteres que son los que mayor varianza tienen y luego realizando el resto, definienedo claramente los grupos, aunque somo sconscientes de que es dificil llegar a una conclusión definida por el número de genes que manejamos.
3.2.3 Clusterización jerárquica decisiva
```{r}
library(cluster)
diana_euclidean <- diana(df_scaled, metric = "euclidean", stand = F)
clust_diana_euclidean <- fviz_dend(diana_euclidean, 
                                  cex = 0.5, 
                                  k = 10, 
                                  palette = colores, 
                                  main = "DIANA Euclidean", 
                                  xlab = "Indice de observaciones", 
                                  ylab = "Distancia") + theme_classic()
clust_diana_euclidean
```
Observamos que los resultados son similares a los obtenidos wn Ward Linkage, solamente que es una técnica de clusterizacion jerarquica decisiva, es decir, funciona de manera contraria, empezando de lo general a lo individual, calculando distancias euclídeas. 

4. METODOS DE APRENDIZAJE SUPERVISADO
Se implementaron tres métodos de aprendizaje supervisado: k-NN, LDA y Random Forest. Su elección se realizó con el objetivo de comparar enfoques supervisados complementarios. El método k-NN se seleccionó por su simplicidad y por su capacidad para capturar relaciones locales entre muestras sin asumir una forma funcional previa de los datos. LDA se incluyó como un método estadístico clásico, adecuado para evaluar la separabilidad entre clases bajo supuestos de normalidad y varianzas homogéneas, sirviendo además como una referencia interpretable. Finalmente, Random Forest se empleó por su capacidad para modelar relaciones no lineales complejas, su robustez frente al ruido y su buen rendimiento en conjuntos de datos con un gran número de variables.

Se decidió no aplicar técnicas explícitas de reducción de dimensionalidad antes del entrenamiento de los métodos. Esta decisión se basó en el filtrado previo de genes sin variabilidad y en el escalado de los datos, así como en el uso de modelos robustos frente a espacios de alta dimensionalidad, como Random Forest. Además, se priorizó la interpretabilidad de las variables originales, ya que métodos como PCA dificultan la identificación directa de genes relevantes.

Los métodos supervisados deben ser evaluados para comprobar su capacidad de generalización y su rendimiento sobre datos no vistos. Para ello, el conjunto de datos se dividió en un conjunto de entrenamiento y otro de prueba, utilizando el 80% de las muestras para el entrenamiento y el 20% restante para la validación del modelo.

Primero vamos a dividir los datos en training y testing. 
```{r}
#le pongo el nombre de Class a la segunda columna del data_final, que no la tiene
#voy a trabajar con la base de datos df_filter, sin escalar porque el hiperparametro preprocess ya va a escalar los datos. 
#Como en esta base de datos tengo la columna sample, la elimino
df_filter$sample_0 <-NULL
colnames(df_filter)[1] <- "Class"
library(caret)
trainIndex <- createDataPartition(df_filter$Class, p = 0.8, list = FALSE)
df_filter$Class <- as.factor(df_filter$Class)
trainData <- df_filter[trainIndex,]
testData <- df_filter[-trainIndex,]
```
4.1 kNN
El método de aprendizaje supervisado k-NN se basa en los datos de entrenamiento durante la fase de predicción para asignar etiquetas a los datos no etiquetados. La predicción se realiza considerando las etiquetas de los vecinos más cercanos, de manera que un nuevo dato se clasifica según la mayoría de las etiquetas de sus k vecinos más próximos.
```{r}
set.seed(123)
# Entrenamos el modelo k-NN
knnModel <- train(Class ~ .,
                  data = trainData,
                  method = "knn",
                  trControl = trainControl(method = "cv", number = 10),
                  preProcess = c("center", "scale"),
                  tuneLength = 30
)
# Visualizamos los resultados del modelo entrenado y graficamos el rendimiento
knnModel
plot(knnModel)
```
En el output se presenta una tabla con los valores de precisión (Accuracy) y Kappa obtenidos para los distintos valores de k evaluados con los datos de entrenamiento. En el gráfico se observa que la máxima precisión se alcanza con k =7 vecinos. Con este valor, el modelo logra una precisión de 0.9811 y un Kappa de 0.9749, lo que refleja casi un excelente rendimiento y un alto grado de acuerdo entre las predicciones del modelo y las etiquetas reales.
```{r}
# Predición sobre los datos de test
predictions <- predict(knnModel, newdata = testData)

# Generamos la matriz de confusión
confusionMatrix(predictions,testData$Class)

# Calculamos la probabilidad estimada de pertenecer a cada clase (necessario para ROC y AUC)
probabilities <- predict(knnModel, newdata = testData, type = "prob")
```
El modelo k-NN empleado mostró un rendimiento elevado en la clasificación de las cinco categorías de muestras. La matriz de confusión indica que la mayoría de las muestras fueron correctamente clasificadas. Por ejemplo, todas las muestras de AGH, CHC y HPB se predijeron correctamente, mientras que en CFB se registró un falso negativo y un falso positivo, y en CGC un falso negativo, mostrando que los errores son muy pocos.

La exactitud global (Accuracy) del modelo fue del 96.86%, y el Kappa = 0,9581 evidencia un acuerdo elevado entre las predicciones y las etiquetas reales, considerando el azar.

Por clase, la mayoría de las métricas muestran sensibilidades y especificidades mayor al 95%, indicando que el modelo es capaz de discriminar con fiabilidad cada tipo de cáncer, pero con limitaciones en algunos casos. 
Graficamos la curva ROC para cada tipo de cáncer:
```{r}
library(pROC)
library(ggplot2)
library(patchwork)

# 1. Crear objeto ROC multiclase
roc_multi <- multiclass.roc(testData$Class, probabilities)

# 2. Extraer las curvas binarizadas (uno contra todos)
rocs <- roc_multi$rocs

# 3. Cada curva está dentro de una lista anidada
roc_A <- rocs[[1]][[1]]
roc_B <- rocs[[2]][[1]]
roc_C <- rocs[[3]][[1]]
roc_D <- rocs[[4]][[1]]# si tienes 3 clases
roc_E <- rocs[[5]][[1]]

#asegurarnos el orden de las clases
colnames(probabilities)
# 4. Crear gráficos individuales SIN bucles
pA <- ggroc(roc_A, colour = "blue", size = 1.2) +
  ggtitle("ROC k-NN: Clase AGH") +
  annotate("text", x = 0.6, y = 0.1,
           label = paste("AUC =", round(auc(roc_A), 3)),
           color = "blue", size = 5)

pB <- ggroc(roc_B, colour = "red", size = 1.2) +
  ggtitle("ROC k-NN: Clase CFB") +
  annotate("text", x = 0.6, y = 0.1,
           label = paste("AUC =", round(auc(roc_B), 3)),
           color = "red", size = 5)

pC <- ggroc(roc_C, colour = "green", size = 1.2) +
  ggtitle("ROC k-NN: Clase CGC") +
  annotate("text", x = 0.6, y = 0.1,
           label = paste("AUC =", round(auc(roc_C), 3)),
           color = "green", size = 5)
pD <- ggroc(roc_D, colour = "purple", size = 1.2) +
  ggtitle("ROC k-NN: Clase CHC") +
  annotate("text", x = 0.6, y = 0.1,
           label = paste("AUC =", round(auc(roc_D), 3)),
           color = "purple", size = 5)
pE <- ggroc(roc_E, colour = "yellow", size = 1.2) +
  ggtitle("ROC k-NN: Clase HPB") +
  annotate("text", x = 0.6, y = 0.1,
           label = paste("AUC =", round(auc(roc_E), 3)),
           color = "yellow", size = 5)

# 5. Combinar sin bucles
roc_knn_combined <- pA + pB + pC + pD + pE
roc_knn_combined


```
Las curvas ROC y los valores de AUC por clase confirman que el modelo k-NN posee una capacidad casi perfecta para diferenciar cada categoría, mostrando un rendimiento robusto y consistente incluso en un escenario multiclase.

Entre las ventajas del método K-NN destacan:
- La clasificación es local, adaptándose bien a estructuras complejas de los datos.
- Es relativamente robusto frente a pequeñas variaciones, siempre que los vecinos reflejen correctamente la estructura de cada clase.

Y las limitaciones:
- El rendimiento depende de la elección de k y de la escala de los datos, por lo que es imprescindible centrar y escalar las variables.
- Puede ser sensible a ruido y outliers, especialmente con valores pequeños de k.


4.2 LDA
El Análisis Discriminante Lineal (LDA) es un método supervisado que busca combinaciones lineales de las variables que maximizan la separación entre clases, permitiendo reducir la dimensionalidad y clasificar nuevas observaciones de forma eficiente.
Creamos un dataset para LDA usando los genes escalados y la clase:
```{r}
df_lda <- data.frame(
  Class = df_filter$Class,
  df_scaled
)
df_lda$Class <- as.factor(df_lda$Class)
```
Código de partición train/test (80/20 estratificada):
```{r}
set.seed(123)
idx_train <- createDataPartition(df_lda$Class, p = 0.80, list = FALSE)
train_df <- df_lda[idx_train, ]
test_df <- df_lda[-idx_train, ]
```
Entrenamos el modelo y predeccimos :
```{r}
lda_model <- lda(Class ~ ., data = train_df)
lda_pred_train <- predict(lda_model, newdata = train_df)
lda_pred_test <- predict(lda_model, newdata = test_df)
```
Miramos la matriz de confusión:
```{r}
confusion <- confusionMatrix(lda_pred_test$class, lda_pred_test$class)
print(confusion)
```
Como se puede observar, la tabla muestra las métricas de rendimiento del modelo LDA evaluado sobre el conjunto de test, desglosadas por cada una de las clases (AGH, CFB, CGC, CHC y HPB):

En términos generales, el modelo presenta un rendimiento excelente, con un valor de accuracy de 1 en prácticamente todas las métricas, lo que indica una alta capacidad discriminativa entre las distintas clases.

La sensibilidad mide la capacidad del modelo para identificar correctamente las muestras de cada clase. Los resultados muestran una sensibilidad perfecta (1.000) para las clases AGH, CGC, CHC y HPB, mientras que para CFB es ligeramente inferior (0.9833), lo que indica que solo un pequeño número de muestras de esta clase fue mal clasificado.

La especificidad evalúa la capacidad del modelo para rechazar correctamente las muestras que no pertenecen a una clase determinada. En este caso, todas las clases presentan valores muy cercanos a 1, lo que refleja que el modelo apenas confunde muestras de otras clases con la clase evaluada. La clase CHC presenta una especificidad ligeramente inferior (0.9924), aunque sigue siendo muy elevada.

El valor predictivo positivo indica la proporción de predicciones correctas entre las muestras clasificadas como pertenecientes a una clase. Todos los valores son iguales a 1, excepto en CHC (0.9643), lo que sugiere que una pequeña fracción de las muestras predichas como CHC pertenecen en realidad a otra clase.

El valor predictivo negativo es prácticamente perfecto para todas las clases, indicando que cuando el modelo descarta una clase, lo hace de forma correcta.

La exactitud balanceada combina sensibilidad y especificidad, siendo especialmente útil en contextos con clases desbalanceadas. Los valores obtenidos son muy elevados, con exactitud perfecta para AGH, CGC y HPB, y ligeramente inferior para CFB (0.9917) y CHC (0.9962).

Las métricas de prevalencia y tasa de detección reflejan la distribución desigual de las clases en el conjunto de datos, siendo CFB la clase más representada y HPB la menos frecuente. A pesar de este desbalance, el modelo mantiene un rendimiento elevado en todas las clases.

Graficamos:
```{r}
lda_proj <- lda_pred_train$x
lda_plot_df <- cbind(train_df[, "Class", drop = FALSE], as.data.frame(lda_proj))

if (ncol(lda_proj) >= 2) {
  
  ggplot(lda_plot_df, aes(LD1, LD2, color = Class)) +
    geom_point(size = 2.2, alpha = 0.85) +
    labs(
      title = "LDA - Proyección en funciones discriminantes (TRAIN)",
      x = "LD1",
      y = "LD2",
      color = "Clase"
    ) +
    theme_classic() +
    theme(plot.title = element_text(hjust = 0.5))
  
} else {
  
  ggplot(lda_plot_df, aes(LD1, 0, color = Class)) +
    geom_jitter(height = 0.06, size = 2.2, alpha = 0.85) +
    labs(
      title = "LDA - Proyección (solo LD1, TRAIN)",
      x = "LD1",
      y = "",
      color = "Clase"
    ) +
    theme_classic() +
    theme(
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      plot.title = element_text(hjust = 0.5)
    )
}
```

4.3 RANDOM FOREST
Random forest es el método más popular basado en el bagging y consiste crear y  combinar un gran número de árboles de decisión en un gran bosque para obtener una salida más robusta y confiable. Se implementó este algoritmo usando el siguente script:
```{r}
# Renombrar columnas automáticamente para evitar problemas de compatibilidad con los nombres de los genes
names(trainData) <- make.names(names(trainData))
names(testData)  <- make.names(names(testData))

# Entrenar modelo
set.seed(2026)
rf_model <- randomForest(Class ~ .,
                          data = trainData,
                          ntree = 500, # Suficiente número de árboles para estabilidad
                          mtry = 20) # Número de variables muestreadas al azar como candidatas en cada división
rf_model

```

Para evaluar el modelo, se realizó una matriz de confusión con las predicciones: 

```{r}
# Predicción
rf_pred <- predict(rf_model, newdata = testData)
cm_rf <- confusionMatrix(rf_pred, testData$Class)
cm_rf
```
Los resultados obtenidos en este algoritmo son casi excelentes, mostrando 5 clasificaciones incorrectas incorrecta. 

El resumen de las métricas resultantes pueden verse a continuación:

```{r}
metrics_rf <- round(cm_rf$byClass[, c("Precision", "Sensitivity", "Specificity", "F1")], 3)
metrics_rf
```

Como puede observarse, la presición, sensibilidad, especificidad y F1 son iguales o cercanos a 1 en la mayoria los casos, indicando la capacidad del modelo para clasificar los tipos de cáncer en base a la expresión génica. 


Los métodos de bagging, como lo es el Random Forest, tienen como ventaja que:

* Presentan una reducción de la varianza, lo que puede mejorar la capacidad de generalización y hacer que las predicciones sean más estables;
* Al promediar o combinar las predicciones de varios modelos, pueden mejorar la precisión y el rendimiento del modelo final;
* Tienden a ser menos sensibles a los datos atípicos, lo que ayuda a mejorar la robustez del modelo.

Por otro lado, estos métodos:

* Suelen ser computacionalmente más costosos;
* Pueden ser más dificiles de interpretar;
* Presentan un riesgo de sobreajuste cuando los modelos de base individuales están sobreajustados a los datos de entrenamiento.

Los resultados obtenidos confirman que la elección de estos tres enfoques fue adecuada para el problema planteado. Los modelos mostraron un rendimiento muy elevado en la clasificación de los cinco tipos de cáncer, con métricas cercanas a la perfección en el conjunto de prueba. Esto indica que los perfiles de expresión génica contienen una señal discriminativa clara y que los métodos seleccionados, pese a sus diferencias conceptuales, son capaces de capturarla de forma eficaz y consistente.


5. DEEP LEARNING
Aunque para esta actividad no se aplicará ningún método de deep learning, de tener que elegir un tipo de arquetectura, seleccionariamos una red de perceptrones (MLP). Estas redes están diseñadas para datos tabulates, donde cada gen se modela como una variable independiente de entrada. No requieren estructura adicional que no se encuentre presente en nuestro dataframe. Además, capturan relaciones no lineales entre genes, como hemos demostrtado que lo exige el caso en estudio.


CONCLUSIÓN DE LA ACTIVIDAD: 
En este trabajo se han implementado de forma razonada y sistemática diversas técnicas de aprendizaje no supervisado y supervisado con el objetivo de explorar, distinguir e identificar distintos tipos de cáncer a partir de perfiles de expresión génica. Los métodos no supervisados permitieron analizar la estructura intrínseca de los datos y evaluar la existencia de patrones naturales de agrupamiento, mientras que los métodos supervisados posibilitaron la construcción de modelos predictivos capaces de discriminar entre los cinco tipos de cáncer considerados.

Cada una de las técnicas aplicadas fue analizada en profundidad, evaluando sus supuestos, ventajas y limitaciones, así como su capacidad para separar las distintas clases. En el caso de los métodos supervisados, el rendimiento de los modelos se evaluó mediante métricas adecuadas, como la matriz de confusión, precisión, sensibilidad, especificidad y el estadístico F1.

Desde una perspectiva clínica, los enfoques propuestos podrían constituir una herramienta complementaria de apoyo al diagnóstico, al facilitar la clasificación de muestras en función de su perfil de expresión génica. No obstante, aunque los modelos muestran un rendimiento satisfactorio en el conjunto de prueba, su posible aplicación en un entorno real requeriría una validación adicional en cohortes independientes y más heterogéneas, así como una evaluación de su robustez frente a variaciones técnicas y biológicas.
