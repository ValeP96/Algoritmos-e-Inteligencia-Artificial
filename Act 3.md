# Actividad 3. Análisis de un conjunto de datos de origen biológico mediante técnicas de machine learning supervisadas y no supervisadas

## Introducción

El presente análisis se centra en un dataset que recopila información sobre la expresión de cientos de genes en muestras asociadas a cinco tipos distintos de cáncer:

* AGH
* CHC
* HPB: Hiperplasia Benigna de Próstata
* CGC
* CFB           

El objetivo principal es implementar de forma razonada técnicas de aprendizaje supervisado y no supervisados para detectar patrones intrínsecos en el conjunto de datos y evaluar la capacidad de estos métodos para distinguir e identificar los distintos tipos de cáncer en base a los distintos perfiles de expresión génica.

## Entorno

En primer lugar, se cargaron las siguientes librerías para poder llevar a cabo el análisis:

```{r}
# Manipulación, importación, exploración y visualización de datos
library(tidyverse) 
library(skimr) 

# Graficos
library(ggplot2)
library(pheatmap)

# Métodos de aprendizaje no supervisados
library(RDRToolbox)
library(cluster)
library(stats)
# Métodos de aprendizaje supervisados
library(caret)
library(randomForest)
```

## Procesamiento de los datos 

Se importaron los tres archivos provistos y se generó un único dataframe.

```{r}
classes <- read_delim("classes.csv", delim = ";", escape_double = FALSE,  col_names = FALSE, trim_ws = TRUE)

gene_exp <- read_delim("gene_expression.csv", delim = ";", escape_double = FALSE, col_names = FALSE, trim_ws = TRUE)
gene_exp[] <- lapply(gene_exp, as.numeric)

column_names <- read.table("column_names.txt", quote="\"", comment.char="")

colnames(gene_exp) <- column_names[[1]]

df <- cbind(classes, gene_exp)
colnames(df)[1:2] <- c("sample", "class")
df$class <- factor(df$class)
df <- as.data.frame(df)
```

Tras una inspección inicial del conjunto de datos, se confirmó la ausencia de valores perdidos (NA) en todas las variables:

```{r}
which(is.na(df))
```
Por lo tanto, no fue necesario aplicar ningún método de imputación.

Luego, se utilizó la librería skim para realizar una rápida exploración de los datos, incluyendo el calculo de estadísticos descriptivos:

```{r}
skim_sd0 <- skim_df %>%
  filter(skim_type == "numeric") %>%
  arrange(numeric.sd) %>%
  head()
```

Se identificaron 3 genes con desviación estándar igual a cero (sd = 0), es decir, genes cuya expresión es constante en todas las muestras. Estos genes fueron eliminados del análisis, ya que no aportan información discriminativa y pueden causar problemas numéricos en métodos como PCA, LDA o clustering.

```{r}
genes_sd0 <- c("RPL22L1", "ZCCHC12", "MIER3")

df_filter <- df %>%
  select(-all_of(genes_sd0))   # elimina los genes con solo 0
```

Finalmente, los datos de expresión génica fueron escalados a media cero y varianza unitaria:

```{r}
df_scaled <- df_filter %>%
  select(-sample, -class) %>%   # solo expresión génica
  scale()                       # escala a media 0 y varianza 1
```

El escalado elimina diferencias de magnitud entre genes, reduce el impacto de valores altos, y es esencial para alguno de los métodos utilizados a continuación. 
##HEATMAP
Hacemos un heatmap para ver la expresion de los genes
```{r}
varianza <- apply(df_numeric, 2, var, na.rm = TRUE) #nos quedamos con los que mayor varianza tengan
top_genes100 <- names(sort(varianza, decreasing = TRUE))[1:100] #acotamos primero a 100 genes, y probamos
matriz_genes <- df_numeric[, top_genes100] #matriz de 100 genes
top_genes2 <- names(sort(varianza, decreasing = TRUE))[1:50] #despues acotamos finalmente a matriz de 50 genes
matriz_genes2 <- df_numeric[, top_genes2] #creamos la matriz de 50 genes 
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

## Métodos no supervisados 

### Técnicas de reducción de dimensionalidad

Se implementaron y evaluaron cuatro métodos de aprendizaje no supervisado de reducción de dimensionalidad: PCA, isomap, LLE y t-SNE.

•	PCA → Carla
•	Isomap → Anna
•	LLE → Valeria
•	t-SNE → Judit

#### PCA
```{r}
library(stats)
library(ggplot2)
#vamos a utilizar la base de datos eescaladas, pero tengo que añadirle la columna de Class de la base de datos ya filtrada
df_scaled_class <- cbind(df_scaled, df_filter[, 2, drop = FALSE])

pca_resultados <- prcomp(df_scaled, center = TRUE, scale. = FALSE) #hago con data_final_heatmap, en vez de data_final porque es la que contiene solamente datos numéricos

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

#### Isomap

#### Locally linear embedding (LLE)

El LLE es un algoritmo altamente eficiente en descubrir las estructuras no lineales en los datos y preservar las distancias dentro del vecindario local. El LLE presenta ciertas limitaciones, pues es muy sensible al ruido y a los valores atípicos. Además, es posible que dos puntos de datos, que en realidad no se encuentren en el mismo parche localmente lineal, sean agrupados como si lo estuvieran.

El número de vecinos, k, es su único parámetro libre, lo que simplifica el ajuste el algoritmo. En esta ocación fueron evaluados varios valores de k entre 10 y 80, seleccionando 50 como el óptimo. El algoritmo se implementó de la siguiente manera:

```{r}
lle.results.50 <- LLE(df_scaled, dim = 2, k = 50) # La dimensión final es 2, ya que son las dos que se van a representar gráficamente

lle.df.50 <- data.frame(lle.results.50) # Creación de un dataframe a partir de los resultados

lle.df.50$class <- df$class # Incorporación de la columna especificando los tipos de cáncer
```

Para la representación gráfica, se utilizó el siguiente código:

```{r}
plot_LLE_50 <- ggplot(lle.df.50, aes(x = X1, y = X2, color = class)) +
  geom_point(size = 3) +
  scale_color_manual(values = rainbow(5)) +
  labs(title = "Método LLE", x = "X1", y = "X2") +
  theme_classic()
```

Este metodo permite una clara distinción de los tipos de cáncer AGH, HPB y CHC, aunque se superponen aún CGC y CHC. Es evidetne la ventaja de utilizar, para este dataframe en particular, un método no lineal.

#### t-SNE




Como puede observarse en el análisis anterior, la mejor separación de los tipos de cáncer se obtuvo con t-SNE, seguido de LLE. Ambas presentan un manejo eficiente de datos no lineales, pero el t-SNE es claramente superior a la hora de identificar y revelar la importante estructura global.

### Técnicas de clusterización
k-means -> Ana
•	Jerarquico → Carla


```

TÉCNICAS DE CLUSTERIZACIÓN:
1) CLUSTERIZACION AGLOMERATIVA
Como 800 genes no nos da resultado de nada, vamos a hacer las mismas técnicas con los 50 genes selecionados con mas varianza
Reduzco a 50 genes.
```{r}
top_genes50 <- names(sort(varianza, decreasing = TRUE))[1:50]
matriz_genes50 <- df_filter[, top_genes50]
View(matriz_genes50)
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

#graficamos resultados

plot(hclust_single50, main = "Single Linkage")
plot(hclust_complete50, main = "Complete Linkage")
plot(hclust_average50, main = "Average Linkage")
plot(hclust_ward50, main = "Ward Linkage")

#por colores y con la funcion fviz_den para hacer el dendograma

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
dendogramas_todos
```
2) CLUSTERIZACION DECISIVA: DIANA
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
## Métodos supervisados 

Los métodos supervisados necesitan ser puestos a prueba y evaluados para confirmar que funcione correctamente con datos nuevos y que pueda generar resultados precisos. Luego, el conjunto de datos debe ser dividido en datos de entrenamiento y datos de prueba. En esta ocación utilizamos el 80% de los datos para el entrenamiento, y el 20% restante para la prueba, aplicando el siguiente código:

```{r}
set.seed(2026) # Reproducibilidad
train_index <- createDataPartition(df_filtered$class, p = 0.8, list = FALSE) # 80% de prueba

## Dividimos datos en entrenamiento y test según los índices.
train_data <- df_filtered[train_index, ]
test_data <- df_filtered[-train_index, ]

## Extraemos las variables numéicas para el conjuno de entrenamiento
train_num <- train_data[, !(names(train_data) %in% c("sample", "class"))]

## Escalamos usando SOLO entrenamiento
scale_params <- preProcess(train_num, method = c("center", "scale"))
train_scaled <- predict(scale_params, train_num)

## Reconstruimos dataset de entrenamiento
train_scaled <- as.data.frame(train_scaled)
train_data_scaled <- cbind(train_scaled, class = train_data$class)

## Extraemos las variables numéicas para el conjuno de test
test_num <- test_data[, !(names(test_data) %in% c("sample", "class"))]

## Aplicamos los MISMOS parámetros al test
test_scaled <- predict(scale_params, test_num)

## Reconstruimos dataset de test
test_scaled <- as.data.frame(test_scaled)
test_data_scaled <- cbind(test_scaled, class = test_data$class)
```


•	K-NN -> Judit
•	LDA -> Ana
•	Random forest -> Valeria

#### Random forest

Random forest es el método más popular basado en el bagging y consiste crear y  combinar un gran número de árboles de decisión en un gran bosque para obtener una salida más robusta y confiable. Se implementó este algoritmo usando el siguente script:

```{r}
# Renombrar columnas automáticamente para evitar problemas de compatibilidad con los nombres de los genes
names(train_data_scaled) <- make.names(names(train_data_scaled))
names(test_data_scaled)  <- make.names(names(test_data_scaled))

# Entrenar modelo
set.seed(2026)
rf_model <- randomForest(class ~ .,
                          data = train_data_scaled,
                          ntree = 500, # Suficiente número de árboles para estabilidad
                          mtry = 20) # Número de variables muestreadas al azar como candidatas en cada división

# Predicción
rf_pred <- predict(rf_model, newdata = test_data_scaled)
```

Para evaluar el modelo, se realizó una matriz de confusióm: 
```{r}
cm_rf <- confusionMatrix(rf_pred, test_data_scaled$class)
cm_rf
```
Los resultados obtenidos en este algoritmo son excelentes, mostrando una única clasificación incorrecta. 

El resumen de las métricas resultantes pueden verse a continuación:

```{r}
metrics_rf <- round(cm_rf$byClass[, c("Precision", "Sensitivity", "Specificity", "F1")], 3)
metrics_rf
```

Como puede observarse, la presición, sensibilidad, especificidad y F1 son iguales o cercanos a 1 en todos los casos, indicando la excelente capacidad del modelo para clasificar los tipos de cáncer en base a la expresión génica. 

Los métodos de bagging tienen como ventaja que:

* Presentan una reducción de la varianza, lo que puede mejorar la capacidad de generalización y hacer que las predicciones sean más estables;
* Al promediar o combinar las predicciones de varios modelos, pueden mejorar la precisión y el rendimiento del modelo final;
* Tienden a ser menos sensibles a los datos atípicos, lo que ayuda a mejorar la robustez del modelo.

Por otro lado, estos métodos:

* Suelen ser computacionalmente más costosos;
* Pueden ser más dificiles de interpretar;
* Presentan un riesgo de sobreajuste cuando los modelos de base individuales están sobreajustados a los datos de entrenamiento.

## Deep learning

Aunque para esta actividad no se aplicará ningún método de deep learning, de tener que elegir un tipo de arquetectura, seleccionariamos una red de perceptrones (MLP). Estas redes están diseñadas para datos tabulates, donde cada gen se modela como una variable independiente de entrada. No requieren estructura adicional que no se encuentre presente en nuestro dataframe. Además, capturan relaciones no lineales entre genes, como hemos demostrtado que lo exige el caso en estudio.
