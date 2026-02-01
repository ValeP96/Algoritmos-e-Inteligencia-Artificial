# Actividad 3. Análisis de un conjunto de datos de origen biológico mediante técnicas de machine learning supervisadas y no supervisadas

## Introducción

El presente análisis se centra en un dataset que recopila información sobre la expresión de cientos de genes en muestras asociadas a cinco tipos distintos de cáncer:

* AGH
* CHC
* HPB
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
library(patchwork)

# Métodos de aprendizaje no supervisados
library(RDRToolbox)
library(cluster)
library(stats)
library(Rtsne)

# Métodos de aprendizaje supervisados
library(caret)
library(pROC)
library(randomForest)
library(MASS)

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

#### Heatmap
Para tener una visión de la expresión de los genes se hace un heatmap, en este caso solo de 50 genes para que sea más clara la visualización.
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
```

## Métodos no supervisados 

### Técnicas de reducción de dimensionalidad

Se implementaron cuatro métodos no supervisados de reducción de dimensionalidad (PCA, Isomap, LLE y t-SNE) con el objetivo de explorar la estructura interna del conjunto de datos y evaluar su capacidad para revelar patrones y agrupamientos entre los distintos tipos de cáncer. Estos métodos fueron seleccionados por ofrecer enfoques complementarios, tanto lineales como no lineales, adecuados para datos de expresión génica con relaciones complejas.

PCA se utilizó como método lineal de referencia para capturar la varianza global de los datos. Isomap se seleccionó por su capacidad para preservar la estructura global en datos no lineales, LLE por mantener las relaciones locales entre muestras, y t-SNE por su elevada capacidad para visualizar agrupamientos locales y separar clases en espacios de baja dimensión.

#### PCA
El objetivo principal del PCA (Análisis de Componentes Principales) es maximizar la varianza y reducir la dimensionalidad del conjunto de variables. Se caracteriza por combinar linealmente las variables originales y transformarlas en un nuevo conjunto de variables no correlacionadas, conocidas como componentes principales (PC).
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
ISOMAP es un algoritmo eficaz para descubrir estructuras no lineales en conjuntos de datos de alta dimensionalidad, ya que busca preservar las distancias geodésicas entre los puntos, es decir, las distancias a lo largo de la variedad subyacente. Este método extiende el Análisis de Componentes Principales (PCA) a contextos no lineales mediante la construcción de un grafo de vecinos y el cálculo de las distancias más cortas entre ellos.

La ventaja principal de este método es que es especialmente eficaz para conjuntos de datos cuyas variables mantienen relaciones no lineales. 

También presenta dos principales limitaciones:
Por un lado, se necesita una densidad suficiente de putos en el espacio de alta dimension para determinar sus distancias dandose el caso de que  las distancias en el espacio de menor dimensión podrían ser muy diferentes a las distancias en el espacio original.

Por otro lado, encontrar el parámetro k correcto, incluso si existe, es muy difícil. Si para el vecino más cercano es demasiado pequeño, es posible que el grafo ni siquiera esté conectado y, si es grande, el grafo podría ser demasiado denso, lo que nos lleva a determinar distancias incorrectas.

El número de vecinos está representado por el número k. En este caso, se escogió el valor de k=10 ya que nos permitía visualizar mejor graficamente las relaciones locales entre las diferentes clases. 

```{r}
set.seed(123)
X <- as.matrix(df_scaled[, -1])
y   <- df_filter$class

isomap.df <- data.frame(
  Dim1  = isomap.results$dim2[, 1],
  Dim2  = isomap.results$dim2[, 2],
  class = y
)
```
Para graficar, se utilizó el siguiente código:
```{r}
ggplot(isomap.df, aes(Dim1, Dim2, color = class)) +
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
El método t-distribuited Stochastic Neighbor Embedding (t-SNE) es una técnica de reducción de dimensionalidad no lineal que preserva las relaciones locales entre las muestras. Este método de análisis és especialmente útil cuando los datos no se distribuyan de forma lineal, como ocurre en la expresión génica.
```{r}
set.seed(123)
# Matriz numérica de estructura local con 30 vecinos (perplexity)
tsne <- Rtsne(as.matrix(df_scaled), perplexity = 30) 

# Dataframe con las dimensiones y las clases (tipos de cáncer)
tsne_df <- data.frame(Dim1 = tsne$Y[,1], Dim2 = tsne$Y[,2], Class = df_scaled_class$class) 

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
Como se observó en el análisis previo, la mejor separación entre los distintos tipos de cáncer se obtuvo con t-SNE, seguida de LLE. Ambos métodos manejan eficientemente la estructura no lineal de los datos, aunque t-SNE destaca por su capacidad para revelar patrones locales y la estructura global del dataset.

Entre las ventajas de t-SNE se incluyen:
- Visualización clara de agrupamientos locales incluso en datos con relaciones no lineales.
- Facilita la detección de subgrupos y patrones en un espacio reducido de dimensión.

Limitaciones a considerar:
- La interpretación de distancias globales entre grupos lejanos puede no reflejar similitudes reales.
- Los resultados dependen de parámetros como perplexity y número de iteraciones, requiriendo ajuste cuidadoso.
- No genera un modelo aplicable a nuevas muestras; se utiliza únicamente para exploración y visualización.

### Técnicas de clusterización
Se aplicaron distintos métodos de clusterización para identificar agrupamientos naturales en los datos de expresión génica. K-means se utilizó sobre la proyección Isomap 2D por su simplicidad y eficiencia, fijando el número de clústeres según las clases reales.

Además, se emplearon métodos jerárquicos aglomerativos (single, complete, average y Ward) y el método divisivo DIANA sobre los genes de mayor varianza para reducir ruido y explorar la estructura de los datos desde enfoques complementarios.

#### Técnicas de clusterización jerárquico (WARD) sobre isomap 
```{r}
set.seed(123)                                    # Fija la semilla para reproducibilidad

# Número de clusters = número de clases reales
k <- length(unique(y))                           # Calcula k como el nº de clases distintas

# Aplicación de K-means sobre las dos dimensiones de Isomap
km <- kmeans(
  isomap.df[, c("Dim1", "Dim2")],                # Datos 2D obtenidos con Isomap
  centers = k,                                   # Número de clusters
  nstart = 25                                    # Nº de inicializaciones aleatorias
)

# Añadir el clúster asignado a cada muestra
isomap.df$cluster_km <- factor(km$cluster)       # Convierte el clúster a factor

# Cálculo de la envolvente convexa para cada clúster
hulls <- isomap.df %>%
  group_by(cluster_km) %>%                       # Agrupa por clúster de K-means
  slice(chull(Dim1, Dim2)) %>%                   # Selecciona los puntos extremos
                                                  # que forman el polígono del clúster
  ungroup()                                      # Elimina la estructura de agrupación

# Gráfico de los resultados de K-means sobre Isomap
ggplot(isomap.df, aes(Dim1, Dim2, color = cluster_km)) +
  
  geom_polygon(                                  # Dibuja los polígonos de cada clúster
    data = hulls,                              
    aes(fill = cluster_km, group = cluster_km),  
    alpha = 0.2,                                
    color = NA                                   
  ) +
  
  geom_path(                                    
    data = hulls,
    aes(group = cluster_km),
    linewidth = 0.8                              
  ) +
  
  geom_point(                                    
    size = 2,                                   
    alpha = 0.85                                
  ) +
  
  labs(
    title = paste0("K-Means plot, centros = ", k),
    x = "Isomap - Dimensión 1",                     
    y = "Isomap - Dimensión 2",                     
    color = "Cluster",                            
    fill  = "Cluster"                              
  ) +
  
  theme_minimal()                              
```
El algoritmo K-means aplicado sobre la proyección Isomap 2D identifica cinco clústeres bien definidos, con escaso solapamiento y una clara diferenciación espacial entre los grupos.


#### Clusterización aglomerativa
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
#### Clusterización decisiva: DIANA
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
La elección de los métodos k-NN, LDA y Random Forest se realizó con el objetivo de comparar enfoques supervisados complementarios. El método k-NN se seleccionó por su simplicidad y por su capacidad para capturar relaciones locales entre muestras sin asumir una forma funcional previa de los datos. LDA se incluyó como un método estadístico clásico, adecuado para evaluar la separabilidad entre clases bajo supuestos de normalidad y varianzas homogéneas, sirviendo además como una referencia interpretable. Finalmente, Random Forest se empleó por su capacidad para modelar relaciones no lineales complejas, su robustez frente al ruido y su buen rendimiento en conjuntos de datos con un gran número de variables.

Los métodos supervisados deben ser evaluados para comprobar su capacidad de generalización y su rendimiento sobre datos no vistos. Para ello, el conjunto de datos se dividió en un conjunto de entrenamiento y otro de prueba, utilizando el 80% de las muestras para el entrenamiento y el 20% restante para la validación del modelo.

```{r}
set.seed(2026) # Reproducibilidad
train_index <- createDataPartition(df_filter$class, p = 0.8, list = FALSE) # 80% de prueba

## Dividimos datos en entrenamiento y test según los índices.
train_data <- df_filter[train_index, ]
test_data <- df_filter[-train_index, ]

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
#### K-NN
El método de aprendizaje supervisado k-NN se basa en los datos de entrenamiento durante la fase de predicción para asignar etiquetas a los datos no etiquetados. La predicción se realiza considerando las etiquetas de los vecinos más cercanos, de manera que un nuevo dato se clasifica según la mayoría de las etiquetas de sus k vecinos más próximos.
```{r}
set.seed(123)
# Entrenamos el modelo k-NN
knnModel <- train(class ~ .,
                  data = train_data_scaled,
                  method = "knn",
                  trControl = trainControl(method = "cv", number = 10),
                  tuneLength = 30
)
# Visualizamos los resultados del modelo entrenado y graficamos el rendimiento
knnModel
plot(knnModel)
```
En el output se presenta una tabla con los valores de precisión (Accuracy) y Kappa obtenidos para los distintos valores de k evaluados con los datos de entrenamiento. En el gráfico se observa que la máxima precisión se alcanza con k = 11 vecinos. Con este valor, el modelo logra una precisión de 0.998 y un Kappa de 0.998, lo que refleja un excelente rendimiento y un alto grado de acuerdo entre las predicciones del modelo y las etiquetas reales.
```{r}
# Predición sobre los datos de test
predictions <- predict(knnModel, newdata = test_data_scaled)

# Generamos la matriz de confusión
confusionMatrix(predictions,test_data_scaled$class)

# Calculamos la probabilidad estimada de pertenecer a cada clase (necessario para ROC y AUC)
probabilities <- predict(knnModel, newdata = test_data_scaled, type = "prob")
```
El modelo k-NN empleado mostró un rendimiento muy elevado en la clasificación de las cinco categorías de muestras. La matriz de confusión indica que la mayoría de las muestras fueron correctamente clasificadas. Por ejemplo, todas las muestras de AGH, CHC y HPB se predijeron correctamente, mientras que en CFB se registró un falso negativo y un falso positivo, y en CGC un falso negativo, mostrando que los errores son muy pocos.

La exactitud global (Accuracy) del modelo fue del 99,37%, y el Kappa = 0,992 evidencia un acuerdo casi perfecto entre las predicciones y las etiquetas reales, considerando el azar.

Por clase, las métricas muestran sensibilidades y especificidades cercanas al 100%, indicando que el modelo es capaz de discriminar con fiabilidad cada tipo de cáncer, con un número muy reducido de falsos positivos y falsos negativos.

Graficamos la curva ROC para cada tipo de cáncer:
```{r}
# Lista para guardar los ggplots de k-NN
roc_knn_plots <- list()

# Loop sobre cada clase
for (cls in levels(df_filter$class)) {
  
  # ROC one-vs-all para k-NN
  roc_knn_obj <- roc(as.numeric(test_data_scaled$class == cls), probabilities[, cls])
  
  # Dataframe para ggplot
  roc_knn_df <- data.frame(
    FPR = 1 - roc_knn_obj$specificities,
    TPR = roc_knn_obj$sensitivities
  )
  
  # Crear el gráfico de k-NN
  p_knn <- ggplot(roc_knn_df, aes(x = FPR, y = TPR)) +
    geom_line(color = "lightblue", size = 1.2) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
    theme_minimal() +
    labs(
      title = paste("ROC k-NN:", cls),
      x = "1 - Especificidad (FPR)",
      y = "Sensibilidad (TPR)"
    ) +
    annotate("text", x = 0.6, y = 0.1, 
             label = paste0("AUC = ", round(auc(roc_knn_obj), 3)),
             size = 5, color = "coral")
  
  roc_knn_plots[[cls]] <- p_knn
}

# Unir los gráficos k-NN en una sola figura
roc_knn_combined <- wrap_plots(roc_knn_plots, ncol = 3)
roc_knn_combined
```
Las curvas ROC y los valores de AUC por clase confirman que el modelo k-NN posee una capacidad casi perfecta para diferenciar cada categoría, mostrando un rendimiento robusto y consistente incluso en un escenario multiclase.

Entre las ventajas del método K-NN destacan:
- La clasificación es local, adaptándose bien a estructuras complejas de los datos.
- Es relativamente robusto frente a pequeñas variaciones, siempre que los vecinos reflejen correctamente la estructura de cada clase.

Y las limitaciones:
- El rendimiento depende de la elección de k y de la escala de los datos, por lo que es imprescindible centrar y escalar las variables.
- Puede ser sensible a ruido y outliers, especialmente con valores pequeños de k.

#### LDA
El Análisis Discriminante Lineal (LDA) es un método supervisado que busca combinaciones lineales de las variables que maximizan la separación entre clases, permitiendo reducir la dimensionalidad y clasificar nuevas observaciones de forma eficiente.

Entre las principales ventajas que tiene estan:
- Si el número de observaciones es bajo y la distribución de los predictores es aproximadamente normal en cada una de las clases, el LDA es más estable que la regresión logística.
- Si las clases están bien separadas, los parámetros estimados en el modelo de regresión logística son inestables. El método de LDA no sufre este problema.
- Cuando se trata de un problema de clasificación con solo 2 niveles, ambos métodos suelen llegar a resultados similares.

Creamos un dataset para LDA usando los genes escalados y la clase:
```{r}
df_lda <- data.frame(
  class = df_filter$class,
  df_scaled
)
df_lda$class <- as.factor(df_lda$class)
```
Código de partición train/test (80/20 estratificada):
```{r}
set.seed(123)
idx_train <- createDataPartition(df_lda$class, p = 0.80, list = FALSE)
train_df <- df_lda[idx_train, ]
test_df <- df_lda[-idx_train, ]
```
Entrenamos el modelo y predeccimos :
```{r}
lda_model <- lda(class ~ ., data = train_df)
lda_pred_train <- predict(lda_model, newdata = train_df)
```
Miramos la matriz de confusión:
```{r}
cm_lda <- confusionMatrix(lda_pred_test, test_df$class)

# Matriz de confusión
matriz_confusion <- cm_lda$table

# Métricas 
metricas <- cm_lda$byClass[, c("Precision", "Sensitivity", "Specificity", "F1")]
metricas <- round(metricas, 3)

matriz_confusion
metricas
```
El modelo LDA presenta un desempeño global sobresaliente, con valores de precisión, sensibilidad y especificidad iguales a 1.000 en la mayoría de las clases (AGH, CGC y HPB), lo que indica una clasificación perfecta sin falsos positivos ni falsos negativos. En la clase CFB, aunque la precisión y especificidad alcanzan el valor máximo (1.000), la sensibilidad desciende ligeramente a 0.983, reflejando una mínima pérdida en la detección de algunos casos reales. Por su parte, la clase CHC muestra una precisión de 0.964 y una especificidad de 0.992, manteniendo aun así una sensibilidad perfecta (1.000). A pesar de estas pequeñas variaciones, todos los valores F1-score son superiores a 0.98, lo que confirma un equilibrio excelente entre precisión y sensibilidad y evidencia la alta capacidad discriminante y robustez del modelo.

Graficamos:
```{r}
lda_proj <- lda_pred_train$x
lda_plot_df <- cbind(train_df[, "class", drop = FALSE], as.data.frame(lda_proj))

if (ncol(lda_proj) >= 2) {
  
  ggplot(lda_plot_df, aes(LD1, LD2, color = class)) +
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
  
  ggplot(lda_plot_df, aes(LD1, 0, color = class)) +
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

Esto tambien puede evidenciarse en las curvas ROC, las cuales fueron calculadas con sel siguiente script, mediante una estrategia one-vs-rest para cada clase, utilizando las probabilidades predichas por el modelo Random Forest. Los valores de AUC iguales a 1 demuestran también la excelente capacidad discriminativa de este modelo.

```{r}
test_x <- test_data_scaled[, !names(test_data_scaled) %in% "class"]

rf_prob <- predict(
  rf_model,
  newdata = test_x,
  type = "prob"
)
classes <- levels(test_data_scaled$class)

roc_list <- lapply(classes, function(cl) {
  roc(
    response = as.numeric(test_data_scaled$class == cl),
    predictor = rf_prob[, cl],
    quiet = TRUE
  )
})

names(roc_list) <- classes

plot(
  roc_list[[1]],
  col = 1,
  lwd = 2,
  main = "Curvas ROC – Random Forest (One-vs-Rest)"
)

for (i in 2:length(roc_list)) {
  plot(roc_list[[i]], col = i, lwd = 2, add = TRUE)
}

legend(
  "bottomright",
  legend = paste0(classes, " (AUC = ",
                  round(sapply(roc_list, auc), 3), ")"),
  col = 1:length(classes),
  lwd = 2,
  cex = 0.9
)
```

Los métodos de bagging, como lo es el Random Forest, tienen como ventaja que:

* Presentan una reducción de la varianza, lo que puede mejorar la capacidad de generalización y hacer que las predicciones sean más estables;
* Al promediar o combinar las predicciones de varios modelos, pueden mejorar la precisión y el rendimiento del modelo final;
* Tienden a ser menos sensibles a los datos atípicos, lo que ayuda a mejorar la robustez del modelo.

Por otro lado, estos métodos:

* Suelen ser computacionalmente más costosos;
* Pueden ser más dificiles de interpretar;
* Presentan un riesgo de sobreajuste cuando los modelos de base individuales están sobreajustados a los datos de entrenamiento.

## Deep learning

Aunque para esta actividad no se aplicará ningún método de deep learning, de tener que elegir un tipo de arquetectura, seleccionariamos una red de perceptrones (MLP). Estas redes están diseñadas para datos tabulates, donde cada gen se modela como una variable independiente de entrada. No requieren estructura adicional que no se encuentre presente en nuestro dataframe. Además, capturan relaciones no lineales entre genes, como hemos demostrtado que lo exige el caso en estudio.
