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

# Métodos de aprendizaje no supervisados

# Métodos de aprendizaje supervisados

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
## Métodos no supervisados 

### Técnicas de reducción de dimensionalidad

Se implementaron y evaluaron cuatro métodos de aprendizaje no supervisado de reducción de dimensionalidad: PCA, isomap, LLE y t-SNE.

•	PCA → Carla
•	Isomap → Anna
•	LLE → Valeria
•	t-SNE → Judit



Como puede observarse en el análisis anterior, la mejor separación de los tipos de cáncer se obtuvo con t-SNE, seguido de LLE.

### Técnicas de clusterización
k-means -> Ana
•	Jerarquico → Carla

HEATMAP
Hacemos un heatmap para ver la expresion de los genes
```{r}
library(pheatmap)
#Probamos a hacer con el dataset completo, pero son muchos genes, asi que para que sea visual, elegimos 50 genes con mayor varianza
top_genes2 <- names(sort(varianza, decreasing = TRUE))[1:50]
matriz_genes2 <- df_numeric[, top_genes2]
View(matriz_genes)
#hago el heatmap con estos 50 genes
heatmap50 <- pheatmap(
  matriz_genes,
  show_rownames = TRUE,
  show_colnames = TRUE,
  clustering_distance_rows = "euclidean",
  clustering_distance_cols = "euclidean",
  clustering_method = "complete"
)
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

•	K-NN -> Judit
•	LDA -> Ana
•	Random forest -> Valeria

