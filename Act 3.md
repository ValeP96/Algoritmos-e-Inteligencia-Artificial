# Actividad 3. Análisis de un conjunto de datos de origen biológico mediante técnicas de machine learning supervisadas y no supervisadas

## Entorno

```{r}
library(tidyverse)
library(skimr)
```

## Procesamiento de los datos 

Creacion del dataframe:

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

Confirmo que no hay NA

```{r}
which(is.na(df))
```

Revición del df

```{r}
skim_sd0 <- skim_df %>%
  filter(skim_type == "numeric") %>%
  arrange(numeric.sd) %>%
  head()
```

Vemos que hay 3 genes con sd 0 y los eliminamos:

```{r}
genes_sd0 <- c("RPL22L1", "ZCCHC12", "MIER3")

df_filtered <- df %>%
  select(-all_of(genes_sd0))   # elimina los genes con solo 0

df_scaled <- df_filtered %>%
  select(-sample, -class) %>%   # solo expresión génica
  scale()                        # escala a media 0 y varianza 1
```

