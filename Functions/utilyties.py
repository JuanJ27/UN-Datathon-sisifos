import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Función para mostrar información básica de cada DataFrame
def explorar_dataframe(df, nombre):
    print(f"\nInformación del {nombre}:")
    display(df.head())
    print("\nDescripción estadística:")
    display(df.describe())

# Función para calcular y mostrar la matriz de correlación
def analizar_correlacion(df, nombre):
    # Seleccionar solo columnas numéricas
    numeric_cols = df.select_dtypes(include=['float64', 'int64', 'int32']).columns
    # Calcular matriz de correlación
    corr_matrix = df[numeric_cols].corr()
    #print(f"\nMatriz de correlación para {nombre}:")
    #display(corr_matrix)
    # Visualización con mapa de calor
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title(f'Matriz de correlación para {nombre}')
    plt.show()

# Ejemplo con EsperanzaVida (asumiendo que tiene datos temporales de 2018 a 2023)
def analizar_tendencias(df, nombre, valor_columnas, categoria_columnas=None, figsize=(10, 6)):
    """
    Grafica tendencias temporales utilizando Seaborn para mayor eficiencia y estética.
    
    Parametros:
    - df: DataFrame con datos temporales.
    - nombre: Nombre descriptivo del DataFrame (para títulos de gráficos).
    - valor_columnas: Lista de nombres de columnas que representan los valores a graficar.
    - categoria_columnas: Lista de nombres de columnas para categorizar las series (opcional).
    - figsize: Tupla que define el tamaño de la figura (ancho, alto).
    """
    plt.figure(figsize=figsize)
    if categoria_columnas:
        # Supone que 'categoria_columnas' contiene una sola columna para la categorización
        if len(categoria_columnas) > 1:
            raise ValueError("Sólo se soporta una columna para 'categoria_columnas'.")
        categoria = categoria_columnas[0]
        melted_df = df.melt(id_vars=[categoria], value_vars=valor_columnas, var_name='Año', value_name='Valor')
        sns.lineplot(data=melted_df, x='Año', y='Valor', hue=categoria)
    else:
        melted_df = df.melt(value_vars=valor_columnas, var_name='Año', value_name='Valor')
        sns.lineplot(data=melted_df, x='Año', y='Valor', estimator=None)
    plt.title(f'Tendencia temporal en {nombre}')
    plt.xlabel('Año')
    plt.ylabel('Valor')
    plt.legend(title=categoria if categoria_columnas else None, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


# Ejemplo de visualización geoespacial para SedesEducativas (establecimientos educativos)
def visualizar_mapa(df, nombre, columna_categoria=None, figsize=(10, 10), cmap='OrRd', legend_kwds=None):
    """
    Visualiza un GeoDataFrame en un mapa, opcionalmente coloreado por una categoría.
    
    Parámetros:
    - df: GeoDataFrame a visualizar.
    - nombre: Nombre descriptivo del GeoDataFrame (para títulos de gráficos).
    - columna_categoria: Nombre de la columna para colorear las geometrías (opcional).
    - figsize: Tupla que define el tamaño de la figura (ancho, alto).
    - cmap: Esquema de colores para la visualización.
    - legend_kwds: Diccionario de parámetros para la leyenda.
    """
    plt.figure(figsize=figsize)
    if columna_categoria:
        df.plot(column=columna_categoria, legend=True, cmap=cmap, legend_kwds=legend_kwds)
    else:
        df.plot()
    plt.title(f'Mapa geoespacial de {nombre}')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.tight_layout()
    plt.show()


def graficar_correlacion_dos_df(df1, df2, columnas_df1=None, columnas_df2=None, clave_primaria=None, nombre_df1='DataFrame1', nombre_df2='DataFrame2'):
    """
    Grafica la matriz de correlación entre las columnas numéricas de dos DataFrames.

    Parámetros:
    - df1: Primer DataFrame.
    - df2: Segundo DataFrame.
    - columnas_df1: Lista de columnas numéricas del primer DataFrame a incluir en el análisis. Si es None, se incluyen todas las numéricas.
    - columnas_df2: Lista de columnas numéricas del segundo DataFrame a incluir en el análisis. Si es None, se incluyen todas las numéricas.
    - clave_primaria: Columna o lista de columnas para alinear los DataFrames (por ejemplo, 'codigo' o ['codigo', 'nombre']).
    - nombre_df1: Nombre descriptivo para el primer DataFrame (para etiquetas en el gráfico).
    - nombre_df2: Nombre descriptivo para el segundo DataFrame.

    Retorna:
    - Matriz de correlación entre las columnas seleccionadas de df1 y df2.
    """

    # Renombrar columnas de df1 y df2 (excepto la clave primaria) para incluir sufijos
    df1_renombrado = df1.copy()
    df2_renombrado = df2.copy()

    # Verificar si la clave primaria es una lista o una cadena
    if isinstance(clave_primaria, list):
        claves_primarias = clave_primaria
    else:
        claves_primarias = [clave_primaria]

    # Obtener columnas de df1 y df2 sin la clave primaria
    columnas_df1_sin_clave = df1.columns.difference(claves_primarias)
    columnas_df2_sin_clave = df2.columns.difference(claves_primarias)

    # Renombrar columnas
    df1_renombrado = df1_renombrado.rename(columns={col: f"{col}_{nombre_df1}" for col in columnas_df1_sin_clave})
    df2_renombrado = df2_renombrado.rename(columns={col: f"{col}_{nombre_df2}" for col in columnas_df2_sin_clave})

    # Fusionar los DataFrames en base a la clave primaria
    if clave_primaria is not None:
        df_comb = pd.merge(df1_renombrado, df2_renombrado, on=clave_primaria)
    else:
        df_comb = pd.concat([df1_renombrado, df2_renombrado], axis=1)

    # Seleccionar columnas numéricas si no se especifican
    if columnas_df1 is None:
        columnas_df1 = df1.select_dtypes(include=['float64', 'int64', 'int32']).columns.tolist()
    if columnas_df2 is None:
        columnas_df2 = df2.select_dtypes(include=['float64', 'int64', 'int32']).columns.tolist()

    # Añadir sufijos a los nombres de las columnas para identificar el origen
    columnas_df1_pref = [f"{col}_{nombre_df1}" for col in columnas_df1 if col not in claves_primarias]
    columnas_df2_pref = [f"{col}_{nombre_df2}" for col in columnas_df2 if col not in claves_primarias]

    # Seleccionar las columnas de interés del DataFrame combinado
    df_corr = df_comb[columnas_df1_pref + columnas_df2_pref]

    # Eliminar filas con valores nulos
    df_corr = df_corr.dropna()

    # Calcular la matriz de correlación
    corr_matrix = df_corr.corr()

    # Extraer sólo las correlaciones entre df1 y df2
    corr_submatrix = corr_matrix.loc[columnas_df1_pref, columnas_df2_pref]

    # Visualizar la matriz de correlación
    plt.figure(figsize=(6, 6))
    sns.heatmap(corr_submatrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f'Matriz de correlación entre {nombre_df1} y {nombre_df2}')
    plt.xlabel(nombre_df2)
    plt.ylabel(nombre_df1)
    plt.show()

    return corr_submatrix

def contar_puntos_en_poligonos(gdf_puntos, gdf_poligonos, columna_poligono_id='id_poligono'):
    """
    Cuenta cuántos puntos están dentro de cada polígono.

    Parámetros:
    - gdf_puntos: GeoDataFrame con geometrías de puntos.
    - gdf_poligonos: GeoDataFrame con geometrías de polígonos.
    - columna_poligono_id: Nombre de la columna en gdf_poligonos que identifica de manera única cada polígono.

    Retorna:
    - GeoDataFrame con el conteo de puntos dentro de cada polígono.
    """

    # Asegurarse de que ambos GeoDataFrames estén en el mismo sistema de coordenadas
    if gdf_puntos.crs != gdf_poligonos.crs:
        gdf_puntos = gdf_puntos.to_crs(gdf_poligonos.crs)

    # Realizar un 'spatial join' para asignar cada punto al polígono correspondiente
    puntos_en_poligonos = gpd.sjoin(
        gdf_puntos, 
        gdf_poligonos[[columna_poligono_id, 'geometry']], 
        how='left', 
        predicate='within'  # Reemplazo de 'op' por 'predicate'
    )

    # Contar cuántos puntos hay en cada polígono
    conteo = puntos_en_poligonos.groupby(columna_poligono_id).size().reset_index(name='conteo_puntos')

    # Combinar el conteo con el GeoDataFrame de polígonos
    resultado = gdf_poligonos.merge(conteo, on=columna_poligono_id, how='left')

    # Reemplazar valores nulos por cero (para polígonos sin puntos dentro)
    resultado['conteo_puntos'] = resultado['conteo_puntos'].fillna(0).astype(int)

    return resultado

def contar_puntos_en_poligonos_detallado(gdf_puntos, gdf_poligonos, columna_poligono_id='id_poligono', columna_punto_id='id_punto'):
    """
    Cuenta cuántos puntos están dentro de cada polígono y proporciona detalles de los puntos.

    Parámetros:
    - gdf_puntos: GeoDataFrame con geometrías de puntos.
    - gdf_poligonos: GeoDataFrame con geometrías de polígonos.
    - columna_poligono_id: Nombre de la columna en gdf_poligonos que identifica de manera única cada polígono.
    - columna_punto_id: Nombre de la columna en gdf_puntos que identifica de manera única cada punto.

    Retorna:
    - GeoDataFrame de polígonos con el conteo de puntos y una lista de IDs de puntos dentro de cada polígono.
    """

    # Asegurar que ambos GeoDataFrames están en el mismo CRS
    if gdf_puntos.crs != gdf_poligonos.crs:
        gdf_puntos = gdf_puntos.to_crs(gdf_poligonos.crs)

    # Realizar un 'spatial join' para asignar cada punto al polígono correspondiente
    puntos_en_poligonos = gpd.sjoin(gdf_puntos[[columna_punto_id, 'geometry']], gdf_poligonos[[columna_poligono_id, 'geometry']], how='left', predicate='within')

    # Agrupar por polígono y obtener listas de IDs de puntos
    agrupado = puntos_en_poligonos.groupby(columna_poligono_id).agg({
        columna_punto_id: list
    }).reset_index()

    # Calcular el conteo de puntos
    agrupado['conteo_puntos'] = agrupado[columna_punto_id].apply(len)

    # Combinar con el GeoDataFrame de polígonos
    resultado = gdf_poligonos.merge(agrupado, on=columna_poligono_id, how='left')

    # Reemplazar valores nulos por cero y listas vacía
    # Agrupars
    resultado['conteo_puntos'] = resultado['conteo_puntos'].fillna(0).astype(int)
    resultado[columna_punto_id] = resultado[columna_punto_id].apply(lambda x: x if isinstance(x, list) else [])

    return resultado

def evolucionTemporal(*dfCols, primerAño, ultmioAño, pasoAño=1, titulo, xlabel, ylabel, figsize=(10, 6)):
    for i in range(primerAño, ultmioAño, pasoAño):
        plt.figure(figsize=figsize)
        for dfCol in dfCols:
            plt.plot(dfCol.loc[i], label=dfCol.loc[i].name)
        plt.title(titulo)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()


def pivot_comunas_indice(df, tipo_indice='total', prefijo_columna='total_', columna_nombre='nombre'):
    """
    Transforma el GeoDataFrame para que cada comuna tenga una columna con sus índices por año.

    Parámetros:
    - df: GeoDataFrame original.
    - tipo_indice: Tipo de índice a utilizar (e.g., 'total', 'hombres', 'mujeres').
    - prefijo_columna: Prefijo de las columnas que contienen los años (e.g., 'total_', 'hombres_').
    - columna_nombre: Nombre de la columna que contiene los nombres de las comunas (default='nombre').

    Retorna:
    - DataFrame pivotado con columnas por comuna y filas por año.
    """
    
    # 1. Identificar las columnas que corresponden al tipo de índice y a los años
    columnas_año = [col for col in df.columns if col.startswith(prefijo_columna)]
    
    if not columnas_año:
        raise ValueError(f"No se encontraron columnas que comiencen con el prefijo '{prefijo_columna}'. Verifica el prefijo proporcionado.")
    
    # 2. Transformar el DataFrame a formato largo
    df_melt = df.melt(
        id_vars=[columna_nombre],
        value_vars=columnas_año,
        var_name='año',
        value_name='indice'
    )
    
    # 3. Extraer el año numérico de las columnas
    df_melt['año'] = df_melt['año'].str.replace(prefijo_columna, '').astype(int)
    
    # 4. Pivotar el DataFrame para tener comunas como columnas
    df_pivot = df_melt.pivot_table(
        index='año',
        columns=columna_nombre,
        values='indice'
    ).reset_index()
    
    # 5. Ordenar el DataFrame por año
    df_pivot = df_pivot.sort_values('año').reset_index(drop=True)
    
    return df_pivot
                   