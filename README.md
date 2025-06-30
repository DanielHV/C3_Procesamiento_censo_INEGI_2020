# C3_Procesamiento_Censo_INEGI_2020

Este repositorio contiene el código necesario para procesar variables del Censo de Población y Vivienda 2020 del INEGI, generando indicadores a partir de datos preprocesados a diferentes niveles territoriales: estatal, municipal y AGEB.

también incluye un script auxiliar que permite cargar el archivo resultante del procesamiento (.csv) directamente a una base de datos PostgreSQL, utilizando las credenciales definidas en un archivo .env.

⸻

## Uso del procesador principal

### Ejecución

Para ejecutar el procesador principal, utilice el siguiente comando desde la terminal:

```
python main_procesador.py --config ./config.json
```

Parámetros:

- `--config`: Archivo de configuración para el procesamiento a realizar.

### Estructura del archivo config.json

A continuación se describe la estructura esperada del archivo config.json utilizado como entrada:

#### Archivos de entrada

```
{
    "data_state" : "../data/preprocessed/state_cpv2020.csv",
    "data_mun" : "../data/preprocessed/mun_cpv2020.csv",
    "data_ageb" : "../data/preprocessed/ageb_cpv2020.csv",
    "dict_data" : "../data/raw/iter_00_cpv2020/diccionario_datos/diccionario_datos_iter_00CSV20.csv",

```

- `data_state`: Ruta al archivo CSV con datos filtrados a nivel estatal.
- `data_mun`: Ruta al archivo CSV con datos filtrados a nivel municipal.
- `data_ageb`: Ruta al archivo CSV con datos filtrados a nivel AGEB.
- `dict_data`: Ruta al archivo CSV con el diccionario de datos original proporcionado por INEGI. Este se usa para mapear descripciones a variables.

#### Variables a procesar (por lista explícita)

```
"variables_a_procesar_list" : {
    "POBTOT" : ["OCUPVIVPAR"],
    "TOTHOG" : ["HOGJEF_F"],
    "None" : ["REL_H_M"]
},
```

- En este bloque se indica qué variables procesar dentro de listas como valor.
- Las claves especifican la variable base de normalización. Si se usa "None", las variables encontradas se procesan sin normalizar.

En este ejemplo, se procesaría lo siguiente:

- `OCUPVIVPAR / POBTOT`
- `HOGJEF_F / TOTHOG`
- `REL_H_M` (sin normalizar)

#### Variables a procesar (por expresiones regulares)

```
"variables_a_procesar_regex" : {
    "POBTOT" : "^(?!.*TOT)(?!.*PROM)(?!.*PRO_OCUP)P.*",
    "TVIVPARHAB" : "^VPH.*"
},
```

- En este bloque, en lugar de enlistar variables manualmente como valor, se pueden seleccionar mediante expresiones regulares como valor.
- Las claves especifican la variable base de normalización. Si se usa "None", las variables encontradas se procesan sin normalizar.

En este ejemplo, se procesaría lo siguiente:

- Todas las variables que comiencen con `P` pero no contengan `TOT`, `PROM` o `PRO_OCUP`, divididas entre `POBTOT`.
- Las variables que comiencen con `VPH`, divididas entre `TVIVPARHAB`.

#### Número de categorías

```
    "q" : 10,
```

- Especifica el número de categorías (rangos) a generar para cada variable.
- Si existen valores no numéricos, se asignará la categoría especial "Sin categoría".

#### Ruta de salida

```
    "ruta_salida" : "./output/procesamiento.csv"
}
```

- Ruta donde se guardará el archivo resultante del procesamiento.

### Salida

El resultado será un archivo CSV que incluye las variables procesadas (normalizadas o no), categorizadas en rangos, listo para análisis posterior.

⸻

## Carga de datos procesados a base de datos

### Ejecución

Para ejecutar el script de carga de datos a la base de datos, utilice el siguiente comando desde la terminal:

```
python cargar_a_db.py --ruta-datos-procesados ./output/procesamiento.csv --ruta-env ./credenciales.env --crear-tabla
```

Parámetros:

- `--ruta-datos-procesados`: Ruta al archivo CSV generado por el procesamiento que se desea cargar en la base de datos.
- `--ruta-env` (opcional): Ruta al archivo .env que contiene las credenciales y parámetros de conexión a la base de datos. Por defecto: `./.env`.
- `--crear-tabla` (opcional): Si se incluye esta bandera, el script creará automáticamente la tabla de destino en caso de que no exista.

### Estructura del archivo .env

El archivo .env debe contener las siguientes variables:

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=nombre_de_base
DB_USER=usuario
DB_PASSWORD=contraseña
DB_TABLE=nombre_de_tabla
```

