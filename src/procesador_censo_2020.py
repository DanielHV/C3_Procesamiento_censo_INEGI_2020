import os
import numpy as np
import pandas as pd
import re

def importar_conjunto_datos(ruta:str, version:str):
    """
    Importa un conjunto de datos del Censo INEGI 2020 y lo convierte en un DataFrame de pandas.

    Configura los tipos de datos de columnas específicas según la versión del conjunto de datos 
    ("iter" o "ageb") para mantener consistencia.

    Args:
        ruta (str): Ruta al archivo .csv que contiene el conjunto de datos.
        version (str): Versión del conjunto de datos. Puede ser:
            - "iter": Para conjuntos de datos a nivel estatal, municipal o de localidad, se configuran
                como cadenas las columnas 'ENTIDAD', 'MUN' y 'LOC'
            - "ageb": Para conjuntos de datos a nivel de AGEB (Área Geoestadística Básica), se configuran
                como cadenas las columnas 'ENTIDAD', 'MUN', 'LOC', 'AGEB' y 'MZA'

    Returns:
        pd.DataFrame: DataFrame con los datos importados y los tipos de datos configurados.

    Raises:
        ValueError: Si los parámetros no cumplen con las validaciones:
            - La ruta especificada no existe o no fue especificada.
            - El valor del parámetro 'version' no es "iter" o "ageb".
    """
    
    if not os.path.exists(ruta) or ruta is None:
        raise ValueError(f"No se encontró la ruta especificada para 'ruta': {ruta}")
    
    if version.lower() == "iter":
        dtype_dict = {
            "ENTIDAD": str,
            "MUN": str,
            "LOC": str
        }
    elif version.lower() == "ageb":
        dtype_dict = {
            "ENTIDAD": str,
            "MUN": str,
            "LOC": str,
            "AGEB": str,
            "MZA": str
        }
    else:
        raise ValueError("El valor del parámetro 'version' debe ser \"iter\" o \"ageb\"")
    
    return pd.read_csv(ruta, dtype=dtype_dict, low_memory=False)

    
def importar_diccionario_datos(ruta):
    """
    Importa un diccionario de datos del Censo INEGI 2020 que servirá para obtener las descripciones de las variables,
    y lo convierte en un diccionario de Python.

    Filtra las filas relevantes y mapea los valores de las columnas "Mnemónico" e "Indicador" en un diccionario.

    Args:
        ruta (str): Ruta al archivo .csv que contiene el diccionario de datos.

    Returns:
        dict: Diccionario donde las llaves son los valores de la columna "Mnemónico" y los valores son
        los correspondientes de la columna "Indicador".

    Raises:
        ValueError: Si:
            - La ruta especificada no existe o no fue especificada.
            - El conjunto de datos proporcionado no tiene las columnas requeridas para crear el diccionario 
    """
    
    if not os.path.exists(ruta) or ruta is None:
        raise ValueError(f"No se encontró la ruta especificada para 'ruta': {ruta}")
    
    diccionario_datos = pd.read_csv(ruta)
    
    mascara = diccionario_datos.drop(columns=diccionario_datos.columns[-4:]).notna().all(axis=1)
    diccionario_datos = diccionario_datos[mascara]
    
    diccionario_datos.columns = diccionario_datos.iloc[0]
    diccionario_datos = diccionario_datos[1:]
    
    if "Mnemónico" not in diccionario_datos.columns or "Indicador" not in diccionario_datos.columns:
        raise ValueError("El conjunto de datos proporcionado no contiene las columnas necesarias")

    return dict(zip(diccionario_datos.Mnemónico, diccionario_datos.Indicador))


def generar_conjunto_datos_AGEB(ruta_principal, destino):
    """
    Genera un conjunto de datos a nivel de AGEB (Área Geoestadística Básica) a partir de múltiples conjuntos de datos
    que solo abarcan una entidad a nivel MZA .
    
    Los conjuntos de datos usados para construir el conjunto a nivel AGEB deben estar en la ruta principal con el formato:
    ruta_principal/ageb_mza_urbana_{num}_cpv2020/conjunto_de_datos/conjunto_de_datos_ageb_urbana_{num}_cpv2020.csv

    Args:
        ruta_principal (str): Ruta al directorio principal que contiene los subdirectorios con los conjuntos de datos.
        destino (str): Ruta donde se guardará el archivo .csv combinado.

    Returns:
        None: Guarda el archivo combinado en la ruta especificada.

    Raises:
        ValueError: Si:
            - La ruta principal no existe o no fue especificada.
            - Los subdirectorios en la ruta principal no cumplen con el formato esperado.
    """
    
    if not os.path.exists(ruta_principal) or ruta_principal is None:
        raise ValueError(f"No se encontró la ruta especificada para 'ruta_principal': {ruta_principal}")
    
    dataframes = []
    
    subdirectorios = list(os.walk(ruta_principal, topdown=True))[0][1]
    if not all(re.match(r"^ageb_mza_urbana_\d+_cpv2020$", subdir) for subdir in subdirectorios):
        raise ValueError(f"La ruta principal debe contener únicamente subdirectorios con el formato: ageb_mza_urbana_{{num}}_cpv2020")

    subdirectorios = sorted(subdirectorios, key=lambda x: int(re.search(r"ageb_mza_urbana_(\d+)_cpv2020", x).group(1)))
    print(subdirectorios)
    
    for subdir in subdirectorios:
        nombre_subdir = subdir.split("_")
        
        nombre_archivo = f"conjunto_de_datos_ageb_urbana_{nombre_subdir[3]}_cpv2020.csv"
        ruta = os.path.join(ruta_principal, subdir, "conjunto_de_datos",nombre_archivo)
        
        dtype_dict = {
            "ENTIDAD": str,
            "MUN": str,
            "LOC": str,
            "AGEB": str,
            "MZA": str
        }
        
        if not os.path.exists(ruta):
            print(f"Advertencia: no se encontró una ruta hacia el archivo {ruta} correspondiente, excluyendo")
            continue
        
        df = pd.read_csv(ruta, dtype=dtype_dict, low_memory=False, encoding_errors="ignore")
        df = df[(df["AGEB"] != "0000") & (df["MZA"] == "000")]
        df = df.reset_index(drop=True)
        
        dataframes.append(df)
        print(f"Archivo {ruta} procesado correctamente")

    resultado = pd.concat(dataframes, ignore_index=True)
    resultado.to_csv(destino, index=False)


class ProcesadorCenso2020:
    """
    Clase para procesar datos del Censo de INEGI 2020.

    Permite trabajar con diferentes escalas geográficas: "state", "mun", "loc" o "ageb";
    perimite calcular porcentajes, categorizar variables y procesar dicha categorización.

    Attributes:
        df (pd.DataFrame): DataFrame que contiene los datos del Censo INEGI 2020.
        diccionario (dict): Diccionario que mapea nombres de variables a descripciones.
        escala (str): Escala geográfica seleccionada.
        variables_excluidas (list): Lista de variables excluidas del procesamiento.
    """
    
    def __init__(self, df:pd.DataFrame, diccionario_variables: dict, escala: str="state"):
        """
        Inicializa una instancia de la clase, capaz de procesar el DataFrame a una escala especificada.

        Args:
            df (pd.DataFrame): DataFrame que contiene los datos del Censo INEGI 2020.
            diccionario_variables (dict): Diccionario que mapea nombres de variables a descripciones.
            escala (str): Escala geográfica seleccionada para el procesamiento. Valores posibles:
                - "state": Procesa datos a nivel estatal.
                - "mun": Procesa datos a nivel municipal.
                - "loc": Procesa datos a nivel de localidad.
                - "ageb": Procesa datos a nivel de AGEB (Área Geoestadística Básica).

        Raises:
            ValueError: Si los parámetros no cumplen con las validaciones:
                - El valor especificado para df no es un DataFrame de Pandas.
                - El valor especificado para diccionario_variables no es un diccionario.
                - Las llaves del diccionario no coinciden exactamente con las variables del DataFrame (el orden es irrelevante).
                - El valor especificado para escala es diferente a los permitidos ("state", "mun", "loc" o "ageb").
                - El DataFrame especificado para df no contiene una variable llamada "TAMLOC" 
                    cuando la escala "loc" fue seleccionada.
                - El DataFrame especificado para df no contiene una variable llamada "AGEB"
                    cuando la escala "ageb" fue seleccionada.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("El parámetro 'df' debe ser un DataFrame")
        
        if not isinstance(diccionario_variables, dict):
            raise ValueError("El parámetro 'diccionario_variables' debe ser un diccionario")
        
        self.df = None
        self.diccionario_variables = diccionario_variables
        self.escala = escala
        self.variables_excluidas = []
        
        if escala == "state":
            escalado = df[(df["ENTIDAD"] != "00") & (df["MUN"] == "000") & (df["LOC"] == "0000")]
            escalado = escalado.reset_index(drop=True)
            self.df = escalado
            self.variables_excluidas = ["ENTIDAD", "NOM_ENT", "MUN", "NOM_MUN", "LOC", "NOM_LOC", "LONGITUD", "LATITUD", "ALTITUD", "TAMLOC"]
            self.df = self.__convertir_tipos(self.df)
        elif escala == "mun":
            escalado = df[(df["MUN"] != "000") & (df["LOC"] == "0000")]
            escalado = escalado.reset_index(drop=True)
            self.df = escalado
            self.variables_excluidas = ["ENTIDAD", "NOM_ENT", "MUN", "NOM_MUN", "LOC", "NOM_LOC", "LONGITUD", "LATITUD", "ALTITUD", "TAMLOC"]
            self.df = self.__convertir_tipos(self.df)
        elif escala == "loc":
            if "TAMLOC" not in df.columns:
                raise ValueError('Debe proporcionar el dataset correcto para seleccionar la escala "loc"')
            escalado = df[(df["ENTIDAD"] != "00") & (df["MUN"] != "000") & (df["LOC"] != "0000")]
            escalado = escalado.reset_index(drop=True)
            self.df = escalado
            self.variables_excluidas = ["ENTIDAD", "NOM_ENT", "MUN", "NOM_MUN", "LOC", "NOM_LOC", "LONGITUD", "LATITUD", "ALTITUD"]
            self.df = self.__convertir_tipos(self.df)
        elif escala == "ageb":
            if "AGEB" not in df.columns:
                raise ValueError('Debe proporcionar el dataset correcto para seleccionar la escala "ageb"')
            self.variables_excluidas = ["ENTIDAD", "NOM_ENT", "MUN", "NOM_MUN", "LOC", "NOM_LOC", "AGEB", "MZA"]
            self.df = self.__convertir_tipos(df)   
        else:
            raise ValueError('La escala especificada no es válida (debe ser "state", "mun", "loc" o "ageb")')
        
        variables_consideradas = set(self.df.columns) - set(self.variables_excluidas)
        if not (variables_consideradas).issubset(set(diccionario_variables.keys())):
            variables_faltantes = variables_consideradas - set(diccionario_variables.keys())
            raise ValueError(f"Las siguientes variables numéricas del DataFrame no están presentes en las llaves del diccionario: {', '.join(variables_faltantes)}")
        
        
    def __convertir_tipos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convierte los valores de las columnas del DataFrame a numéricos.
        
        Reemplaza valores no válidos (`"*"` o `"N/D"`) por NaN y transforma las columnas 
        a tipos numéricos (`Int64` o `Float64`).

        Args:
            df (pd.DataFrame): DataFrame a procesar.

        Returns:
            pd.DataFrame: DataFrame con los tipos de datos convertidos.
        """
        for var in df.columns:
            if var not in self.variables_excluidas:
                df[var] = np.where((df[var] == "*") | (df[var] == "N/D"), np.nan, df[var])
                try:
                    df[var] = df[var].astype("Int64")
                except ValueError:
                    df[var] = df[var].astype("Float64")
        return df
        
        
    def calcular_porcentajes_predeterminados(self) -> pd.DataFrame:
        """
        Calcula porcentajes para todas las variables con base predeterminada según su categoría: 
        población, hogares censales y vivienda.

        Returns:
            pd.DataFrame: DataFrame con los porcentajes calculados para todas las variables.
        """
        porcentajes = {}

        # variables relacionadas con vivienda
        for i in range(10, 226):
            columna = self.df.columns[i]
            if columna not in self.columnas_excluidas and pd.api.types.is_integer_dtype(self.df[columna]):
                porcentajes[f"{columna}"] = self.df[columna] / self.df["POBTOT"]
        for i in range(229, 232):
            columna = self.df.columns[i]
            if columna not in self.columnas_excluidas and pd.api.types.is_integer_dtype(self.df[columna]):
                porcentajes[f"{columna}"] = self.df[columna] / self.df["POBTOT"]
        porcentajes["OCUPVIVPAR"] = self.df["OCUPVIVPAR"] / self.df["POBTOT"]

        # variables relacionadas con hogares censales
        for i in range(227, 229):
            columna = self.df.columns[i]
            if columna not in self.columnas_excluidas and pd.api.types.is_integer_dtype(self.df[columna]):
                porcentajes[f"{columna}"] = self.df[columna] / self.df["TOTHOG"]
                
        # variables relacionadas con vivienda
        for i in range(233, 240):
            columna = self.df.columns[i]
            if columna not in self.columnas_excluidas and pd.api.types.is_integer_dtype(self.df[columna]):
                porcentajes[f"{columna}"] = self.df[columna] / self.df["VIVTOT"]
        for i in range(243, 285):
            columna = self.df.columns[i]
            if columna not in self.columnas_excluidas and pd.api.types.is_integer_dtype(self.df[columna]):
                porcentajes[f"{columna}"] = self.df[columna] / self.df["VIVTOT"]
                
        return pd.DataFrame(porcentajes) 
    
    
    def calcular_porcentaje(self, var: str, base_porcentaje: str) -> pd.Series:
        """
        Calcula el porcentaje de una variable con respecto a otra.

        Args:
            var (str): Nombre de la variable a calcular.
            base (str): Nombre de la variable base para calcular el porcentaje.

        Returns:
            pd.Series: Serie con los porcentajes calculados.

        Raises:
            ValueError: Si los parámetros no cumplen con las validaciones:
                - El valor de var no es una cadena.
                - La variable especificada no existe en el DataFrame.
                - La variable especificada no es de tipo numérico (se encuentra en la lista general de variables excluidas).
                - El valor de base_porcentaje no es una cadena.
                - La variable especificada como base para calcular el porcentaje no existe en el DataFrame.
                - La variable especificada como base para calcular el porcentaje no tiene valores numéricos en el DataFrame
                    (se encuentra en la lista general de variables excluidas).
        """
        
        if not isinstance(var, str):
            raise ValueError("El parámetro 'var' debe ser una cadena")
        if var not in self.df.columns:
            raise ValueError(f"La variable '{var}' no existe en el DataFrame")
        if var in self.variables_excluidas:
            raise ValueError(f"La variable '{var}' no tiene valores numéricos en el DataFrame")

        if not isinstance(base_porcentaje, str):
            raise ValueError("El parámetro 'base_porcentaje' debe ser una cadena")
        if base_porcentaje not in self.df.columns:
            raise ValueError(f"La variable '{base_porcentaje}' no existe en el DataFrame")
        if base_porcentaje in self.variables_excluidas:
            raise ValueError(f"'La variable '{base_porcentaje}' no tiene valores numéricos en el DataFrame")
        
        return self.df[var] / self.df[base_porcentaje]

        
        
    def categorizar_variable(self, var: str, base_porcentaje:str=None, q:int=10) -> pd.Series:
        """
        Categorización de una variable en q intervalos mediante la función pd.qcut().

        Args:
            var (str): Nombre de la variable a categorizar.
            base_porcentaje (str, optional): Variable base para calcular un porcentaje antes de categorizar.
            q (int): Número de categorías (deciles por defecto).

        Returns:
            pd.Series: Serie con la variable categorizada.
            
        Raises:
            ValueError: Si los parámetros no cumplen con las validaciones:
                - q no es un entero mayor a 1.
                - El valor de var no es una cadena.
                - La variable especificada no existe en el DataFrame.
                - La variable especificada no es de tipo numérico (se encuentra en la lista general de variables excluidas).
                - El valor de base_porcentaje no es una cadena o None.
                - La variable especificada como base para calcular el porcentaje no existe en el DataFrame.
                - La variable especificada como base para calcular el porcentaje no tiene valores numéricos en el DataFrame
                    (se encuentra en la lista general de variables excluidas).
        """
        
        if not isinstance(q, int) or q < 1:
            raise ValueError("El valor para el parámetro 'q' debe ser un entero mayor a 1")
        
        if not isinstance(var, str):
            raise ValueError("El parámetro 'var' debe ser una cadena")
        if var not in self.df.columns:
            raise ValueError(f"La variable '{var}' no existe en el DataFrame")
        if var in self.variables_excluidas:
            raise ValueError(f"La variable '{var}' no tiene valores numéricos en el DataFrame")
        
        if base_porcentaje is not None:
            if not isinstance(base_porcentaje, str):
                raise ValueError("El parámetro 'base_porcentaje' debe ser una cadena o None")
            if base_porcentaje not in self.df.columns:
                raise ValueError(f"La variable '{base_porcentaje}' no existe en el DataFrame")
            if base_porcentaje in self.variables_excluidas:
                raise ValueError(f"La variable '{base_porcentaje}' no tiene valores numéricos en el DataFrame")
        

        return pd.qcut(self.calcular_porcentaje(var, base_porcentaje), q=q, duplicates="drop") if (base_porcentaje is not None) else pd.qcut(self.df[var], q=q, duplicates="drop")


    def procesar_variable(self, var:str, base_porcentaje:str=None, q:int=10) -> pd.DataFrame:
        """
        Categoriza una variable y la procesa, generando un DataFrame con información sobre los intervalos, 
        código, descripción, e instancias asociadas a cada categoría.

        Args:
            var (str): Nombre de la variable a procesar.
            base_porcentaje (str, optional): Variable base para calcular el porcentaje antes de categorizar.
            q (int): Número de categorías en que se dividirán las variables (deciles por defecto).

        Returns:
            pd.DataFrame: DataFrame con los resultados del procesamiento.

        Raises:
            ValueError: Si los parámetros no cumplen con las validaciones:
                - q no es un entero mayor a 1.
                - El valor de var no es una cadena.
                - La variable especificada no existe en el DataFrame.
                - La variable especificada no es de tipo numérico (se encuentra en la lista general de variables excluidas).
                - El valor de base_porcentaje no es una cadena o None.
                - La variable especificada como base para calcular el porcentaje no existe en el DataFrame.
                - La variable especificada como base para calcular el porcentaje no tiene valores numéricos en el DataFrame
                    (se encuentra en la lista general de variables excluidas).
        """
        if not isinstance(q, int) or q < 1:
            raise ValueError("El valor para el parámetro 'q' debe ser un entero mayor a 1")
        
        if not isinstance(var, str):
            raise ValueError("El parámetro 'var' debe ser una cadena")
        if var not in self.df.columns:
            raise ValueError(f"La variable '{var}' no existe en el DataFrame")
        if var in self.variables_excluidas:
            raise ValueError(f"La variable '{var}' no tiene valores numéricos en el DataFrame")
        
        if base_porcentaje is not None:
            if not isinstance(base_porcentaje, str):
                raise ValueError("El parámetro 'base_porcentaje' debe ser una cadena o None")
            if base_porcentaje not in self.df.columns:
                raise ValueError(f"La variable '{base_porcentaje}' no existe en el DataFrame")
            if base_porcentaje in self.variables_excluidas:
                raise ValueError(f"La variable '{base_porcentaje}' no tiene valores numéricos en el DataFrame")


        resultado = {
            "name": [],
            "bin": [],
            "code": [],
            "interval": [],
            "lim_inf": [],
            "lim_sup": [],
            "mesh": [],
            "cells": []
        }
        
        variable_categorizada = pd.qcut(self.calcular_porcentaje(var, base_porcentaje), q=q, duplicates="drop") if (base_porcentaje is not None) else pd.qcut(self.df[var], q=q, duplicates="drop")
        variable_categorizada = variable_categorizada.cat.add_categories(["NaN"])
        variable_categorizada = variable_categorizada.fillna("NaN")
        nombre = self.diccionario_variables[var]
        codigo = var
        mesh = self.escala        
        cells = {intervalo : [] for intervalo in variable_categorizada.cat.categories}
        
        if self.escala == "state":
            for intervalo, entidad in zip(variable_categorizada, self.df["ENTIDAD"]):
                cells[intervalo].append(f"{entidad}") 
        elif self.escala == "mun":
            for intervalo, (entidad, mun) in zip(variable_categorizada, zip(self.df["ENTIDAD"], self.df["MUN"])):
                cells[intervalo].append(f"{entidad}{mun}") 
        elif self.escala == "loc":
            for intervalo, (entidad, mun, loc) in zip(variable_categorizada, zip(self.df["ENTIDAD"], self.df["MUN"], self.df["LOC"])):
                cells[intervalo].append(f"{entidad}{mun}{loc}")
        elif self.escala == "ageb":
            for intervalo, (entidad, mun, loc, ageb) in zip(variable_categorizada, zip(self.df["ENTIDAD"], self.df["MUN"], self.df["LOC"], self.df["AGEB"])):
                cells[intervalo].append(f"{entidad}{mun}{loc}{ageb}")

        if len(cells["NaN"]) == 0:
            variable_categorizada = variable_categorizada.cat.remove_categories(["NaN"])
            del cells["NaN"]

        for i, intervalo in enumerate(sorted(variable_categorizada.value_counts().index, key=lambda x: (isinstance(x, str), x))):
            if isinstance(intervalo, pd.Interval):
                resultado["name"].append(nombre)
                resultado["bin"].append(i+1)
                resultado["code"].append(codigo)
                resultado["interval"].append(
                    f"{(intervalo.left*100).round(1)}%:{(intervalo.right*100).round(1)}%" if (base_porcentaje is not None) else f"{intervalo.left}:{intervalo.right}"
                )
                resultado["lim_inf"].append(intervalo.left)
                resultado["lim_sup"].append(intervalo.right)
                resultado["mesh"].append(mesh)
                resultado["cells"].append(str(cells[intervalo]))
            else:
                resultado["name"].append(nombre)
                resultado["bin"].append(i+1)
                resultado["code"].append(codigo)
                resultado["interval"].append("NaN")
                resultado["lim_inf"].append(None)
                resultado["lim_sup"].append(None)
                resultado["mesh"].append(mesh)
                resultado["cells"].append(str(cells[intervalo]))
                
        return pd.DataFrame(resultado)
    
    
    def procesar_multiples_variables(self, dicc:dict, q:int=10) -> dict:
        """
        Categoriza múltiples variables y las procesa. 

        Se requiere un diccionario {base_porcentaje : lista_variables} que especifique qué variables se
        van a procesar, y si sus valores serán porcentajes, qué variable usarán como base.
        
        Args:
            dicc (dict): Diccionario donde las llaves son variables que se usan como base para calcular 
                porcentajes (puede ser None cuando no se requiere calcular) y los valores son listas de 
                variables a procesar.
                Ejemplo:
                {
                    "VAR_BASE_1" : ["VAR1", "VAR2", "VAR3"],
                    "VAR_BASE_2" : ["VAR4", "VAR5"],
                    None : ["VAR6", "VAR7", "VAR8"]
                }
                En este caso:
                - "VAR_BASE_1" es la base para calcular porcentajes de las variables "VAR1", "VAR2" y "VAR3".
                - "VAR_BASE_2" es la base para calcular porcentajes de las variables "VAR4" y "VAR5".
                - None indica que para las variables "VAR6", "VAR7" y "VAR8" no se calcularán porcentajes.
            q (int): Número de categorías en que se dividirán las variables (deciles por defecto).
            
        
        Returns:
            dict: Diccionario donde las llaves son las bases de porcentaje y los valores son DataFrames con los 
            resultados del procesamiento de las variables en la lista asociada a cada base.
        
        Raises:
            ValueError: Si los parámetros no cumplen con las validaciones:
                - q no es un entero mayor a 1.
                - El valor especificado para dicc no es un diccionario.
                - Una o más de las valores especificadas como llave del diccionario no son cadenas o None.
                - Una o más de las variables especificadas como llave del diccionario no existen en el DataFrame.
                - Una o más de las variables especificadas como llave del diccionario no son de tipo numérico
                    (se encuentran en la lista general de variables excluidas).
                - Uno o más de los valores del diccionario no son listas de cadenas.
                - Una o más de las variables especificadas en las listas no existen en el DataFrame.
                - Una o más de las variables especificadas en las listas no son de tipo numérico
                    (se encuentran en la lista general de variables excluidas).
        """
        resultado = {}
        
        if not isinstance(q, int) or q < 1:
            raise ValueError("El valor para el parámetro 'q' debe ser un entero mayor a 1")
        
        if not isinstance(dicc, dict):
            raise ValueError("El valor para el parámetro 'dicc' debe ser un diccionario de forma {base_porcentaje : lista_variables}")
        if dicc == {}:
            raise ValueError("El diccionario no contiene items, debe ser un diccionario de forma {base_porcentaje : lista variables}")
        
        for base_porcentaje, lista in dicc.items():

            if base_porcentaje is not None:
                if not isinstance(base_porcentaje, str):
                    raise ValueError("Las llaves del diccionario deben ser cadenas o None")
                if base_porcentaje not in self.df.columns:
                    raise ValueError(f"La variable '{base_porcentaje}' no existe en el DataFrame")
                if base_porcentaje in self.variables_excluidas:
                    raise ValueError(f"La variable '{base_porcentaje}' no tiene valores numéricos en el DataFrame")
            
            if not isinstance(lista, list):
                raise ValueError(f"El valor asociado a la llave '{base_porcentaje}' debe ser una lista")

            variables_procesadas = []
            
            for var in lista:
            
                if not isinstance(var, str):
                    raise ValueError("La lista de variables debe incluir únicamente cadenas")
                if var not in self.df.columns:
                    raise ValueError(f"La variable '{var}' no existe en el DataFrame")
                if var in self.variables_excluidas:
                    raise ValueError(f"La variable '{var}' no tiene valores numéricos en el DataFrame")
                
                variables_procesadas.append(self.procesar_variable(var=var, base_porcentaje=base_porcentaje, q=q))
                
            resultado[base_porcentaje] = pd.concat(variables_procesadas, ignore_index=True)
            
        return resultado
    
    
    def obtener_variables_regex(self, regex:str) -> list:
        """
        Filtra todas las variables en el DataFrame que hagan match con la expresión regular dada.

        Args:
            regex (str): Expresión regular que se usará para filtrar variables.

        Returns:
            list: Lista con las variables filtradas.
            
        Raises:
                ValueError: Si el valor especificado para regex no es una cadena.
        """
        if not isinstance(regex, str):
            raise ValueError("El parámetro 'regex' debe ser una cadena")
        
        return list(self.df.filter(regex=regex).columns)
            
    
    def procesar_multiples_variables_regex(self, dicc:dict, excluidas:list=None, q:int=10) -> dict:
        """
        Categoriza múltiples variables especificadas por una expresión regular y las procesa. 
        
        Se requiere un diccionario {base_porcentaje : expresion_regular} que especifique qué variables se
        van a procesar, y si sus valores serán porcentajes, qué variable usarán como base.
        
        Args:
            dicc (dict): Diccionario donde las llaves son variables que se usan como base para calcular 
                porcentajes (puede ser None cuando no se requiere calcular) y los valores son expresiones regulares
                con las cuales se obtendrá una lista de variables que coinciden.
                Ejemplo:
                {
                    "VAR_BASE_1" : regex1,
                    "VAR_BASE_2" : regex2,
                    None : "regex3"
                }
                En este caso:
                - "VAR_BASE_1" es la base para calcular porcentajes de las variables que coincidan con la expresión regular regex1.
                - "VAR_BASE_2" es la base para calcular porcentajes de las variables que coincidan con la expresión regular regex2.
                - None indica que para las variables que coincidan con la expresión regular regex3 no se calcularán porcentajes.
            excluidas (list, optional): Lista de variables que se excluirán del procesamiento, aún cuando coincidan con alguna expresión regular.
            q (int): Número de categorías en que se dividirán las variables (deciles por defecto).
            
        
        Returns:
            dict: Diccionario donde las llaves son las bases de porcentaje y los valores son DataFrames con los 
            resultados del procesamiento de las variables que coinciden con la expresión regular asociada a cada base.
        
        Raises:
            ValueError: Si los parámetros no cumplen con las validaciones:
                - q no es un entero mayor a 1.
                - El valor especificado para dicc no es un diccionario.
                - El valor especificado para excluidas no es una lista de cadenas o None.
                - Una o más de las valores especificadas como llave del diccionario no son cadenas o None.
                - Una o más de las variables especificadas como llave del diccionario no existen en el DataFrame.
                - Una o más de las variables especificadas como llave del diccionario no son de tipo numérico.
                    (se encuentran en la lista general de variables excluidas).
                - Uno o más de los valores del diccionario no son cadenas.
                - Una o más de las cadenas especificadas como valores del diccionario no son expresiones regulares válidas.
        """
        resultado = {}
        
        if not isinstance(q, int) or q < 1:
            raise ValueError("El valor para q debe ser un entero mayor a 1")
        
        if not isinstance(dicc, dict):
            raise ValueError("El valor para el parámetro 'dicc' debe ser un diccionario de forma {base_porcentaje : regex}")
        if dicc == {}:
            raise ValueError("El diccionario no contiene items, debe ser un diccionario de forma {base_porcentaje : regex}")
        
        if excluidas is not None:
            if not isinstance(excluidas, list):
                raise ValueError("El parámetro 'excluidas' debe ser una lista de cadenas")
            if not all(isinstance(item, str) for item in excluidas):
                raise ValueError("Todos los elementos del parámetro 'excluidas' deben ser cadenas")
        
        for base_porcentaje, regex in dicc.items():
            
            if base_porcentaje is not None:
                if not isinstance(base_porcentaje, str):
                    raise ValueError("Las llaves del diccionario deben ser cadenas o None")
            if base_porcentaje is not None and base_porcentaje not in self.df.columns:
                raise ValueError(f"La variable '{base_porcentaje}' no existe en el DataFrame")
            if base_porcentaje in self.variables_excluidas:
                raise ValueError(f"La variable '{base_porcentaje}' no tiene valores numéricos en el DataFrame")
            
            if not isinstance(regex, str):
                raise ValueError(f"El valor asociado a la llave '{base_porcentaje}' no es una cadena")
            try:
                re.compile(regex)
            except re.error:
                raise ValueError(f"La expresión regular '{regex}' no es válida")
                
            variables_regex = list(filter(lambda x: x not in excluidas, self.obtener_variables_regex(regex))) if excluidas is not None else self.obtener_variables_regex(regex) 
            
            if len(variables_regex) == 0:
                print(f"Advertencia: la expresión regular '{regex}' no coincide con ninguna variable")
                
            variables_procesadas = []
            
            for var in variables_regex:
                if var in self.variables_excluidas:
                    print(f"Advertencia: la expresión regular '{regex}' coincide con una variable que no tiene valores numéricos en el DataFrame ('{var}'), excluyendo")
                    continue
                variables_procesadas.append(self.procesar_variable(var=var, base_porcentaje=base_porcentaje, q=q))
                
            resultado[base_porcentaje] = pd.concat(variables_procesadas, ignore_index=True) if len(variables_procesadas) > 0 else None
            
        return resultado
    