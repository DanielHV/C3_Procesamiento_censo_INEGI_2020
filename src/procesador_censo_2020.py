import os
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import re

def importar_conjunto_datos(ruta_principal:str, escala:str, destino:str=None):
    """
    Importa un conjunto de datos del Censo INEGI 2020 y lo convierte en un DataFrame de pandas.

    Configura los tipos de datos de columnas específicas según la versión del conjunto de datos 
    ("iter" o "ageb") para mantener consistencia.

    Args:
        ruta (str): Ruta al archivo .csv que contiene el conjunto de datos.
        escala (str): Versión del conjunto de datos. Puede ser:
            - "state": Filtrar los datos a nivel estatal.
            - "mun": Filtrar los datos a nivel municipal.
            - "ageb": Filtrar los datos a nivel AGEB (Área Geoestadística Básica).
        destino (str): Ruta al archivo .csv que contiene el conjunto de datos.

    Returns:
        pd.DataFrame: DataFrame con los datos importados y los tipos de datos configurados.

    Raises:
        ValueError: Si los parámetros no cumplen con las validaciones:
            - El valor especificado para ruta_principal no es una cadena
            - El valor especificado para destino no es una cadena o None
            - La ruta especificada no existe o no fue especificada.
            - El destino especificado no existe.
            - El valor del parámetro 'escala' no es "state", "mun" o "ageb".
            - El archivo de datos encontrado en la ruta especificada no contiene las variables esperadas para la escala especificada.
    """
    
    if not isinstance(ruta_principal, str):
        raise ValueError(f"El parámetro 'ruta_principal' debe ser una cadena")
    if not os.path.exists(ruta_principal) or ruta_principal is None:
        raise ValueError(f"No se encontró la ruta especificada para 'ruta_principal': {ruta_principal}")
    
    if destino is not None:
        if not isinstance(destino, str):
            raise ValueError(f"El parámetro 'destino' debe ser una cadena")
        if not os.path.exists(destino):
            raise ValueError(f"No se encontró la ruta especificada para 'destino': {destino}")
    
    if escala.lower() == "state":
        
        dtype_dict = {
            "ENTIDAD": str,
            "MUN": str,
            "LOC": str
        }
        df = pd.read_csv(ruta_principal, dtype=dtype_dict, low_memory=False)
        
        variables_representativas = ["ENTIDAD", "NOM_ENT", "MUN", "NOM_MUN", "LOC", "NOM_LOC", "LONGITUD", "LATITUD", "ALTITUD", "TAMLOC"]
        variables_diferentes = set(variables_representativas) - set(df.columns)
        if len(variables_diferentes) != 0:
            raise ValueError(f"El archivo de datos encontrado en la ruta {ruta_principal} no contiene las siguientes variables esperadas para la escala 'state': \n {variables_diferentes}")
            
        resultado = df[(df["ENTIDAD"] != "00") & (df["MUN"] == "000") & (df["LOC"] == "0000")]
        resultado = resultado.reset_index(drop=True)
        resultado = convertir_tipos(resultado, variables_a_excluir=variables_representativas)
        print(f"Archivo {ruta_principal} procesado correctamente para escala 'state'")
        
        if destino is not None:
            resultado.to_csv(destino, index=False)
        
        return resultado  
    
    elif escala.lower() == "mun":
        
        dtype_dict = {
            "ENTIDAD": str,
            "MUN": str,
            "LOC": str
        }
        df = pd.read_csv(ruta_principal, dtype=dtype_dict, low_memory=False)
        
        variables_representativas = ["ENTIDAD", "NOM_ENT", "MUN", "NOM_MUN", "LOC", "NOM_LOC", "LONGITUD", "LATITUD", "ALTITUD", "TAMLOC"]
        variables_diferentes = set(variables_representativas) - set(df.columns)
        if len(variables_diferentes) != 0:
            raise ValueError(f"El archivo de datos encontrado en la ruta {ruta_principal} no contiene las siguientes variables esperadas para la escala 'mun': \n {variables_diferentes}")
        
        resultado = df[(df["MUN"] != "000") & (df["LOC"] == "0000")]
        resultado = resultado.reset_index(drop=True)
        resultado = convertir_tipos(resultado, variables_a_excluir=variables_representativas)
        print(f"Archivo {ruta_principal} procesado correctamente para escala 'mun'")
        
        if destino is not None:
            resultado.to_csv(destino, index=False)
        
        return resultado

    elif escala.lower() == "ageb":
        
        dtype_dict = {
            "ENTIDAD": str,
            "MUN": str,
            "LOC": str,
            "AGEB": str,
            "MZA": str
        }
        
        variables_representativas = ["ENTIDAD", "NOM_ENT", "MUN", "NOM_MUN", "LOC", "NOM_LOC", "AGEB", "MZA"]
        dataframes = []
        
        subdirectorios = list(os.walk(ruta_principal, topdown=True))[0][1]
        if not all(re.match(r"^ageb_mza_urbana_\d+_cpv2020$", subdir) for subdir in subdirectorios):
            raise ValueError(f"La ruta principal debe contener únicamente subdirectorios con el formato: ageb_mza_urbana_{{num}}_cpv2020")

        subdirectorios = sorted(subdirectorios, key=lambda x: int(re.search(r"ageb_mza_urbana_(\d+)_cpv2020", x).group(1)))
        
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
            
            variables_diferentes = set(variables_representativas) - set(df.columns)
            if len(variables_diferentes) != 0:
                raise ValueError(f"El archivo de datos encontrado en la ruta {ruta} no contiene las siguientes variables esperadas para la escala 'ageb': \n {variables_diferentes}")
            
            df = df[(df["AGEB"] != "0000") & (df["MZA"] == "000")]
            df = df.reset_index(drop=True)
            
            dataframes.append(df)
            print(f"Archivo {ruta} procesado correctamente para escala 'ageb'")

        resultado = pd.concat(dataframes, ignore_index=True)
        resultado = resultado.reset_index(drop=True)
        resultado = convertir_tipos(resultado, variables_a_excluir=variables_representativas)
            
        if destino is not None:
            resultado.to_csv(destino, index=False)
        
        return resultado

    else:
        raise ValueError("El valor del parámetro 'escala' debe ser \"state\", \"mun\" o \"ageb\"")


def convertir_tipos(df:pd.DataFrame, variables_a_excluir:list=None) -> pd.DataFrame:
    """
    Convierte los valores de las columnas del DataFrame a numéricos. Reemplaza valores no válidos (`"*"` o `"N/D"`) por NaN 
    y transforma las columnas a tipos numéricos (`Int64` o `Float64`).

    Args:
        df (pd.DataFrame): DataFrame a procesar.

    Returns:
        pd.DataFrame: DataFrame con los tipos de datos convertidos.
        
    Raises:
        ValueError: Si el valor del parámetro 'variables_a_excluir' no es una lista de cadenas o None
    """
    
    if variables_a_excluir is not None:
        if not isinstance(variables_a_excluir, list) and all(isinstance(x, str) for x in variables_a_excluir):
            raise ValueError("El parámetro 'variables_a_excluir' debe ser una lista de cadenas o None")
    
    for var in df.columns:
        if var not in variables_a_excluir:
            #df[var] = np.where((df[var] == "*") | (df[var] == "N/D"), np.nan, df[var])
            df[var] = df[var].replace(["*", "N/D"], np.nan)#.infer_objects(copy=False)
            #df[var] = pd.to_numeric(df[var], errors="coerce")
            try:
                df[var] = df[var].astype("Int64")
            except (ValueError, TypeError):
                try:
                    df[var] = df[var].astype("Float64")
                except (ValueError, TypeError):
                    continue
                
    return df

    
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
    
    
def lista_a_pg_array(val):
    """
    Convierte una lista de Python en un string con formato de arreglo de PostgreSQL,
    eliminando cualquier comilla simple o doble de los elementos.

    Si el valor proporcionado es una lista, la función lo transforma en una cadena con llaves y elementos separados por comas,
    compatible con el tipo de dato array de PostgreSQL (por ejemplo: {a,b,c}). Si el valor no es una lista, lo retorna sin cambios.

    Args:
        val (any): Valor a convertir. Si es una lista, se convierte al formato de arreglo de PostgreSQL.

    Returns:
        str or any: Cadena en formato de arreglo de PostgreSQL si el valor es una lista, o el valor original en caso contrario.
    """
    if isinstance(val, list):
        clean = [str(x).replace("'", "").replace('"', "") for x in val]
        return '{' + ','.join(f'{c}' for c in clean) + '}'
    return val



class ProcesadorCenso2020:
    """
    Clase para procesar datos del Censo de INEGI 2020.

    Permite trabajar simultáneamente con diferentes escalas geográficas: estatal:"state", municipal:"mun", o ageb:"ageb";
    permite calcular porcentajes, categorizar variables y procesar dicha categorización.

    Attributes:
        df_state (pd.DataFrame): DataFrame que contiene los datos del Censo INEGI 2020 a escala estatal "state".
        df_mun (pd.DataFrame): DataFrame que contiene los datos del Censo INEGI 2020 a escala municipal "mun".
        df_AGEB (pd.DataFrame): DataFrame que contiene los datos del Censo INEGI 2020 necesarios para trabajar a escala ageb "ageb".
        diccionario_variables (dict): Diccionario que mapea nombres de variables a descripciones.
        variables_excluidas (list): Lista de variables excluidas del procesamiento.
    """
    def __init__(self, diccionario_variables:dict, df_state:pd.DataFrame=None, df_mun:pd.DataFrame=None, df_ageb:pd.DataFrame=None):
        """
        Inicializa una instancia de la clase, capaz de procesar el DataFrame a diferentes escalas.
        
        Se espera como DataFrame el conjunto de datos importado con la función 'importar_conjunto_datos' a partir del archivo 'conjunto_de_datos_iter_00CSV20.csv' para el parámetro 'df_state'.
        Se espera como DataFrame el conjunto de datos importado con la función 'importar_conjunto_datos' a partir del archivo 'conjunto_de_datos_iter_00CSV20.csv' para el parámetro 'df_mun'.
        Se espera como DataFrame el conjunto de datos importado con la función 'importar_conjunto_datos' a partir de los archivos 'conjunto_de_datos_ageb_urbana_{}_cpv2020.csv' para el parámetro 'df_ageb'.
        Se espera como DataFrame el diccionario de datos importado con la función 'importar_diccionario_datos' a partir del archivo 'diccionario_datos_iter_00CSV20.csv' idealmente, o 'diccionario_datos_ageb_urbana_{}_cpv2020.csv'
        el cual contiene menos variables para el parámetro 'diccionario_variables'.

        Args:
            df_state (pd.DataFrame): DataFrame que contiene los datos del Censo INEGI 2020 necesarios para procesar los datos a escala
                estatal "state".
            df_mun (pd.DataFrame): DataFrame que contiene los datos del Censo INEGI 2020 necesarios para procesar los datos a escala
                municipal "mun".
            df_ageb (pd.DataFrame): DataFrame que contiene los datos del Censo INEGI 2020 necesarios para procesar los datos a escala
                ageb "ageb".
            diccionario_variables (dict): Diccionario que mapea nombres de variables a descripciones.

        Raises:
            ValueError: Si los parámetros no cumplen con las validaciones:
                - No se proporcionó por lo menos de los DataFrames df_state, df_mun o df_ageb.
                - El valor especificado para df_state no es un DataFrame de Pandas.
                - El valor especificado para df_mun no es un DataFrame de Pandas.
                - El valor especificado para df_ageb no es un DataFrame de Pandas.
                - El valor especificado para diccionario_variables no es un diccionario.
                - Las llaves del diccionario no coinciden exactamente con las variables del DataFrame (el orden es irrelevante).
        """
        if df_state is None and df_mun is None and df_ageb:
            raise ValueError("Debe proporcionarse al menos un DataFrame")
        
        if not isinstance(df_state, pd.DataFrame) and df_state is not None:
            raise ValueError("El parámetro 'df_state' debe ser un DataFrame")

        if not isinstance(df_mun, pd.DataFrame) and df_mun is not None:
            raise ValueError("El parámetro 'df_mun' debe ser un DataFrame")
        
        if not isinstance(df_ageb, pd.DataFrame) and df_ageb is not None:
            raise ValueError("El parámetro 'df_ageb' debe ser un DataFrame")
        
        if not isinstance(diccionario_variables, dict):
            raise ValueError("El parámetro 'diccionario_variables' debe ser un diccionario")
        
        self.df_state = df_state
        self.df_mun = df_mun
        self.df_ageb = df_ageb
        self.diccionario_variables = diccionario_variables
        self.variables_excluidas = set()
        
        columnas_df_state = set()
        columnas_df_mun = set()
        columnas_df_ageb = set()
        
        if self.df_state is not None:
            
            self.variables_excluidas = self.variables_excluidas | {"ENTIDAD", "NOM_ENT", "MUN", "NOM_MUN", "LOC", "NOM_LOC", "LONGITUD", "LATITUD", "ALTITUD", "TAMLOC"}
            columnas_df_state = set(self.df_state.columns)
            
        if self.df_mun is not None:
            
            self.variables_excluidas = self.variables_excluidas | {"ENTIDAD", "NOM_ENT", "MUN", "NOM_MUN", "LOC", "NOM_LOC", "LONGITUD", "LATITUD", "ALTITUD", "TAMLOC"}
            columnas_df_mun = set(self.df_mun.columns)

        if self.df_ageb is not None:
            
            self.variables_excluidas = self.variables_excluidas | {"ENTIDAD", "NOM_ENT", "MUN", "NOM_MUN", "LOC", "NOM_LOC", "AGEB", "MZA"}
            columnas_df_ageb = set(self.df_ageb.columns)

        variables_consideradas = (columnas_df_state | columnas_df_mun | columnas_df_ageb) - self.variables_excluidas
        variables_en_diccionario = set(diccionario_variables.keys())
        if not (variables_consideradas).issubset(variables_en_diccionario):
            variables_faltantes = variables_consideradas - variables_en_diccionario
            raise ValueError(f"Las siguientes variables numéricas del DataFrame no están presentes en las llaves del diccionario: {', '.join(variables_faltantes)}")
        
        print(f"Variables excluidas: {self.variables_excluidas}")
    
    
    def calcular_porcentaje(self, escala:str, var:str, base_porcentaje: str) -> pd.Series:
        """
        Calcula el porcentaje de una variable con respecto a otra.

        Args:
            escala (str): Escala seleccionada.
            var (str): Nombre de la variable a calcular.
            base (str): Nombre de la variable base para calcular el porcentaje.

        Returns:
            pd.Series: Serie con los porcentajes calculados.

        Raises:
            ValueError: Si los parámetros no cumplen con las validaciones:
                - El valor de escala no es una cadena.
                - La escala seleccionada no es "state", "mun" o "ageb".
                - No fue proporcionado a la clase el DataFrame necesario para la escala seleccionada.
                - El valor de var no es una cadena.
                - La variable especificada no existe en el DataFrame correspondiente a la escala seleccionada.
                - La variable especificada no es de tipo numérico (se encuentra en la lista general de variables excluidas).
                - El valor de base_porcentaje no es una cadena.
                - La variable especificada como base para calcular el porcentaje no existe en el DataFrame correspondiente a la escala seleccionada.
                - La variable especificada como base para calcular el porcentaje no tiene valores numéricos en el DataFrame correspondiente a la escala seleccionada.
                    (se encuentra en la lista general de variables excluidas).
        """
        df = None
        if not isinstance(escala, str):
            raise ValueError("El parámetro 'escala' debe ser una cadena")
        if escala == "state":
            if self.df_state is None: raise ValueError("No es posible seleccionar la escala, pues no se ha proporcionado el DataFrame necesario")
            df = self.df_state
        elif escala == "mun":
            if self.df_mun is None: raise ValueError("No es posible seleccionar la escala, pues no se ha proporcionado el DataFrame necesario")
            df = self.df_mun
        elif escala == "ageb":
            if self.df_ageb is None: raise ValueError("No es posible seleccionar la escala, pues no se ha proporcionado el DataFrame necesario")
            df = self.df_ageb
        else:
            raise ValueError("El parámetro 'escala' debe ser: \"state\", \"mun\" o \"ageb\"")
        
        if not isinstance(var, str):
            raise ValueError("El parámetro 'var' debe ser una cadena")
        if var not in df.columns:
            raise ValueError(f"La variable '{var}' no existe en el DataFrame de la escala seleccionada")
        if var in self.variables_excluidas:
            raise ValueError(f"La variable '{var}' no tiene valores numéricos en el DataFrame de la escala seleccionada")

        if not isinstance(base_porcentaje, str):
            raise ValueError("El parámetro 'base_porcentaje' debe ser una cadena")
        if base_porcentaje not in df.columns:
            raise ValueError(f"La variable '{base_porcentaje}' no existe en el DataFrame de la escala seleccionada")
        if base_porcentaje in self.variables_excluidas:
            raise ValueError(f"'La variable '{base_porcentaje}' no tiene valores numéricos en el DataFrame de la escala seleccionada")

        return df[var] / df[base_porcentaje]
        
        
    def categorizar_variable(self, escala:str, var:str, base_porcentaje:str=None, q:int=10) -> pd.Series:
        """
        Categorización de una variable en q intervalos mediante la función pd.qcut().

        Args:
            escala (str): Escala seleccionada.
            var (str): Nombre de la variable a categorizar.
            base_porcentaje (str, optional): Variable base para calcular un porcentaje antes de categorizar.
            q (int): Número de categorías (deciles por defecto).

        Returns:
            pd.Series: Serie con la variable categorizada.
            
        Raises:
            ValueError: Si los parámetros no cumplen con las validaciones:
                - El valor de escala no es una cadena.
                - La escala seleccionada no es "state", "mun" o "ageb".
                - No fue proporcionado a la clase el DataFrame necesario para la escala seleccionada.
                - q no es un entero mayor a 1.
                - El valor de var no es una cadena.
                - La variable especificada no existe en el diccionario de variables.
                - La variable especificada no existe en el DataFrame.
                - La variable especificada no es de tipo numérico (se encuentra en la lista general de variables excluidas).
                - El valor de base_porcentaje no es una cadena o None.
                - La variable especificada como base para calcular el porcentaje no existe en el diccionario de variables.
                - La variable especificada como base para calcular el porcentaje no existe en el DataFrame.
                - La variable especificada como base para calcular el porcentaje no tiene valores numéricos en el DataFrame
                    (se encuentra en la lista general de variables excluidas).
        """
        df = None
        if not isinstance(escala, str):
            raise ValueError("El parámetro 'escala' debe ser una cadena")
        if escala == "state":
            if self.df_state is None: raise ValueError("No es posible seleccionar la escala, pues no se ha proporcionado el DataFrame necesario")
            df = self.df_state
        elif escala == "mun":
            if self.df_mun is None: raise ValueError("No es posible seleccionar la escala, pues no se ha proporcionado el DataFrame necesario")
            df = self.df_mun
        elif escala == "ageb":
            if self.df_ageb is None: raise ValueError("No es posible seleccionar la escala, pues no se ha proporcionado el DataFrame necesario")
            df = self.df_ageb
        else:
            raise ValueError("El parámetro 'escala' debe ser: \"state\", \"mun\" o \"ageb\"")
        
        if not isinstance(q, int) or q < 1:
            raise ValueError("El valor para el parámetro 'q' debe ser un entero mayor a 1")
        
        if not isinstance(var, str):
            raise ValueError("El parámetro 'var' debe ser una cadena")
        if var not in self.diccionario_variables:
            raise ValueError(f"La variable {var} no existe en diccionario de variables proporcionado")
        if var not in df.columns:
            raise ValueError(f"La variable '{var}' no existe en el DataFrame de la escala seleccionada")
        if var in self.variables_excluidas:
            raise ValueError(f"La variable '{var}' no tiene valores numéricos en el DataFrame de la escala seleccionada")
        
        if base_porcentaje is not None:
            if not isinstance(base_porcentaje, str):
                raise ValueError("El parámetro 'base_porcentaje' debe ser una cadena o None")
            if base_porcentaje not in self.diccionario_variables:
                raise ValueError(f"La variable {base_porcentaje} no existe en diccionario de variables proporcionado")
            if base_porcentaje not in df.columns:
                raise ValueError(f"La variable '{base_porcentaje}' no existe en el DataFrame de la escala seleccionada")
            if base_porcentaje in self.variables_excluidas:
                raise ValueError(f"La variable '{base_porcentaje}' no tiene valores numéricos en el DataFrame de la escala seleccionada")
        
        return pd.qcut(self.calcular_porcentaje(escala, var, base_porcentaje), q=q, duplicates="drop") if (base_porcentaje is not None) else pd.qcut(df[var], q=q, duplicates="drop")
    
    
    def procesar_variable(self, escalas:list, var:str, base_porcentaje:str=None, q:int=10) -> pd.DataFrame:
        """
        Categoriza una variable y la procesa, generando un DataFrame con información sobre los intervalos, 
        código, descripción, e instancias asociadas a cada categoría.

        Args:
            escalas (list): Lista de escalas seleccionadas.
            var (str): Nombre de la variable a procesar.
            base_porcentaje (str, optional): Variable base para calcular el porcentaje antes de categorizar.
            q (int): Número de categorías en que se dividirán las variables (deciles por defecto).

        Returns:
            pd.DataFrame: DataFrame con los resultados del procesamiento.

        Raises:
            ValueError: Si los parámetros no cumplen con las validaciones:
                - El valor de escalas no es una lista.
                - Las escala seleccionadas no son "state", "mun" o "ageb".
                - No fue proporcionado a la clase el DataFrame necesario para alguna de las escalas seleccionada.
                - q no es un entero mayor a 1.
                - El valor de var no es una cadena.
                - La variable especificada no existe en el diccionario de variables.
                - La variable especificada no existe en ninguno de los DataFrames de la escalas especificadas.
                - La variable especificada no es de tipo numérico (se encuentra en la lista general de variables excluidas).
                - El valor de base_porcentaje no es una cadena o None.
                - La variable especificada como base para calcular el porcentaje no existe en el diccionario de variables.
                - La variable especificada como base para calcular el porcentaje no existe en ninguno de los DataFrames de la escalas especificadas.
                - La variable especificada como base para calcular el porcentaje no tiene valores numéricos en el DataFrame.
                    de la escala especificada (se encuentra en la lista general de variables excluidas).
        """
        if not isinstance(escalas, list):
            raise ValueError("El parámetro 'escala' debe ser una lista")
        if not (1 <= len(escalas) <= 3):
            raise ValueError("Debe especificar entre 1 y 3 escalas")
        escalas_validas = ["state", "mun", "ageb"]
        for escala in escalas:
            if escala not in escalas_validas:
                raise ValueError(f"La escala '{escala}' no es válida. Las opciones válidas son: 'state', 'mun' y 'ageb'")
        
        if not isinstance(q, int) or q < 1:
            raise ValueError("El valor para el parámetro 'q' debe ser un entero mayor a 1")
        
        if not isinstance(var, str):
            raise ValueError("El parámetro 'var' debe ser una cadena")
        if var not in self.diccionario_variables:
            raise ValueError(f"La variable {var} (var) no existe en diccionario de variables proporcionado")
        if var in self.variables_excluidas:
            raise ValueError(f"La variable '{var}' (var) no tiene valores numéricos en los DataFrames")
        
        if base_porcentaje is not None:
            if not isinstance(base_porcentaje, str):
                raise ValueError("El parámetro 'base_porcentaje' debe ser una cadena o None")
            if base_porcentaje not in self.diccionario_variables:
                raise ValueError(f"La variable {base_porcentaje} (base_porcentaje) no existe en diccionario de variables proporcionado")
            if base_porcentaje in self.variables_excluidas:
                raise ValueError(f"La variable '{base_porcentaje}' (base_porcentaje) no tiene valores numéricos en el DataFrame de la escala especificada")

        resultado = {
            "name": [],
            "code": [],
            "bin": [],
            "interval_state": [],
            "interval_mun": [],
            "interval_ageb": [],
            "cells_state": [],
            "cells_mun": [],
            "cells_ageb": []
        }
        
        nombre = self.diccionario_variables[var]
        codigo = var
        variable_categorizada_state = None
        variable_categorizada_mun = None
        variable_categorizada_ageb = None
        
        validacion_state_var = False
        validacion_mun_var = False
        validacion_ageb_var = False
        validacion_state_base_porcentaje = False
        validacion_mun_base_porcentaje = False
        validacion_ageb_base_porcentaje = False
        
        if "state" in escalas:
            if var not in self.df_state.columns:
                print(f"La variable '{var}' (var) no existe en el DataFrame de la escala 'state'")
            else:
                validacion_state_var = True
            if base_porcentaje is not None:
                if base_porcentaje not in self.df_state.columns:
                    print(f"La variable '{base_porcentaje}' (base_porcentaje) no existe en el DataFrame de la escala 'state'")
                else:
                    validacion_state_base_porcentaje = True
            else:
                validacion_state_base_porcentaje = True
            
        if "mun" in escalas:
            if var not in self.df_mun.columns:
                print(f"La variable '{var}' (var) no existe en el DataFrame de la escala 'mun'")
            else:
                validacion_mun_var = True
            if base_porcentaje is not None:
                if base_porcentaje not in self.df_mun.columns:
                    print(f"La variable '{base_porcentaje}' (base_porcentaje) no existe en el DataFrame de la escala 'mun'")
                else:
                    validacion_mun_base_porcentaje = True
            else:
                validacion_mun_base_porcentaje = True
                
        if "ageb" in escalas:
            if var not in self.df_ageb.columns:
                print(f"La variable '{var}' (var) no existe en el DataFrame de la escala 'ageb'")
            else:
                validacion_ageb_var = True
            if base_porcentaje is not None:
                if base_porcentaje not in self.df_ageb.columns:
                    print(f"La variable '{base_porcentaje}' (base_porcentaje) no existe en el DataFrame de la escala 'ageb'")
                else:
                    validacion_ageb_base_porcentaje = True
            else:
                validacion_ageb_base_porcentaje = True
                
        if not validacion_state_var and not validacion_mun_var and not validacion_ageb_var:
            raise ValueError(f"La variable {var} (var) no existe en ninguno de los DataFrames de las escalas especificadas")
        if not validacion_state_base_porcentaje and not validacion_mun_base_porcentaje and not validacion_ageb_base_porcentaje:
            raise ValueError(f"La variable {base_porcentaje} (base_porcentaje) no existe en ninguno de los DataFrames de las escalas especificadas")
        
        if validacion_state_var and validacion_state_base_porcentaje:
            
            variable_categorizada_state = self.categorizar_variable("state", var, base_porcentaje, q)
            variable_categorizada_state = variable_categorizada_state.cat.add_categories(["NaN"])
            variable_categorizada_state = variable_categorizada_state.fillna("NaN")
            
            cells_state = {intervalo : [] for intervalo in variable_categorizada_state.cat.categories}
            for intervalo, entidad in zip(variable_categorizada_state, self.df_state["ENTIDAD"]):
                cells_state[intervalo].append(f"{entidad}")
                
            # eliminar categoria NaN cuando no tiene elementos
            if len(cells_state["NaN"]) == 0:
                variable_categorizada_state = variable_categorizada_state.cat.remove_categories(["NaN"])
                del cells_state["NaN"]
                
        if validacion_mun_var and validacion_mun_base_porcentaje:       
               
            variable_categorizada_mun = self.categorizar_variable("mun", var, base_porcentaje, q)
            variable_categorizada_mun = variable_categorizada_mun.cat.add_categories(["NaN"])
            variable_categorizada_mun = variable_categorizada_mun.fillna("NaN")
            
            cells_mun = {intervalo : [] for intervalo in variable_categorizada_mun.cat.categories}
            for intervalo, (entidad, mun) in zip(variable_categorizada_mun, zip(self.df_mun["ENTIDAD"], self.df_mun["MUN"])):
                cells_mun[intervalo].append(f"{entidad}{mun}") 
                
            # eliminar categoria NaN cuando no tiene elementos
            if len(cells_mun["NaN"]) == 0:
                variable_categorizada_mun = variable_categorizada_mun.cat.remove_categories(["NaN"])
                del cells_mun["NaN"]
                
        if validacion_ageb_var and validacion_ageb_base_porcentaje: 
            
            variable_categorizada_ageb = self.categorizar_variable("ageb", var, base_porcentaje, q)
            variable_categorizada_ageb = variable_categorizada_ageb.cat.add_categories(["NaN"])
            variable_categorizada_ageb = variable_categorizada_ageb.fillna("NaN")
            
            cells_ageb = {intervalo : [] for intervalo in variable_categorizada_ageb.cat.categories}
            for intervalo, (entidad, mun, loc, ageb) in zip(variable_categorizada_ageb, zip(self.df_ageb["ENTIDAD"], self.df_ageb["MUN"], self.df_ageb["LOC"], self.df_ageb["AGEB"])):
                cells_ageb[intervalo].append(f"{entidad}{mun}{loc}{ageb}")
                
            # eliminar categoria NaN cuando no tiene elementos
            if len(cells_ageb["NaN"]) == 0:
                variable_categorizada_ageb = variable_categorizada_ageb.cat.remove_categories(["NaN"])
                del cells_ageb["NaN"]
        
        intervalos_state = sorted(variable_categorizada_state.value_counts().index, key=lambda x: (isinstance(x, str), x)) if variable_categorizada_state is not None else []
        intervalos_mun = sorted(variable_categorizada_mun.value_counts().index, key=lambda x: (isinstance(x, str), x)) if variable_categorizada_mun is not None else []
        intervalos_ageb = sorted(variable_categorizada_ageb.value_counts().index, key=lambda x: (isinstance(x, str), x)) if variable_categorizada_ageb is not None else []
        
        for i in range(q+1):
            resultado["name"].append(nombre)
            resultado["code"].append(codigo)
            resultado["bin"].append(i+1)
            try:
                intervalo_state = intervalos_state[i]
                if isinstance(intervalo_state, pd.Interval):
                    resultado["interval_state"].append(
                            f"{(intervalo_state.left*100).round(1)}%:{(intervalo_state.right*100).round(1)}%" if (base_porcentaje is not None) else f"{intervalo_state.left}:{intervalo_state.right}"
                        )
                else:
                    resultado["interval_state"].append("Sin clasificar")
                resultado["cells_state"].append(lista_a_pg_array(cells_state[intervalo_state]))
            except IndexError:
                resultado["interval_state"].append(pd.NA)
                resultado["cells_state"].append(pd.NA)
                
            try:
                intervalo_mun = intervalos_mun[i]
                if isinstance(intervalo_mun, pd.Interval):
                    resultado["interval_mun"].append(
                            f"{(intervalo_mun.left*100).round(1)}%:{(intervalo_mun.right*100).round(1)}%" if (base_porcentaje is not None) else f"{intervalo_mun.left}:{intervalo_mun.right}"
                        )
                else:
                    resultado["interval_mun"].append("Sin clasificar")
                resultado["cells_mun"].append(lista_a_pg_array(cells_mun[intervalo_mun]))
            except IndexError:
                resultado["interval_mun"].append(pd.NA)
                resultado["cells_mun"].append(pd.NA)
                
            try:
                intervalo_ageb = intervalos_ageb[i]
                if isinstance(intervalo_ageb, pd.Interval):
                    resultado["interval_ageb"].append(
                            f"{(intervalo_ageb.left*100).round(1)}%:{(intervalo_ageb.right*100).round(1)}%" if (base_porcentaje is not None) else f"{intervalo_ageb.left}:{intervalo_ageb.right}"
                        )
                else:
                    resultado["interval_ageb"].append("Sin clasificar")
                resultado["cells_ageb"].append(lista_a_pg_array(cells_ageb[intervalo_ageb]))
            except IndexError:
                resultado["interval_ageb"].append(pd.NA)
                resultado["cells_ageb"].append(pd.NA)

        # filtrar filas con todos los valores nulos
        df_resultado = pd.DataFrame(resultado)
        columnas_a_validar = [col for col in df_resultado.columns if col.startswith('interval_') or col.startswith('cells_')]
        filas_nulas = df_resultado[columnas_a_validar].isna().all(axis=1)
        
        return df_resultado[~filas_nulas].reset_index(drop=True)
    
    
    def procesar_multiples_variables(self, escalas:list, dicc:dict, q:int=10) -> dict:
        """
        Categoriza múltiples variables y las procesa. 

        Se requiere un diccionario {base_porcentaje : lista_variables} que especifique qué variables se
        van a procesar, y si sus valores serán porcentajes, qué variable usarán como base.
        
        Args:
            escalas (list): Lista de escalas seleccionadas.
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
                - El valor de escalas no es una lista.
                - Las escalas seleccionadas no son "state", "mun" o "ageb".
                - q no es un entero mayor a 1.
                - El valor especificado para dicc no es un diccionario con al menos un elemento.
                - Una o más de las valores especificadas como llave del diccionario no son cadenas o None.
                - Uno o más de los valores del diccionario no son listas de cadenas.
                - Una o más de las variables especificadas en las listas no son cadenas.
        """
        if not isinstance(escalas, list):
            raise ValueError("El parámetro 'escala' debe ser una lista")
        if not (1 <= len(escalas) <= 3):
            raise ValueError("Debe especificar entre 1 y 3 escalas")
        escalas_validas = ["state", "mun", "ageb"]
        for escala in escalas:
            if escala not in escalas_validas:
                raise ValueError(f"La escala '{escala}' no es válida. Las opciones válidas son: 'state', 'mun' y 'ageb'")
        
        if not isinstance(q, int) or q < 1:
            raise ValueError("El valor para el parámetro 'q' debe ser un entero mayor a 1")
        
        if not isinstance(dicc, dict):
            raise ValueError("El valor para el parámetro 'dicc' debe ser un diccionario de forma {base_porcentaje : lista_variables}")
        if dicc == {}:
            raise ValueError("El diccionario no contiene items, debe ser un diccionario de forma {base_porcentaje : lista variables}")
        
        resultado = {}
        
        for base_porcentaje, lista in dicc.items():
        
            if base_porcentaje is not None:
                if not isinstance(base_porcentaje, str):
                    raise ValueError("Las llaves del diccionario deben ser cadenas o None")
                if not isinstance(lista, list):
                    raise ValueError(f"El valor asociado a la llave '{base_porcentaje}' debe ser una lista")

            variables_procesadas = []
            
            for var in lista:
            
                if not isinstance(var, str):
                    raise ValueError("La lista de variables debe incluir únicamente cadenas")
                
                variables_procesadas.append(self.procesar_variable(escalas=escalas, var=var, base_porcentaje=base_porcentaje, q=q))
                
            resultado[base_porcentaje] = pd.concat(variables_procesadas, ignore_index=True)
            
        return resultado
    
    
    def obtener_variables_regex(self, escala:str, regex:str) -> list:
        """
        Filtra todas las variables en el DataFrame correspondiente a la escala seleccionada que hagan match con la expresión regular dada.

        Args:
            escala (str): Escala seleccionada.
            regex (str): Expresión regular que se usará para filtrar variables.

        Returns:
            list: Lista con las variables filtradas.
            
        Raises:
                ValueError: Si los parámetros no cumplen con las validaciones:
                - El valor de escala no es una cadena.
                - La escala seleccionada no es "state", "mun" o "ageb".
                - No fue proporcionado a la clase el DataFrame necesario para la escala seleccionada.
                - El valor especificado para regex no es una cadena.
        """
        df = None
        if not isinstance(escala, str):
            raise ValueError("El parámetro 'escala' debe ser una cadena")
        if escala == "state":
            if self.df_state is None: raise ValueError("No es posible seleccionar la escala, pues no se ha proporcionado el DataFrame necesario")
            df = self.df_state
        elif escala == "mun":
            if self.df_mun is None: raise ValueError("No es posible seleccionar la escala, pues no se ha proporcionado el DataFrame necesario")
            df = self.df_mun
        elif escala == "ageb":
            if self.df_ageb is None: raise ValueError("No es posible seleccionar la escala, pues no se ha proporcionado el DataFrame necesario")
            df = self.df_ageb
        else:
            raise ValueError("El parámetro 'escala' debe ser: \"state\", \"mun\" o \"ageb\"")
        
        if not isinstance(regex, str):
            raise ValueError("El parámetro 'regex' debe ser una cadena")
        
        return list(df.filter(regex=regex).columns)
            
    
    def procesar_multiples_variables_regex(self, escalas:list, dicc:dict, excluidas:list=None, q:int=10) -> dict:
        """
        Categoriza múltiples variables especificadas por una expresión regular y las procesa. 
        
        Se requiere un diccionario {base_porcentaje : expresion_regular} que especifique qué variables se
        van a procesar, y si sus valores serán porcentajes, qué variable usarán como base.
        
        Args:
            escalas (list): Lista de escalas seleccionadas.
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
                - El valor de escalas no es una lista.
                - Las escala seleccionadas no son "state", "mun" o "ageb".
                - q no es un entero mayor a 1.
                - El valor especificado para dicc no es un diccionario.
                - El valor especificado para excluidas no es una lista de cadenas o None.
                - Una o más de las valores especificadas como llave del diccionario no son cadenas o None.
                - Uno o más de los valores del diccionario no son cadenas (que se utilizan como expresiones regulares).
                - Una o más de las cadenas especificadas como valores del diccionario no son expresiones regulares válidas.
        
        Raises:
            ValueError: Si los parámetros no cumplen con las validaciones:
                - El valor de escalas no es una lista.
                - Las escalas seleccionadas no son "state", "mun" o "ageb".
                - q no es un entero mayor a 1.
                - El valor especificado para dicc no es un diccionario con al menos un elemento.
                - Una o más de las valores especificadas como llave del diccionario no son cadenas o None.
                - Uno o más de los valores del diccionario no son listas de cadenas.
                - Una o más de las variables especificadas en las listas no son cadenas.
        """
        if not isinstance(escalas, list):
            raise ValueError("El parámetro 'escala' debe ser una lista")
        if not (1 <= len(escalas) <= 3):
            raise ValueError("Debe especificar entre 1 y 3 escalas")
        escalas_validas = ["state", "mun", "ageb"]
        for escala in escalas:
            if escala not in escalas_validas:
                raise ValueError(f"La escala '{escala}' no es válida. Las opciones válidas son: 'state', 'mun' y 'ageb'")
        
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
        
        resultado = {}
        
        for base_porcentaje, regex in dicc.items():
            
            if base_porcentaje is not None:
                if not isinstance(base_porcentaje, str):
                    raise ValueError("Las llaves del diccionario deben ser cadenas o None")
                
            if not isinstance(regex, str):
                raise ValueError(f"El valor asociado a la llave '{base_porcentaje}' no es una cadena")
            try:
                re.compile(regex)
            except re.error:
                raise ValueError(f"La expresión regular '{regex}' no es válida")
            
            variables_regex = set()
            if "state" in escalas:
                variables_regex = variables_regex | (
                    set(filter(lambda x: x not in excluidas, self.obtener_variables_regex("state", regex)))
                    if excluidas is not None else set(self.obtener_variables_regex("state", regex))
                )
            if "mun" in escalas:
                variables_regex = variables_regex | (
                    set(filter(lambda x: x not in excluidas, self.obtener_variables_regex("mun", regex)))
                    if excluidas is not None else set(self.obtener_variables_regex("mun", regex))
                )
            if "ageb" in escalas:
                variables_regex = variables_regex | (
                    set(filter(lambda x: x not in excluidas, self.obtener_variables_regex("ageb", regex)))
                    if excluidas is not None else set(self.obtener_variables_regex("ageb", regex))
                )

            if len(variables_regex) == 0:
                print(f"La expresión regular '{regex}' no coincide con ninguna variable")
                
            variables_procesadas = []
            
            for var in variables_regex:
                if var in self.variables_excluidas:
                    print(f"La expresión regular '{regex}' coincide con una variable ('{var}') que no tiene valores numéricos en los DataFrames de las escalas especificadas, excluyendo")
                    continue
                variables_procesadas.append(self.procesar_variable(escalas=escalas, var=var, base_porcentaje=base_porcentaje, q=q))
                
            resultado[base_porcentaje] = pd.concat(variables_procesadas, ignore_index=True) if len(variables_procesadas) > 0 else None
            
        return resultado
