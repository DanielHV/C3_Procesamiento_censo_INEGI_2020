from procesador_censo_2020 import *
import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Procesador Censo INEGI 2020')
    parser.add_argument('--config', type=str, required=True, help='Archivo de configuración')
    args = parser.parse_args()
    
    with open(args.config) as f:
        procesador_config = json.load(f)
    
    if 'data_state' not in procesador_config and 'data_mun' not in procesador_config and 'data_ageb' not in procesador_config:
        raise ValueError('El archivo JSON pasado para --config debe tener al menos una de las claves: data_state, data_mun o data_ageb')
    data_state = procesador_config.get('data_state', None)
    data_mun = procesador_config.get('data_mun', None)
    data_ageb = procesador_config.get('data_ageb', None)
    
    if 'dict_data' not in procesador_config:
        raise ValueError('El archivo JSON pasado para --config debe tener la clave dict_data')
    dict_data = procesador_config['dict_data']
    
    if 'variables_a_procesar_list' not in procesador_config and 'variables_a_procesar_regex' not in procesador_config:
        raise ValueError('El archivo JSON pasado para --config debe tener al menos una de las claves: variables_a_procesar_list, o variables_a_procesar_regex')
    variables_a_procesar_list = procesador_config.get('variables_a_procesar_list', None)
    variables_a_procesar_regex = procesador_config.get('variables_a_procesar_regex', None)
    
    if 'q' not in procesador_config:
        raise ValueError('El archivo JSON pasado para --config debe tener la clave q')
    q = procesador_config['q']
    
    if 'ruta_salida' not in procesador_config:
        raise ValueError('El archivo JSON pasado para --config debe tener la clave ruta_salida')
    ruta_salida = procesador_config['ruta_salida']

    diccionario = importar_diccionario_datos(dict_data)
    
    escalas = []
    df_state = None
    df_mun = None
    df_ageb = None
    if data_state is not None:
        df_state = importar_conjunto_datos(data_state, escala='state')
        escalas.append('state')
    if data_mun is not None:
        df_mun = importar_conjunto_datos(data_mun, escala='mun')
        escalas.append('mun')
    if data_ageb is not None:
        df_ageb = importar_conjunto_datos(data_ageb, escala='ageb')
        escalas.append('ageb')
    
    procesador = ProcesadorCenso2020(diccionario_variables=diccionario, df_state=df_state, df_mun=df_mun, df_ageb=df_ageb)
    
    if variables_a_procesar_list is not None:
        if 'None' in variables_a_procesar_list:
            variables_a_procesar_list[None] = variables_a_procesar_list.pop('None')
        procesamiento_listas_dict = procesador.procesar_multiples_variables(escalas=escalas, dicc=variables_a_procesar_list, q=q)
        procesamiento_listas = pd.concat(list(procesamiento_listas_dict.values()))
        
    if variables_a_procesar_regex is not None:
        if 'None' in variables_a_procesar_regex:
            variables_a_procesar_regex[None] = variables_a_procesar_regex.pop('None')
        procesamiento_regex_dict = procesador.procesar_multiples_variables_regex(escalas=escalas, dicc=variables_a_procesar_regex, q=q)
        procesamiento_regex = pd.concat(list(procesamiento_regex_dict.values()))

    if variables_a_procesar_list and variables_a_procesar_regex:
        resultado = pd.concat([procesamiento_regex, procesamiento_listas])
    elif variables_a_procesar_list:
        resultado = procesamiento_listas
    elif variables_a_procesar_regex:
        resultado = procesamiento_regex
        
    duplicados = resultado.duplicated(['code', 'bin'])

    if duplicados.any():
        print('Advertencia: el resultado del procesamiento contiene categorias duplicadas:')
        print(resultado.loc[duplicados, ['code', 'bin']])
        
    resultado.to_csv(ruta_salida, index=False)
