import psycopg
import os
from dotenv import load_dotenv
from io import StringIO

class GestorCargaDatos:
    
    def __init__(self, env_path=".env"):
        self.env_path = env_path
        with self.__conectar() as conn:
            info = conn.info
            print(f"Conexión verificada exitosamente")

    def __conectar(self):
        load_dotenv(self.env_path, override=True)
        conn = psycopg.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
        return conn

    def ejecutar_comando(self, comando):
        try:
            with self.__conectar() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(comando)
                    conn.commit()
                    print("Comando ejecutado exitosamente")
        except Exception as e:
            print(f"Error al ejecutar el comando: {e}")
 
    def ejecutar_consulta(self, consulta):
        try:
            with self.__conectar() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(consulta)
                    resultados = cursor.fetchall()
                    return resultados
        except Exception as e:
            print(f"Error al ejecutar la consulta: {e}")
            return None

    def cargar_dataframe_a_tabla(self, df, tabla):

        with self.__conectar() as conn:
            with conn.cursor() as cursor:
                # archivo temporal en memoria
                buffer = StringIO()
                df.to_csv(buffer, index=False, header=True)
                buffer.seek(0)  # volver al inicio del archivo

                columnas = ", ".join([f'"{col}"' for col in df.columns])
                with cursor.copy(f"COPY {tabla} ({columnas}) FROM STDIN WITH CSV HEADER") as copy:
                    copy.write(buffer.getvalue())
                
                conn.commit()
                print(f"Datos insertados exitosamente en la tabla '{tabla}'")
